# %%
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from adjustText import adjust_text

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction, LossFunctionsDict
from scatteringlibs.matplotlibmagic import (
    PlotGroup,
    get_bounding_boxes_scatter_plot,
)
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

# sys.exit()

# %% Loading data from disk
# data_folder = "../data/quasirandom"

# sim = CoupledDipoleSimulator.load(folder=data_folder)
# fisher_calc = FisherMatrix.load(folder=data_folder)
# loss_functions_dict = LossFunctionsDict.load(folder=data_folder)
# loss_functions_sweep_dict = LossFunctionsDict.load(
#     name="LossFunctionsSweepDict", folder=data_folder
# )

# %%


class ParetoGroup:
    def __init__(
        self,
        loss_functions_dict: LossFunctionsDict,
        legend_label: str,
        marker: str = None,
        color: str = None,
    ):
        self.loss_functions_dict = loss_functions_dict
        self.legend_label = legend_label
        self.marker = marker
        self.color = color


adjust_text_default_dict = dict(
    # explode=(1.2, 1.2),
    # only_move={
    #     "text": "xy",
    #     "explode": "x",
    #     "static": "xy",
    # },  # allow text to move in both directions, points only in y
    arrowprops=dict(arrowstyle="-", lw=0.5, alpha=1.0, zorder=0),
    # min_arrow_len=20,
    # expand=(1.5, 1.3),  # how much to "inflate" the text boxes,
    # avoid_self=False,
    # force_text=0.5,  # strength of repulsion
    # force_static=0.5,
    # force_pull=0.1,
    # lim=10,  # max iterations
    # expand_axes=True,
)


def pareto_plot(
    pareto_groups: list[PlotGroup],
    x_idx,
    y_idx,
    grid_kwargs: dict = {"visible": False},
    adjust_text_kwargs: dict = {},
    ax=None,
    dpi=100,
    use_logscale: bool = False,
    title: str = "Cram'er-Rao Lower Bounds",
    xlabel: str = None,
    ylabel: str = None,
    x_avoid: list = [],
    y_avoid: list = [],
    xlim=None,
    ylim=None,
    wavelength_unit: float = 1.0,
    photon_budget: float = 1.0,
):
    new_fig = ax is None
    if new_fig:
        fig, ax = plt.subplots(dpi=dpi)

    all_texts = []
    all_points = []

    for pareto_group in pareto_groups:
        loss_functions_dict = pareto_group.data
        crlbs_joint, _ = crlbanalysis.compute_crlb_for_optimized(
            loss_functions_dict
        )

        texts, points = pareto_plot_single(
            loss_functions_dict.shortnames(),
            x_data=crlbs_joint[:, x_idx]
            * wavelength_unit
            / np.sqrt(photon_budget),
            y_data=crlbs_joint[:, y_idx]
            * wavelength_unit
            / np.sqrt(photon_budget),
            marker=pareto_group.marker,
            legend_label=pareto_group.legend_label,
            color=pareto_group.color,
            ax=ax,
        )
        all_texts += texts
        all_points.append(points)
        x_avoid += list(crlbs_joint[:, x_idx].flat)
        y_avoid += list(crlbs_joint[:, y_idx].flat)

    bboxes = []
    for point in all_points:
        # print(getbb(all_points[i], ax))
        bboxes += get_bounding_boxes_scatter_plot(point, ax)

    adjust_text_kwargs_final = {
        **adjust_text_default_dict,
        **adjust_text_kwargs,
    }

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(**grid_kwargs)

    ax.set_title(title)

    if use_logscale:
        plt.xscale("log")
        plt.yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    labels_unique = dict(zip(labels, handles))
    # print(handles, labels, labels_unique)
    ax.legend(labels_unique.values(), labels_unique.keys())
    # print(np.array(x_avoid), np.array(y_avoid))
    adjust_text(
        all_texts,
        x=x_avoid,
        y=y_avoid,
        ax=ax,
        # objects=bboxes,
        **adjust_text_kwargs_final
    )
    plt.tight_layout()

    if new_fig:
        plt.show()


def pareto_plot_single(
    point_labels,
    x_data,
    y_data,
    legend_label: str,
    xmax=None,
    ymax=None,
    dpi=100,
    ax=None,
    marker: str = "x",
    color: str = "k",
):
    new_fig = ax is None
    if new_fig:
        fig, ax = plt.subplots(dpi=dpi)

    points = []
    texts = []

    points = ax.scatter(
        x_data,
        y_data,
        s=50,
        marker=marker,
        color=color,
        label=legend_label,
        zorder=10,
    )

    for point_label, x, y in zip(point_labels, x_data, y_data):
        # loss_function, data = result

        # if loss_function.name in p_front_names:
        #     color = "r"
        #     label = "Efficient"

        # x_out_bounds = not xmax is None and x > xmax
        # y_out_bounds = not ymax is None and y > ymax

        # if x_out_bounds:
        #     x = xmax
        #     marker = matplotlib.markers.CARETRIGHT
        #     color = "black"
        #     label = None
        # if y_out_bounds:
        #     y = ymax
        #     marker = matplotlib.markers.CARETUP
        #     color = "black"
        #     label = None
        # if x_out_bounds and y_out_bounds:
        #     marker = "D"

        text = ax.annotate(
            point_label,
            (x, y),
            fontsize=8,
            color=color,
            bbox=dict(
                boxstyle="round, pad=0.1", fc="white", ec="none", alpha=0.8
            ),
            ha="center",
            va="center",
            zorder=100,
        )
        #     points.append(point)
        texts.append(text)

    if new_fig:
        plt.show()

    return texts, points


def pareto_plot_sweep(
    pareto_groups: list[PlotGroup],
    x_idx,
    y_idx,
    grid_kwargs: dict = {"visible": False},
    adjust_text_kwargs: dict = {},
    ax=None,
    dpi=100,
    use_logscale: bool = False,
    title: str = "Cram'er-Rao Lower Bounds",
    xlabel: str = None,
    ylabel: str = None,
    wavelength_unit: float = 1.0,
    photon_budget: float = 1.0,
):
    new_fig = ax is None
    if new_fig:
        fig, ax = plt.subplots(dpi=dpi)

    all_texts = []
    all_points = []

    x_avoid = []
    y_avoid = []
    for pareto_group in pareto_groups:
        crlbs_sweep_joint = []
        for loss_function in pareto_group.data.values():
            crlbs_sweep_joint_cur = []
            for J in loss_function.Js:
                crlbs_sweep_joint_cur.append(
                    crlbanalysis.compute_joint_crlb_from_J(J)
                )
            crlbs_sweep_joint.append(crlbs_sweep_joint_cur)
        # crlbs_joint, _ = crlbanalysis.compute_crlb_for_optimized(
        #     pareto_group.loss_functions_dict
        # )
        crlbs_sweep_joint = np.array(crlbs_sweep_joint)

        x_data = (
            crlbs_sweep_joint[:, :, x_idx].T
            * wavelength_unit
            / np.sqrt(photon_budget)
        )
        y_data = (
            crlbs_sweep_joint[:, :, y_idx].T
            * wavelength_unit
            / np.sqrt(photon_budget)
        )

        texts, points = pareto_plot_sweep_single(
            # pareto_group.loss_functions_dict.shortnames(),
            x_data=x_data,
            y_data=y_data,
            linestyle=pareto_group.linestyle,
            legend_label=pareto_group.legend_label,
            color=pareto_group.color,
            ax=ax,
            marker=pareto_group.marker,
        )
        all_texts += texts
        all_points.append(points)
        x_avoid += list(x_data.flat)
        y_avoid += list(y_data.flat)

    # bboxes = []
    # for point in all_points:
    #     # print(getbb(all_points[i], ax))
    #     bboxes += get_bounding_boxes_scatter_plot(point, ax)

    adjust_text_kwargs_final = {
        **adjust_text_default_dict,
        **adjust_text_kwargs,
    }

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(**grid_kwargs)

    ax.set_title(title)

    if use_logscale:
        plt.xscale("log")
        plt.yscale("log")

    ax.legend()

    adjust_text(
        all_texts,
        x=x_avoid,
        y=y_avoid,
        ax=ax,
        # objects=bboxes,
        **adjust_text_kwargs_final
    )
    plt.tight_layout()

    if new_fig:
        plt.show()
    return x_avoid, y_avoid


def pareto_plot_sweep_single(
    x_data,
    y_data,
    legend_label: str,
    xmax=None,
    ymax=None,
    dpi=100,
    ax=None,
    linestyle: str = ":",
    color: str = "k",
    marker: str = None,
):
    new_fig = ax is None
    if new_fig:
        fig, ax = plt.subplots(dpi=dpi)

    points = []
    texts = []
    points = ax.plot(
        x_data,
        y_data,
        # s=50,
        linestyle=linestyle,
        color=color,
        label=legend_label,
        zorder=9,
        marker=marker,
    )

    # for point_label, x, y in zip(point_labels, x_data, y_data):
    #     # loss_function, data = result

    #     # if loss_function.name in p_front_names:
    #     #     color = "r"
    #     #     label = "Efficient"

    #     # x_out_bounds = not xmax is None and x > xmax
    #     # y_out_bounds = not ymax is None and y > ymax

    #     # if x_out_bounds:
    #     #     x = xmax
    #     #     marker = matplotlib.markers.CARETRIGHT
    #     #     color = "black"
    #     #     label = None
    #     # if y_out_bounds:
    #     #     y = ymax
    #     #     marker = matplotlib.markers.CARETUP
    #     #     color = "black"
    #     #     label = None
    #     # if x_out_bounds and y_out_bounds:
    #     #     marker = "D"

    #     text = ax.annotate(
    #         point_label,
    #         (x, y),
    #         fontsize=8,
    #         color=color,
    #         bbox=dict(
    #             boxstyle="round, pad=0.1", fc="white", ec="none", alpha=1.0
    #         ),
    #         ha="center",
    #         va="center",
    #         zorder=100,
    #     )
    #     #     points.append(point)
    #     texts.append(text)

    if new_fig:
        plt.show()

    return texts, points
