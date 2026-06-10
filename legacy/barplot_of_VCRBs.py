# %%
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction, LossFunctionsDict
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

sys.exit()

# %%
data_folder = "../data/quasirandom"

sim = CoupledDipoleSimulator.load(folder=data_folder)
fisher_calc = FisherMatrix.load(folder=data_folder)
loss_functions_dict = LossFunctionsDict.load(folder=data_folder)

# %%

# crlbanalysis.visualize_crlb_comparison(
#     optimized_wavefronts=optimized_wavefronts,
#     F_operator_tensor=optimizer.F_op,
#     photon_budget=photon_budget,
#     param_names=param_names,
#     sim_kwargs=sim_kwargs,
#     kx_array=kx_array,
#     figure_title="CRLB Comparison for different types of Maximum Information States",
#     show_plane_waves=True,
#     show_random_wavefronts=True,
#     num_random_samples=2000,
# )
from matplotlib.transforms import Affine2D


class DoubleHatchHandler(matplotlib.legend_handler.HandlerPatch):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        # 1) pull style from the real rectangle
        rect = orig_handle.patches[0]
        hatch = rect.get_hatch() or ""

        # Build new rectangle with double hatch
        p = Rectangle(
            (x0, y0),
            width,
            height,
            facecolor=rect.get_facecolor(),
            hatch=hatch * 2,
        )

        return [p]


def visualize_crlb_comparison(
    loss_function_names: list,
    param_legends: list,
    optimized_crlbs_joint: np.ndarray = None,
    optimized_crlbs_scalar: np.ndarray = None,
    plane_wave_crlbs_joint: np.ndarray = None,
    random_wavefronts_crlbs_joint: np.ndarray = None,
    title: str = "CRLB Comparison",
    show_optimized: bool = True,
    show_plane_waves: bool = True,
    show_random_wavefronts: bool = True,
    num_random_samples: int = 100,
    ax: matplotlib.axes._axes.Axes = None,
    bar_width: float = 0.2,
    y_label: str = "Standard Deviation (CRLB)",
    joint_opacity: float = 0.7,
    hatches: tuple = ("/", "\\", "x"),
    fontsize: int = 10,
    title_fontsize: int = 12,
    ylabel_fontsize: int = 14,
):
    """
    Visualizes and compares Cramer-Rao Lower Bounds (CRLBs) for various input wavefronts.
    Adapted from original script's block 9.

    Args:
        optimized_wavefronts (dict): Wavefronts to plot (name:coeffs). Coeffs are numpy arrays.
        F_operator_tensor (np.ndarray): Fisher operator tensor (numpy format).
        photon_budget (float): Photon budget for normalization.
        param_names (list): Names of the parameters being estimated.
        sim_kwargs (dict, optional): Simulation keyword arguments, for plane waves.
        angles_deg (np.ndarray, optional): Array of angles in degrees, for plane waves.
        kx_array (np.ndarray, optional): Array of kx values, for plane waves & N_in.
        figure_title (str, optional): Title for the plot.
        show_plane_waves (bool, optional): Whether to compute and show plane wave CRLBs.
        show_random_wavefronts (bool, optional): Whether to compute and show random wavefront CRLBs.
        num_random_samples (int, optional): Number of random wavefronts to sample.
    """

    num_params = len(param_legends)
    param_legends_joint = ["joint " + legend for legend in param_legends]
    param_legends_scalar = ["scalar " + legend for legend in param_legends]

    # --- Plotting ---
    num_optimized_wavefronts = len(loss_function_names)
    num_boxplot_groups = 0
    if show_plane_waves:
        num_boxplot_groups += 1
    if show_random_wavefronts:
        num_boxplot_groups += 1

    new_fig = ax is None
    if new_fig:
        # Adjust figure width based on number of boxplot groups
        fig_width = (
            7 + 2.5 * num_boxplot_groups + 0.5 * num_optimized_wavefronts
        )
        fig, ax = plt.subplots(
            figsize=(max(10, fig_width), 7)
        )  # Ensure a minimum width

    cur_x_position = 0
    xtick_positions = np.array([], dtype=float)
    xtick_labels = []

    # Colors for parameters (e.g., CRLB_x, CRLB_z, CRLB_theta)
    # Ensure enough colors if num_params > 3
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    param_colors = [colors[i % len(colors)] for i in range(num_params)]

    # plt.rcParams['hatch.color'] = (0.25,) * 3

    if show_optimized and optimized_crlbs_joint is not None:

        x_positions_optimized = cur_x_position + np.arange(
            num_optimized_wavefronts
        )
        xtick_positions = np.append(xtick_positions, x_positions_optimized)
        cur_x_position += len(x_positions_optimized)
        xtick_labels += list(loss_function_names)

        joint_alpha = (
            joint_opacity if optimized_crlbs_scalar is not None else 1.0
        )

        # for i, optimized_crlb_joint in enumerate(optimized_crlbs_joint):
        for p in range(num_params):
            offset = (p - (num_params - 1) / 2) * bar_width
            ax.bar(
                x_positions_optimized + offset,
                optimized_crlbs_joint[:, p],
                width=bar_width,
                label=param_legends_joint[p],
                color=param_colors[p],
                alpha=joint_alpha,
                hatch=hatches[p],
            )

    if show_optimized and optimized_crlbs_scalar is not None:
        # for i, optimized_crlb_scalar in enumerate(optimized_crlbs_scalar):
        for p in range(num_params):
            offset = (p - (num_params - 1) / 2) * bar_width
            ax.bar(
                x_positions_optimized + offset,
                optimized_crlbs_scalar[:, p],
                width=bar_width,
                label=param_legends_scalar[p],
                color=param_colors[p],
                alpha=1.0,
                hatch=hatches[p],
            )

    def plot_reference_lines(
        ref_values, positions, label, bar_width=bar_width
    ):
        """
        Show reference values of optimized wavefronts to show scale. Here we
        picked D-optimality.
        """
        for p in range(num_params):
            ax.hlines(
                ref_values[p],
                xmin=positions[p] - bar_width / 2,
                xmax=positions[p] + bar_width / 2,
                color="r",
                linestyle="--",
                label="D-optimality",
            )

    # Boxplots for plane waves
    if show_plane_waves and plane_wave_crlbs_joint is not None:
        # current_xtick_pos += 1.0  # Spacing before boxplot group
        box_positions_plane = [
            cur_x_position + (i - (num_params - 1) / 2) * bar_width * 1.2
            for i in range(num_params)
        ]

        xtick_positions = np.append(
            xtick_positions, np.mean(box_positions_plane)
        )
        cur_x_position += 1

        bp_plane = ax.boxplot(
            plane_wave_crlbs_joint,
            positions=box_positions_plane,
            widths=bar_width * 1.1,  # Slightly wider boxes
            patch_artist=True,  # Fill with color
            showfliers=False,  # Hide outliers for cleaner plot
            medianprops={"color": "black", "linewidth": 1.5},
            label=param_legends_joint,
        )
        for patch, color, hatch in zip(
            bp_plane["boxes"], param_colors, hatches
        ):
            patch.set_facecolor(color)
            patch.set_alpha(joint_opacity)
            patch.set_hatch(hatch)

        xtick_labels.append("Plane Waves")

        # Show largest CRLB of optimized wavefronts as dashed line:

        optimized_det = optimized_crlbs_joint[
            loss_function_names.index("Determinant")
        ]
        plot_reference_lines(
            optimized_det, box_positions_plane, "D-optimality"
        )

    # Boxplots for plane waves
    if show_random_wavefronts and random_wavefronts_crlbs_joint is not None:
        # current_xtick_pos += 1.0  # Spacing before boxplot group
        box_positions_random = [
            cur_x_position + (i - (num_params - 1) / 2) * bar_width * 1.2
            for i in range(num_params)
        ]
        xtick_positions = np.append(
            xtick_positions, np.mean(box_positions_random)
        )
        cur_x_position += 1

        bp_random = ax.boxplot(
            random_wavefronts_crlbs_joint,
            positions=box_positions_random,
            widths=bar_width * 1.1,  # Slightly wider boxes
            patch_artist=True,  # Fill with color
            showfliers=False,  # Hide outliers for cleaner plot
            medianprops={"color": "black", "linewidth": 1.5},
            label=param_legends_joint,
        )
        for patch, color, hatch in zip(
            bp_random["boxes"], param_colors, hatches
        ):
            patch.set_facecolor(color)
            patch.set_alpha(joint_opacity)
            patch.set_hatch(hatch)

        xtick_labels.append(f"Random WFs\n({num_random_samples} samples)")

        # optimized_max = np.max(optimized_crlbs_joint, axis=0)
        optimized_det = optimized_crlbs_joint[
            loss_function_names.index("Determinant")
        ]
        plot_reference_lines(
            optimized_det, box_positions_random, "D-optimality"
        )

    # Final plot adjustments
    ax.set_xticks(xtick_positions)
    # Rotate labels only if there are many or they are long
    if len(xtick_labels) > 5 or any(len(label) > 10 for label in xtick_labels):
        ax.set_xticklabels(
            xtick_labels, rotation=25, ha="right", fontsize=fontsize
        )
    else:
        ax.set_xticklabels(xtick_labels, fontsize=fontsize)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel(y_label, fontsize=ylabel_fontsize)
    ax.grid(True, linestyle=":", alpha=1.0)
    ax.set_ylim(bottom=0)

    # Prevent label duplicates
    handles, labels = ax.get_legend_handles_labels()
    labels_unique = dict(zip(labels, handles))
    ncols_legend = 2 if show_optimized else 1
    ax.legend(
        labels_unique.values(),
        labels_unique.keys(),
        title="Parameters",
        fontsize=fontsize,
        ncols=ncols_legend,
        handler_map={
            matplotlib.container.BarContainer: DoubleHatchHandler()
        },  # increase density of hatches
    )
    if new_fig:
        plt.tight_layout()
        plt.show()


include_lossfunctions = [
    "x",
    "z",
    "theta",
    "tr",
    "tr_norm",
    "det",
    "spec_rad",
    "tr_inv",
]

loss_functions_dict_plotting = {
    key: value
    for key, value in loss_functions_dict.items()
    if key in include_lossfunctions
}


optimized_crlbs = crlbanalysis.compute_crlb_for_optimized(
    loss_functions_dict_plotting
)
plane_wave_crlbs = crlbanalysis.compute_crlb_for_planewaves(
    sim_kwargs=sim.get_kwargs(),
    F_operator_tensor=fisher_calc.F,  # Expects numpy array
    photon_budget=1.0,
    kx_array=sim.k_x_array,
)
random_wavefronts_crlbs = crlbanalysis.compute_crlb_for_random_wavefronts(
    F_operator_tensor=fisher_calc.F,  # Expects numpy array
    photon_budget=1.0,
    num_samples=100,
)


optimized_crlbs_joint, optimized_crlbs_scalar = optimized_crlbs
plane_wave_crlbs_joint, plane_wave_crlbs_scalar = plane_wave_crlbs
random_wavefronts_crlbs_joint, random_wavefronts_crlbs_scalar = (
    random_wavefronts_crlbs
)

loss_function_names = [
    lf.displayname for lf in loss_functions_dict_plotting.values()
]

fig, axes = plt.subplots(
    ncols=2,
    figsize=(14, 6),
    width_ratios=[len(loss_function_names) + 1, 2 + 1],
    dpi=100,
    layout="constrained",
)
for i, label in enumerate("ab"):
    axes[i].annotate(
        label + ")",
        xy=(0, 1),
        xycoords="axes fraction",
        textcoords="offset fontsize",
        xytext=(+0.5, -0.5),
        verticalalignment="top",
        fontsize=24,
        bbox=dict(facecolor="w", edgecolor="none", pad=3.0),
    )


visualize_crlb_comparison(
    loss_function_names=loss_function_names,
    param_legends=[
        "CRLB of x",
        "CRLB of z",
        r"CRLB of $\phi$",
    ],
    optimized_crlbs_joint=optimized_crlbs_joint,
    optimized_crlbs_scalar=optimized_crlbs_scalar,
    # plane_wave_crlbs_joint=plane_wave_crlbs_joint,
    # plane_wave_crlbs_scalar=plane_wave_crlbs_scalar,
    # random_wavefronts_crlbs_joint=random_wavefronts_crlbs_joint,
    # random_wavefronts_crlbs_scalar=random_wavefronts_crlbs_scalar,
    show_plane_waves=False,
    show_random_wavefronts=False,
    show_optimized=True,
    ax=axes[0],
    title="CRLB for optimized wavefronts",
)

visualize_crlb_comparison(
    loss_function_names=loss_function_names,
    param_legends=[
        "CRLB of x",
        "CRLB of z",
        r"CRLB of $\phi$",
    ],
    optimized_crlbs_joint=optimized_crlbs_joint,
    # optimized_crlbs_scalar=optimized_crlbs_scalar,
    plane_wave_crlbs_joint=plane_wave_crlbs_joint,
    plane_wave_crlbs_scalar=plane_wave_crlbs_scalar,
    random_wavefronts_crlbs_joint=random_wavefronts_crlbs_joint,
    random_wavefronts_crlbs_scalar=random_wavefronts_crlbs_scalar,
    show_optimized=False,
    show_plane_waves=True,
    show_random_wavefronts=True,
    ax=axes[1],
    y_label=None,
    title="CRLB for plane wave and random wavefronts",
)
plt.show()
