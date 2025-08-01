# %%
import string
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from pareto_plot_nuisance import ParetoGroup, pareto_plot, pareto_plot_sweep

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction, LossFunctionsDict
from scatteringlibs.matplotlibmagic import (
    PlotGroup,
    make_leftax_look_like_inset,
)
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

sys.exit()

# %% Loading in data
project_name = "quasirandom_longrun"
data_folder = f"../data/{project_name}"

sim = CoupledDipoleSimulator.load(folder=data_folder)
fisher_calc = FisherMatrix.load(folder=data_folder)
loss_functions_dict = LossFunctionsDict.load(folder=data_folder)
loss_functions_sweep_dict = LossFunctionsDict.load(
    name="LossFunctionsSweepDict", folder=data_folder
)

print("Lossfunction Names")
for lf_name in loss_functions_dict.keys():
    print(f"\t- {lf_name}")

# %%
for loss_function in loss_functions_dict.values():
    loss_function._J *= 4
# %% Plotting parameters
DPI = 300
FONTSIZE = 10
TITLE_FONTSIZE = 14
YLABEL_FONTSIZE = 12
SUBPLOT_ANNOTATION_FONTSIZE = 24

GRID_KWARGS = dict(which="both", linestyle=":")

save_plots = True
save_folder = Path("../figures")


def annotate_subplots_abc(axes, format_str="{})"):
    """
    Generates a), b), etc. in the corner of a subplot.

    """
    for i in range(len(axes)):
        label = string.ascii_lowercase[i]
        axes[i].annotate(
            format_str.format(label),
            xy=(0, 1),
            xycoords="axes fraction",
            textcoords="offset fontsize",
            xytext=(+0.5, -0.5),
            verticalalignment="top",
            fontsize=SUBPLOT_ANNOTATION_FONTSIZE,
            bbox=dict(facecolor="w", edgecolor="none", pad=3.0),
        )


# %% 1) Bar plot of CRLBs


barplot_lossfunctions = [
    "x",
    "z",
    "phi",
    # "tr",
    # "tr_norm",
    "det",
    # "tr_inv",
    "spec_rad",
    "dropped_spec_rad",
    "partial_spec_rad",
]

loss_functions_dict_bar = loss_functions_dict.select_only(
    barplot_lossfunctions
)

optimized_crlbs = crlbanalysis.compute_crlb_for_optimized(
    loss_functions_dict_bar
)
plane_wave_crlbs = crlbanalysis.compute_crlb_for_planewaves(
    sim_kwargs=sim.get_kwargs(),
    F_operator_tensor=fisher_calc.F,
    photon_budget=1.0,
    kx_array=sim.k_x_array,
)
random_wavefronts_crlbs = crlbanalysis.compute_crlb_for_random_wavefronts(
    F_operator_tensor=fisher_calc.F,
    photon_budget=1.0,
    num_samples=100,
)


optimized_crlbs_joint, optimized_crlbs_scalar = optimized_crlbs
plane_wave_crlbs_joint, plane_wave_crlbs_scalar = plane_wave_crlbs
random_wavefronts_crlbs_joint, random_wavefronts_crlbs_scalar = (
    random_wavefronts_crlbs
)

loss_function_names = loss_functions_dict_bar.displaynames()

fig, axes = plt.subplots(
    ncols=2,
    figsize=(14, 6),
    width_ratios=[len(loss_function_names) + 1, 2 + 1],
    dpi=DPI,
    layout="constrained",
)


crlbanalysis.visualize_crlb_comparison(
    loss_function_names=loss_function_names,
    param_legends=[
        "CRLB of x",
        "CRLB of z",
        r"CRLB of $\phi$",
    ],
    optimized_crlbs_joint=optimized_crlbs_joint,
    optimized_crlbs_scalar=optimized_crlbs_scalar,
    show_plane_waves=False,
    show_random_wavefronts=False,
    show_optimized=True,
    ax=axes[0],
    title="Optimized Wavefronts",
    fontsize=FONTSIZE,
    title_fontsize=TITLE_FONTSIZE,
    ylabel_fontsize=YLABEL_FONTSIZE,
    ylim=(-0.000, 0.0185),
    hatch_color=(0.8,) * 3,
    joint_opacity=0.65,
    y_label=r"CRLB (wavelengths)",
)

crlbanalysis.visualize_crlb_comparison(
    loss_function_names=loss_function_names,
    param_legends=[
        "CRLB of x",
        "CRLB of z",
        r"CRLB of $\phi$",
    ],
    optimized_crlbs_joint=optimized_crlbs_joint,
    # optimized_crlbs_scalar=optimized_crlbs_scalar,
    plane_wave_crlbs_joint=plane_wave_crlbs_joint,
    # plane_wave_crlbs_scalar=plane_wave_crlbs_scalar,
    random_wavefronts_crlbs_joint=random_wavefronts_crlbs_joint,
    # random_wavefronts_crlbs_scalar=random_wavefronts_crlbs_scalar,
    show_optimized=False,
    show_plane_waves=True,
    show_random_wavefronts=True,
    ax=axes[1],
    y_label=None,
    ylim=(-0.002, None),
    title="Plane Waves and Random Wavefronts",
    fontsize=FONTSIZE,
    title_fontsize=TITLE_FONTSIZE,
    ref_lines=[0, 0.0185],
)
annotate_subplots_abc(axes)

make_leftax_look_like_inset(
    fig, leftax=axes[0], rightax=axes[1], moveticksright=True
)
fig.set_constrained_layout_pads(wspace=0.1)

if save_plots:
    plt.savefig(save_folder / f"{project_name}_crlb_barplot.png")
    plt.savefig(save_folder / f"{project_name}_crlb_barplot.pdf")
    print(f"Saved CRLB bar plot to '{data_folder}'")
plt.show()


# %% 2) Pareto plot

"""
Select the different loss functions (optimality criteria) from the 
loss_functions_dict
"""

lf_dict_single_param = loss_functions_dict.select_only(["x", "z"])
lf_dict_dropped = loss_functions_dict.select_only(
    ["dropped_det", "dropped_spec_rad", "dropped_tr_inv"]
)
lf_dict_full = loss_functions_dict.select_only(
    ["det", "spec_rad", "tr_inv", "x_inv", "z_inv"]
)
lf_dict_partial = loss_functions_dict.select_only(
    [
        "partial_det",
        "partial_spec_rad",
        "partial_tr_inv",
    ]
)


fig, ax = plt.subplots(figsize=(7, 5), dpi=DPI)

"""
Control the formatting of the different groups here.
"""
pareto_groups = [
    PlotGroup(
        lf_dict_single_param,
        legend_label="Single-param. optimality",
        color="purple",
        marker=".",
    ),
    PlotGroup(
        lf_dict_dropped,
        legend_label="Two-param. optimality",
        color="red",
        marker="x",
    ),
    PlotGroup(
        lf_dict_full,
        legend_label="Three-param. optimality",
        color="green",
        marker="^",
    ),
    PlotGroup(
        lf_dict_partial,
        legend_label="Two-param. optimality (partial)",
        color="blue",
        marker="s",
    ),
]
pareto_groups_sweep = [
    PlotGroup(
        loss_functions_sweep_dict.select_only("xz_joint"),
        legend_label="Single-param. optimality",
        color="green",
        linestyle="",
        # marker=".",
    ),
    PlotGroup(
        loss_functions_sweep_dict.select_only("xz_scalar"),
        legend_label="Two-param. optimality",
        color="purple",
        linestyle="",
        # marker="x",
    ),
]

x_idx = 1  # Display Z on x-axis
y_idx = 0  # Display X on y-axis

pareto_plot_sweep(pareto_groups_sweep, x_idx, y_idx, ax=ax)

pareto_plot(
    pareto_groups,
    x_idx,
    y_idx,
    xlabel=r"CRLB for $z$ (wavelenghts)",
    ylabel=r"CRLB for $x$ (wavelenghts)",
    title=r"Nuisance Aware Optimality Criteria",
    ax=ax,
    grid_kwargs=GRID_KWARGS,
    use_logscale=False,
    adjust_text_kwargs=dict(
        force_static=0.1, force_text=0.9, expand=(1.1, 1.1), min_arrow_len=40
    ),
)


if save_plots:
    fig_name = f"{project_name}_pareto_plot_nuisance"
    plt.savefig(save_folder / f"{fig_name}.png")
    plt.savefig(save_folder / f"{fig_name}.pdf")
    print(f"Saved pareto plot to '{fig_name}'")

plt.show()


# %%


DPI = 300
sim = CoupledDipoleSimulator.load(folder=data_folder)
sim_kwargs = sim.get_kwargs()
# sim_kwargs["grid_points_x"] = 40
# sim_kwargs["k_x_array"] = None
# sim_kwargs["theta_inc_deg"] = np.linspace(-70, 70, num=20)

sim = CoupledDipoleSimulator(**sim_kwargs)
E_kx_array = np.zeros_like(sim.theta_inc)
E_kx_array[0] = 1
sim.set_input_field_amplitudes(E_kx_array, flux_normalize=True)

fig, ax = sim.visualize_intensity(dpi=DPI)

n_frames = 2 * len(E_kx_array) - 2


def init():
    pass


def update(i):
    global E_kx_array
    print(f"\rAnimating frame {i+1:>2} / {n_frames} ", end="")
    # Boomerang effect, quite wasteful to redo the computations but oh well
    # i = i + 1
    i = min(2 * len(E_kx_array) - 2 - i, i)
    E_kx_array *= 0
    E_kx_array[i] = 1
    sim.set_input_field_amplitudes(E_kx_array)

    sim.visualize_intensity(dpi=DPI, ax=ax, title="Field Intensity Over Slab")


ani = matplotlib.animation.FuncAnimation(
    fig, update, init_func=init, frames=n_frames
)

if save_plots:
    fig_name = f"{project_name}_scattering_angles_animation"
    ani.save(save_folder / f"{fig_name}.gif")
    ani.save(save_folder / f"{fig_name}.mp4")

plt.show()
