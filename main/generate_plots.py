# %%
import string
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes._axes import Axes
from pareto_plot_nuisance import ParetoGroup, pareto_plot, pareto_plot_sweep

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction, LossFunctionsDict
from scatteringlibs.matplotlibmagic import (
    PlotGroup,
    make_leftax_look_like_inset,
    register_turbo,
)

register_turbo()
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

sys.exit()

# %% Loading in data
project_name = "quasirandom_longrun"
# project_name = "quasirandom"
# project_name = "square"
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


# %% Plotting parameters
DPI = 300
TEXTWIDTH = 5.90666  # inches
FONTSIZE = 10
TITLE_FONTSIZE = 10
AXES_LABEL_FONTSIZE = 8
LEGEND_FONTSIZE = 8
TICK_FONTSIZE = 6
SUBPLOT_ANNOTATION_FONTSIZE = 14

GRID_KWARGS = dict(which="both", linestyle=":")

save_plots = False
save_folder = Path("../figures")

plt.rcdefaults()
plt.rcParams.update(
    {
        "font.size": FONTSIZE,  # Main text
        "axes.labelsize": AXES_LABEL_FONTSIZE,  # Axis labels
        "legend.fontsize": LEGEND_FONTSIZE,  # Legend
        "xtick.labelsize": TICK_FONTSIZE,  # X-tick labels
        "ytick.labelsize": TICK_FONTSIZE,  # Y-tick labels
        "axes.titlesize": TITLE_FONTSIZE,  # Title
    }
)


def annotate_subplots_abc(
    axes: list[Axes],
    format_str: str = "{})",
    background: str = "white",
    **annotate_kwargs,
):
    """
    Generates a), b), etc. in the corner of a subplot.

    Args:
        axes: list of matplotlib Axes that should be annotated
        format_str: string builder that is used to annotate. Every {} will be
            replaced by a, b, etc
        background: either 'white' or 'dark'.
        **annotate_kwargs: keyword args that are passed to plt.annotate.
    """

    default_annotate_args = dict(
        xy=(0, 1),
        xycoords="axes fraction",
        textcoords="offset fontsize",
        xytext=(+0.5, -0.5),
        verticalalignment="top",
        fontsize=SUBPLOT_ANNOTATION_FONTSIZE,
        bbox=dict(pad=3.0, edgecolor="none"),
    )

    if background == "white":
        default_annotate_args["bbox"].update(dict(facecolor="w"))
    elif background == "dark":
        default_annotate_args["bbox"].update(dict(facecolor="#00000080"))
        default_annotate_args.update(c="w")

    default_annotate_args.update(annotate_kwargs)

    for i in range(len(axes)):
        label = string.ascii_lowercase[i]
        axes[i].annotate(format_str.format(label), **default_annotate_args)


# %% Incident plane wave

planewave = np.zeros_like(sim.E_kx_array)
k_x_array = sim.k_x_array
planewave[np.argmin(np.abs(k_x_array))] = 1


sim.set_input_field_amplitudes(planewave)


fig, ax = plt.subplots(dpi=DPI, figsize=(sim.Lz, sim.Lx))
sim.visualize_intensity(title="", ax=ax, use_colorbar=False)
plt.axis("off")
# fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
# plt.autoscale(tight=True)
plt.tight_layout(pad=0)
if save_plots:
    plt.savefig(
        save_folder / Path(f"{project_name}_planewave.png"), pad_inches=0
    )
    print(f"Saved plot to {save_folder}")
plt.show()

# %%
from matplotlib import cm

# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

X = sim.Z_grid
Y = sim.X_grid
Z = np.abs(sim.E_total) ** 2
norm = plt.Normalize(Z.min(), Z.max())
colors = cm.BuPu(norm(Z))
rcount, ccount, _ = colors.shape

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
surf = ax.plot_surface(
    X, Y, Z, rcount=150, ccount=150, facecolors=colors, shade=False
)
# surf.set_facecolor((0,0,0,0))
plt.axis("off")


# ax.plot_wireframe(sim.Z_grid, sim.X_grid, , color=None, linewidth=1, cmap='viridis')
ax.set_zlim(0, 20)

plt.show()
# %% -

sim = CoupledDipoleSimulator.load(folder=data_folder)
lf_names = ["x", "z", "phi", "x_inv", "z_inv", "phi_inv", "det"]
# lf_names = ["det", "tr_inv", "spec_rad"]
# lf_names = [
#     "x",
#     "z",
#     "phi",
#     "x_inv",
#     "z_inv",
#     "phi_inv",
#     "tr_norm",
#     "det",
#     "tr_inv",
#     "spec_rad",
# ]
# lf_names = ["x", "det"]
for lf in loss_functions_dict.select_only(lf_names).values():

    sim.set_input_field_amplitudes(lf.wavefront)

    fig, ax = plt.subplots(figsize=(4.45, 4.0), dpi=DPI)
    # fig, ax = plt.subplots(
    #     figsize=(0.495 * TEXTWIDTH, 0.45 * TEXTWIDTH), dpi=DPI
    # )

    # name = lf.shortname.replace(" (scalar)", "")
    name = lf.displayname.replace("Singe parameter ", "")
    name = name.replace("Single parameter ", "")

    # add colorbar label to last plot
    if lf.name == lf_names[-1]:
        use_colorbar = True
        colorbar_label = {}
    else:
        use_colorbar = False
        colorbar_label = dict(colorbar_label=None)

    use_colorbar = True

    # print(lf.name, lf_names[-1])
    sim.visualize_intensity(
        title=f"Maximum Information State for {name}",
        # title=f"",
        ax=ax,
        imshow_kwargs=dict(cmap="turbo"),
        movable_kwargs=dict(
            marker="o", facecolors="none", edgecolors="r", s=80, c=None
        ),
        frozen_kwargs=dict(
            marker="s", facecolors="none", edgecolors="r", s=80, c=None
        ),
        # legend_kwargs=dict(fontsize=9),
        use_colorbar=use_colorbar,
        **colorbar_label,
    )
    # plt.clim(0, 10)
    plt.tight_layout(h_pad=0.0, w_pad=0.0)
    # plt.figure(fig)
    if save_plots:
        optimized_path = (
            save_folder
            / Path(f"optimized_wavefronts_{project_name}")
            / Path(f"optimized_{lf.name}.png")
        )
        optimized_path.parents[0].mkdir(exist_ok=True)
        plt.savefig(optimized_path, pad_inches=0.0)
        print(f"Saved plot to {optimized_path}")
    plt.show()


# %%

sim = CoupledDipoleSimulator.load(folder=data_folder)
# lf_names = ["x", "phi", "z"]
lf_names = ["det", "tr_inv", "spec_rad"]
lf_names = [
    "x",
    "z",
    "phi",
    "x_inv",
    "z_inv",
    "phi_inv",
    "tr_norm",
    "det",
    "tr_inv",
    "spec_rad",
]

fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(0.95 * TEXTWIDTH,))

for lf in loss_functions_dict.select_only(lf_names).values():

    sim.set_input_field_amplitudes(lf.wavefront)

    # fig, ax = plt.subplots(figsize=(5.0, 4.5), dpi=DPI)
    fig, ax = plt.subplots(
        figsize=(0.495 * TEXTWIDTH, 0.45 * TEXTWIDTH), dpi=DPI
    )

    # name = lf.shortname.replace(" (scalar)", "")
    name = lf.displayname.replace("Singe parameter ", "")
    name = name.replace("Single parameter ", "")

    # add colorbar label to last plot
    if lf.name == lf_names[-1]:
        colorbar = True
        colorbar_label = {}
    else:
        colorbar = False
        colorbar_label = dict(colorbar_label=None)

    # print(lf.name, lf_names[-1])
    sim.visualize_intensity(
        title=f"Maximum Information State\nfor {name}",
        ax=ax,
        # imshow_kwargs=dict(vmax=91),
        movable_kwargs=dict(
            marker="o", facecolors="none", edgecolors="r", s=80, c=None
        ),
        frozen_kwargs=dict(
            marker="s", facecolors="none", edgecolors="r", s=80, c=None
        ),
        legend_kwargs=dict(fontsize=9),
        colorbar=colorbar,
        **colorbar_label,
    )
    # plt.clim(0, 10)
    # plt.tight_layout(pad=0.0)
    # plt.figure(fig)
    if save_plots:
        optimized_path = (
            save_folder
            / Path(f"optimized_wavefronts_{project_name}")
            / Path(f"optimized_{lf.name}.png")
        )
        optimized_path.parents[0].mkdir(exist_ok=True)
        plt.savefig(optimized_path)
        print(f"Saved plot to {optimized_path}")
    plt.show()

# %% Double optimized plot

from mpl_toolkits.axes_grid1 import ImageGrid

sim = CoupledDipoleSimulator.load(folder=data_folder)

lf1 = loss_functions_dict["z"]
lf2 = loss_functions_dict["det"]

sim2 = sim.copy()

fig = plt.figure(figsize=(0.985 * TEXTWIDTH, 0.5 * TEXTWIDTH), dpi=DPI)
# fig = plt.figure(figsize=(6, 4), dpi=DPI)
axes = ImageGrid(
    fig,
    111,
    nrows_ncols=(1, 2),
    axes_pad=0.15,
    share_all=True,  # Shares both X and Y axes
    cbar_location="right",
    cbar_mode="edge",
    cbar_size="5%",
    cbar_pad=0.1,
)


sim.set_input_field_amplitudes(lf1.wavefront)
sim2.set_input_field_amplitudes(lf2.wavefront)

sim.compute_total_field()
sim2.compute_total_field()

max_intensity = np.max([np.abs(sim.E_total) ** 2, np.abs(sim2.E_total) ** 2])

plots = sim.visualize_intensity(
    ax=axes[0],
    colorbar_label=None,
    title="",
    imshow_kwargs=dict(cmap="turbo", clim=(0, max_intensity)),
    use_colorbar=False,
    use_legend=False,
)
plots2 = sim2.visualize_intensity(
    ax=axes[1],
    colorbar_label=None,
    title="",
    imshow_kwargs=dict(cmap="turbo", clim=(0, max_intensity)),
    use_colorbar=False,
    use_legend=False,
)
# plt.colorbar()

axes.cbar_axes[0].colorbar(plots2["field_img"])

annotate_subplots_abc(axes, format_str="({})", background="dark")

# axes[1].set_ylabel("")
# axes.cbar_axes[0].legend([plots2['frozen_plot']], ['lol'], loc='best')
# axes[0].legend([plots2['frozen_plot']], ['lol'])
fig.legend(
    *axes[1].get_legend_handles_labels(),
    ncols=2,
    bbox_to_anchor=(0.49, -0.01),
    loc="upper center",
)


plt.tight_layout(pad=0)
if save_plots:
    optimized_path = (
        save_folder
        / f"{project_name}_optimized_double_{lf1.name}_{lf2.name}.png"
    )
    fig.savefig(optimized_path, dpi=DPI, bbox_inches=0)
    print(f"Saved plot to {optimized_path}")
plt.show()
# %% 1) Bar plot of CRLBs


barplot_lossfunctions = [
    "x",
    "z",
    "phi",
    # "tr",
    "tr_norm",
    "det",
    "tr_inv",
    "spec_rad",
    # "dropped_spec_rad",
    # "partial_spec_rad",
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

loss_function_names = loss_functions_dict_bar.shortnames()

plt.rcdefaults()
# plt.rcParams['axes.labelsize'] = 110
# plt.rcParams['xtick.minor.size'] = 10
# plt.rcParams['xtick.major.size'] = 10
# # plt.rcParams['xtick.labelsize'] = 100
# plt.rcParams['xtick.color'] = 'g'
# plt.rcParams['font.size'] = 20

fig, axes = plt.subplots(
    ncols=2,
    # figsize=(14, 6),
    figsize=(1.0 * TEXTWIDTH, 0.4 * TEXTWIDTH),
    width_ratios=[len(loss_function_names) + 1, 2 + 1],
    dpi=DPI,
    layout="constrained",
)


crlbanalysis.visualize_crlb_barplot(
    loss_function_names=loss_function_names,
    param_legends=[
        "CRLB of x",
        "CRLB of z",
        r"CRLB of $\phi$",
    ],
    optimized_crlbs_joint=optimized_crlbs_joint,
    optimized_crlbs_scalar=optimized_crlbs_scalar,
    ax=axes[0],
    # title="Optimized Wavefronts",
    title="",
    tick_fontsize=6,
    ylim=(-0.000, None),
    hatch_color=(0.8,) * 3,
    joint_opacity=0.65,
    y_label=r"CRLB (wavelengths)",
    rotate_labels=False,
)

crlbanalysis.visualize_crlb_boxplot(
    loss_function_names=loss_function_names,
    param_legends=[
        "CRLB of x",
        "CRLB of z",
        r"CRLB of $\phi$",
    ],
    plane_wave_crlbs_joint=plane_wave_crlbs_joint,
    random_wavefronts_crlbs_joint=random_wavefronts_crlbs_joint,
    ax=axes[1],
    y_label=None,
    ylim=(-0.002, None),
    # title="Plane Waves and Random Wavefronts",
    title="",
    fontsize=6,
    ref_lines=[0, 0.0185],
)
annotate_subplots_abc(axes, background="white")

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
lf_dict_single_param_inv = loss_functions_dict.select_only(["x_inv", "z_inv"])
lf_dict_dropped = loss_functions_dict.select_only(
    ["dropped_det", "dropped_spec_rad", "dropped_tr_inv"]
)
lf_dict_full = loss_functions_dict.select_only(["det", "spec_rad", "tr_inv"])
lf_dict_partial = loss_functions_dict.select_only(
    [
        "partial_det",
        "partial_spec_rad",
        "partial_tr_inv",
    ]
)


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
        lf_dict_single_param_inv,
        legend_label="Three-param. optimality",
        color="green",
        marker="^",
    ),
    PlotGroup(
        lf_dict_full,
        legend_label="Three-param. optimality",
        color="green",
        marker="^",
    ),
    PlotGroup(
        lf_dict_dropped,
        legend_label="Two-param. optimality",
        color="red",
        marker="x",
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
        legend_label="Interpolation",
        color="green",
        linestyle="-",
        marker="",
        # marker="x",
    ),
]
# pareto_groups_sweep = [
#     PlotGroup(
#         loss_functions_sweep_dict.select_only("xz_joint_eps1"),
#         legend_label="$\epsilon=1.0$",
#         color="purple",
#         linestyle="-",
#         marker="",
#         # marker="x",
#     ),
#     PlotGroup(
#         loss_functions_sweep_dict.select_only("xz_joint_eps0.1"),
#         legend_label="$\epsilon=0.1$",
#         color="green",
#         linestyle="-",
#         marker="",
#     ),
#     PlotGroup(
#         loss_functions_sweep_dict.select_only("xz_joint_eps0"),
#         legend_label="$\epsilon=0.0$",
#         color="red",
#         linestyle="-",
#         marker="",
#     ),
#     PlotGroup(
#         loss_functions_sweep_dict.select_only("xz_joint_eps-0.1"),
#         legend_label="$\epsilon=-0.1$",
#         color="blue",
#         linestyle="-",
#         marker="",
#     ),
#     PlotGroup(
#         loss_functions_sweep_dict.select_only("xz_joint_eps-0.5"),
#         legend_label="$\epsilon=-1.0$",
#         color="orange",
#         linestyle="-",
#         marker="",
#     ),
# ]

x_idx = 1  # Display Z on x-axis
y_idx = 0  # Display X on y-axis

include_plot_indices = [(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (1, 5)]
# include_plot_indices = [(5, 2)]
include_plot_indices = [(1, 5)]
for i, (sweep_index, scatter_index) in enumerate(include_plot_indices):

    fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)

    wavelength_unit_order = -3
    wavelength_unit = 10 ** (-wavelength_unit_order)

    x_avoid, y_avoid = pareto_plot_sweep(
        pareto_groups_sweep[:sweep_index],
        x_idx,
        y_idx,
        ax=ax,
        wavelength_unit=wavelength_unit,
    )
    # line = np.linspace(0, 10)
    # plt.plot(line, line, ls='-', c='#AAA', lw=1)
    pareto_plot(
        pareto_groups[:scatter_index],
        x_idx,
        y_idx,
        xlabel=rf"CRLB for $z$ ($10^{{{wavelength_unit_order}}}$ wavelenghts)",
        ylabel=rf"CRLB for $x$ ($10^{{{wavelength_unit_order}}}$ wavelenghts)",
        title=r"Nuisance Aware Optimality Criteria",
        ax=ax,
        grid_kwargs=GRID_KWARGS,
        use_logscale=False,
        adjust_text_kwargs=dict(
            force_static=0.5,
            force_text=0.5,
            force_explode=(0.5, 0.5),
            force_pull=0.1,
            expand=(2.0, 2.5),
            min_arrow_len=40,
            iter_lim=1000,
            avoid_self=True,
            max_move=(100, 100),
            ensure_inside_axes=True,
        ),
        x_avoid=x_avoid,
        y_avoid=y_avoid,
        # xlim=(0.0017 * wavelength_unit, 0.009 * wavelength_unit),
        # ylim=(0.00340 * wavelength_unit, 0.0056 * wavelength_unit),
        xlim=None,
        ylim=None,
        wavelength_unit=wavelength_unit,
    )

    if save_plots:
        fig_name = f"{project_name}_pareto_plot_nuisance_fig{i}"
        plt.savefig(save_folder / f"{fig_name}.png")
        plt.savefig(save_folder / f"{fig_name}.pdf")
        print(f"Saved pareto plot to '{fig_name}'")

plt.show()


# %%
"""
Animation showing all the incident scattering angles
"""

DPI = 300
sim = CoupledDipoleSimulator.load(folder=data_folder)
sim_kwargs = sim.get_kwargs()
# sim_kwargs["grid_points_x"] = 40
# sim_kwargs["k_x_array"] = None
# sim_kwargs["theta_inc_deg"] = np.linspace(-70, 70, num=20)

sim = CoupledDipoleSimulator(**sim_kwargs)
E_kx_array = np.zeros_like(sim.theta_inc)
zero_index = np.argmin(np.abs(sim.k_x_array))
E_kx_array[zero_index] = 1
sim.set_input_field_amplitudes(E_kx_array / 2.17, flux_normalize=True)

fig, ax = sim.visualize_intensity(title="Field Intensity Over Slab", dpi=DPI)

if save_plots:
    fig.savefig(save_folder / f"{project_name}_zero_angle.png")

    sim.animate_time_evolution(
        frames=3,
        fps=30,
        save_gif=True,
        save_mp4=False,
        gif_filename=f"../figures/{project_name}_time_evolution.gif",
        mp4_filename=f"../figures/{project_name}_time_evolution.mp4",
    )
# fig.save()

n_frames = 2 * len(E_kx_array) - 2
# %%


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
