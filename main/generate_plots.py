# %% Import Libraries
import string
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes._axes import Axes
from pareto_plot_nuisance import (
    ParetoGroup,
    interpolate_J,
    pareto_plot,
    pareto_plot_sweep,
)

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction, LossFunctionsDict
from scatteringlibs.matplotlibmagic import (
    PlotGroup,
    add_dots_to_legend,
    make_leftax_look_like_inset,
    register_turbo,
)
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

sys.exit()

# %% Loading in data
project_name = "quasirandom_longrun - Copy"
project_name = "quasirandom_shortrun_dist2-8"
# project_name = "quasirandom_shortrun_dist2-6"
# project_name = "quasirandom_shortrun_dist2-4"
# project_name = "quasirandom_shortrun_dist3-0"

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
PHOTON_BUDGET = 1e3

GRID_KWARGS = dict(which="both", linestyle=":")

save_plots = False
save_folder = Path(f"../figures/{project_name}/")
save_folder.mkdir(parents=True, exist_ok=True)

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

planewave = np.zeros_like(sim.k_x_array)
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

# %% 3D test
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
# %% Plot optimized states

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

fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(0.95 * TEXTWIDTH, None))

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

lf1 = loss_functions_dict["x"]
lf2 = loss_functions_dict["det"]
# lf1 = loss_functions_dict["tr_norm"]

sim2 = sim.copy()


sim.set_input_field_amplitudes(lf1.wavefront)
sim2.set_input_field_amplitudes(lf2.wavefront)

sim.compute_total_field()
sim2.compute_total_field()

intensity = np.abs([sim.E_total, sim2.E_total]) ** 2
# intensity2 = np.abs(sim2.E_total) ** 2
plt.hist(intensity.flatten(), bins=100, range=(10, 100))
# plt.hist(intensity2.flatten(), bins=100, range=(10, 100))
plt.xticks(range(10, 100, 5))
plt.yscale("log")
plt.show()


# max_intensity = np.max([np.abs(sim.E_total) ** 2, np.abs(sim2.E_total) ** 2])
max_intensity = 65


fig = plt.figure(figsize=(0.495 * TEXTWIDTH, 0.27 * TEXTWIDTH), dpi=DPI)
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

# Set color of intensities over max_intensity
plots["field_img"].cmap.set_over("w")
plots2["field_img"].cmap.set_over("w")
axes.cbar_axes[0].colorbar(plots2["field_img"], extend="max")

annotate_subplots_abc(axes, format_str="({})", background="dark")

# axes[1].set_ylabel("")
# axes.cbar_axes[0].legend([plots2['frozen_plot']], ['lol'], loc='best')
# axes[0].legend([plots2['frozen_plot']], ['lol'])

handles, labels, handler_map = add_dots_to_legend(
    axes[1].get_legend_handles_labels()
)
fig.legend(
    handles,
    labels,
    handler_map=handler_map,
    ncols=2,
    bbox_to_anchor=(0.5, 0.95),
    loc="lower center",
)


plt.tight_layout(pad=0)
if save_plots:
    optimized_path = (
        save_folder
        / f"{project_name}_optimized_double_{lf1.name}_{lf2.name}.png"
    )
    fig.savefig(optimized_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved plot to {optimized_path}")
plt.show()
# %% Bar plot of CRLBs


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
PHOTON_BUDGET = 1e3

optimized_crlbs = crlbanalysis.compute_crlb_for_optimized(
    loss_functions_dict_bar,
    photon_budget=PHOTON_BUDGET,
)
plane_wave_crlbs = crlbanalysis.compute_crlb_for_planewaves(
    sim_kwargs=sim.get_kwargs(),
    F_operator_tensor=fisher_calc.F,
    photon_budget=PHOTON_BUDGET,
    kx_array=sim.k_x_array,
)
random_wavefronts_crlbs = crlbanalysis.compute_crlb_for_random_wavefronts(
    F_operator_tensor=fisher_calc.F,
    photon_budget=PHOTON_BUDGET,
    num_samples=100,
)


optimized_crlbs_joint, optimized_crlbs_scalar = optimized_crlbs
plane_wave_crlbs_joint, plane_wave_crlbs_scalar = plane_wave_crlbs
random_wavefronts_crlbs_joint, random_wavefronts_crlbs_scalar = (
    random_wavefronts_crlbs
)

loss_function_names = loss_functions_dict_bar.shortnames()


fig, axes = plt.subplots(
    ncols=2,
    # figsize=(14, 6),
    figsize=(1.4 * TEXTWIDTH, 0.7 * TEXTWIDTH),
    width_ratios=[len(loss_function_names) + 1, 2 + 1],
    dpi=DPI,
    layout="constrained",
)


wavelength_unit_order = -3
wavelength_unit = 10 ** (-wavelength_unit_order)
if wavelength_unit_order == 0:
    wavelength_unit_str = ""
else:
    wavelength_unit_str = rf"$10^{{{wavelength_unit_order}}}$ "

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
    xtick_fontsize=None,
    ytick_fontsize=None,
    axes_label_fontsize=None,
    ylim=(-0.000, None),
    hatch_color=(0.8,) * 3,
    joint_opacity=0.65,
    # y_label=rf"CRLB ({wavelength_unit_str}wavelenghts)",
    y_label=rf"Cramér-Rao Lower Bound",
    rotate_labels=False,
    wavelength_unit=wavelength_unit,
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
    ylim=(0, None),
    # title="Plane Waves and Random Wavefronts",
    title="",
    fontsize=None,
    # ref_lines=[0, 0.0185*PHOTON_BUDGET],
    # ref_lines=[0, 500],
    wavelength_unit=wavelength_unit,
)
annotate_subplots_abc(axes, background="white")

make_leftax_look_like_inset(
    fig, leftax=axes[0], rightax=axes[1], moveticksright=True
)
fig.set_constrained_layout_pads(wspace=0.1)

if save_plots:

    fig_name = f"{project_name}_crlb_barplot_PB-{PHOTON_BUDGET:.2e}"
    plt.savefig(save_folder / (fig_name + ".png"))
    plt.savefig(save_folder / (fig_name + ".pdf"))
    print(f"Saved CRLB bar plot to '{save_folder}\{fig_name}'")
plt.show()


# %% Pareto plot

"""
Select the different loss functions (optimality criteria) from the 
loss_functions_dict
"""

lf_dict_single_param = loss_functions_dict.select_only(["x", "z"])
lf_dict_single_param_inv = loss_functions_dict.select_only(["x_inv", "z_inv"])
lf_dict_dropped = loss_functions_dict.select_only(
    ["dropped_det", "dropped_spec_rad", "dropped_tr_inv", "dropped_tr_norm"]
)
lf_dict_full = loss_functions_dict.select_only(
    ["det", "spec_rad", "tr_inv", "tr_norm"]
)
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
        # legend_label="Single-param. optimality",
        legend_label="Scalar",
        color="purple",
        marker=".",
    ),
    PlotGroup(
        lf_dict_single_param_inv,
        legend_label="Full",
        color="green",
        marker="^",
    ),
    PlotGroup(
        lf_dict_full,
        legend_label="Full",
        color="green",
        marker="^",
    ),
    PlotGroup(
        lf_dict_dropped,
        legend_label="Subblock",
        color="red",
        marker="x",
    ),
    PlotGroup(
        lf_dict_partial,
        legend_label="Partial",
        color="blue",
        marker="s",
    ),
]
pareto_groups_sweep = [
    PlotGroup(
        loss_functions_sweep_dict.select_only(["xz_scalar"]),
        legend_label="Linear\nCombination",
        color="purple",
        linestyle="-",
        marker="",
        # marker="x",
    ),
    PlotGroup(
        loss_functions_sweep_dict.select_only(["xz_joint"]),
        legend_label="Linear\nCombination",
        color="green",
        linestyle="-",
        marker="",
        # marker="x",
    ),
]


x_idx = 1  # Display Z on x-axis
y_idx = 0  # Display X on y-axis

"""
Use this line to plot all the components incrementally. (1, 4) means plot 1 
pareto_groups_sweep plot and 4 pareto_groups plots. Nice for presentations to
introduce elements one at a time.
"""
include_plot_indices = [(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (1, 5)]
include_plot_indices = [(3, 5)]

PHOTON_BUDGET = 1e3

for i, (sweep_index, scatter_index) in enumerate(include_plot_indices):

    # fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
    fig, ax = plt.subplots(
        figsize=(0.95 * TEXTWIDTH, 0.7 * TEXTWIDTH), dpi=DPI
    )

    wavelength_unit_order = -3
    wavelength_unit = 10 ** (-wavelength_unit_order)
    if wavelength_unit_order == 0:
        wavelength_unit_str = ""
    else:
        wavelength_unit_str = rf"$10^{{{wavelength_unit_order}}}$ "

    x_avoid, y_avoid = pareto_plot_sweep(
        pareto_groups_sweep[:sweep_index],
        x_idx,
        y_idx,
        ax=ax,
        wavelength_unit=wavelength_unit,
        photon_budget=PHOTON_BUDGET,
    )

    # x_avoid += [0.228]
    # y_avoid += [0.252]
    pareto_plot(
        pareto_groups[:scatter_index],
        x_idx,
        y_idx,
        xlabel=rf"CRLB for $z$ ({wavelength_unit_str}wavelenghts)",
        ylabel=rf"CRLB for $x$ ({wavelength_unit_str}wavelenghts)",
        # title=r"Nuisance Aware Optimality Criteria",
        title="",
        ax=ax,
        grid_kwargs=GRID_KWARGS,
        use_logscale=False,
        adjust_text_kwargs=dict(
            force_static=0.24,
            force_text=0.2,
            force_explode=(0.0, 0.0),
            force_pull=0.15,
            expand=(1.8, 1.8),
            min_arrow_len=40,
            iter_lim=1000,
            avoid_self=True,
            max_move=(5000, 5000),
            ensure_inside_axes=True,
        ),
        x_avoid=x_avoid,
        y_avoid=y_avoid,
        # xlim=(0.2*PHOTON_BUDGET, None),
        # ylim=(0.233*PHOTON_BUDGET, 0.288*PHOTON_BUDGET),
        wavelength_unit=wavelength_unit,
        photon_budget=PHOTON_BUDGET,
    )

    # Plot  z = x
    x = np.array([0, 1000])
    plt.plot(x, x, ls="-", c="#AAA", lw=1, alpha=0.5)

    if save_plots:
        fig_name = f"{project_name}_pareto_plot_nuisance_PB-{PHOTON_BUDGET:.2e}_fig{i}"
        plt.savefig(save_folder / f"{fig_name}.png")
        plt.savefig(save_folder / f"{fig_name}.pdf")
        print(f"Saved pareto plot to '{fig_name}'")
# ax.set_aspect("equal")
plt.show()

# %% Measurement interpolation
"""
How much Fisher information do we get if we measure state 1 for fraction alpha
of the measurement time and state 2 for fraction (1-alpha) of the measurement 
time. Do we outperform using multiparameter maximum information states?
"""
multiparam_sweep = loss_functions_sweep_dict["xz_joint"]
J1 = multiparam_sweep.Js[0]
J2 = multiparam_sweep.Js[-1]
interp_sweep = interpolate_J(J1, J2, "xz_joint_J_interp")
interp_sweep_dict = LossFunctionsDict(xz_joint_J_interp=interp_sweep)

pareto_groups_sweep = [
    PlotGroup(
        loss_functions_sweep_dict.select_only(["xz_joint"]),
        legend_label="Interpolation",
        color="green",
        linestyle="-",
        marker=".",
    ),
    PlotGroup(
        interp_sweep_dict,
        legend_label="Time Sharing",
        color="k",
        linestyle="-",
        marker="^",
    ),
]


x_idx = 1  # Display Z on x-axis
y_idx = 0  # Display X on y-axis

wavelength_unit_order = 0
wavelength_unit = 10 ** (-wavelength_unit_order)
if wavelength_unit_order == 0:
    wavelength_unit_str = ""
else:
    wavelength_unit_str = rf"$10^{{{wavelength_unit_order}}}$ "


fig, ax = plt.subplots(figsize=(0.5 * TEXTWIDTH, 0.5 * TEXTWIDTH), dpi=DPI)
x_avoid, y_avoid = pareto_plot_sweep(
    pareto_groups_sweep,
    x_idx,
    y_idx,
    ax=ax,
    wavelength_unit=wavelength_unit,
    grid_kwargs=GRID_KWARGS,
    xlabel=rf"CRLB for $z$ ({wavelength_unit_str}wavelenghts)",
    ylabel=rf"CRLB for $x$ ({wavelength_unit_str}wavelenghts)",
)


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

sim.visualize_intensity(title="Field Intensity Over Slab", dpi=DPI)

if save_plots:
    plt.savefig(save_folder / f"{project_name}_zero_angle.png")

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

fig, ax = plt.subplots()


def init():
    pass


def update(i):
    global E_kx_array
    print(f"\rAnimating frame {i+1:>2} / {n_frames} ", end="")
    fig.clear()
    ax.clear()
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

# %% Sensitivity animation
"""
Animate shifting the dipoles
"""
# import placedipoles

lf = loss_functions_dict["x"]
wavefront = "max_info"
# wavefront = 'planewave'
# wavefront = "min_info"

optimizer = Optimizer(F_operator_tensor=fisher_calc.F)
lf.optimize(optimizer, num_eig=-1)
sim = CoupledDipoleSimulator.load(folder=data_folder)
k_x_array = sim.k_x_array

if wavefront == "max_info":
    sim.set_input_field_kx(k_x_array, lf.wavefronts[0])
elif wavefront == "planewave":
    sim.set_input_field_kx([0], [1])
elif wavefront == "min_info":
    sim.set_input_field_kx(k_x_array, lf.wavefronts[-1])


movable_dipoles_orig = sim.movable_scatterers.copy()
frozen_dipoles_orig = sim.frozen_scatterers.copy()

x = np.linspace(-1, 1, num=7)


# %%%

# --- 1. PRE-COMPUTATION ("PRELOADING") ---


x = np.linspace(-0.2, 0.2, num=30)
# x = np.fft.fftshift(x) # Start animation at zero devation

precomputed_frames = []
global_vmin = np.inf
global_vmax = -np.inf

# Loop through all states, run the simulation, and store the results
for dx in x:
    print(f"\rStarting pre-computation... {dx:.2f} / {x[-1]:.2f}\t\t", end="")
    # Run the simulation logic
    movable_dipoles = placedipoles.shift_and_rotate_dipoles(
        movable_dipoles_orig, dx, 0, 0
    )
    sim.set_scatterers(movable_scatterers=movable_dipoles)
    sim.reset()
    sim.compute_total_field()
    # Get the results
    intensity_map = np.abs(sim.E_total) ** 2
    E_total = sim.E_total.copy()
    movable_pos = sim.movable_scatterers.copy()

    # Store them
    precomputed_frames.append(
        {
            "intensity": intensity_map,
            "movable_pos": movable_pos,
            "E_total": sim.E_total.copy(),
        }
    )

    # Update global color limits
    if intensity_map.min() < global_vmin:
        global_vmin = intensity_map.min()
    if intensity_map.max() > global_vmax:
        global_vmax = intensity_map.max()

print("Global max:", global_vmax)
global_vmax = 26
global_vmin = 0
print(f"\nPre-computation complete for {len(precomputed_frames)} frames.")

# %%%
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- 2. ANIMATION ---

# We need these from the simulator object to set up the plot
X_grid_lambda = sim.X_grid / sim.lambda_0
Z_grid_lambda = sim.Z_grid / sim.lambda_0
frozen_pos_lambda = frozen_dipoles_orig / sim.lambda_0

# # These will hold the artist objects
# fig, axes = plt.subplots(
#     ncols=2, figsize=(16, 8), dpi=DPI, sharey=True, gridspec_kw={'wspace': 0, 'width_ratios':[1,.5]}
# )  # Use the same figsize as your original
# ax = axes[0]
# ax2 = axes[1]


fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)
divider = make_axes_locatable(ax)
ax2 = divider.append_axes("right", size="50%", pad=0.0)
ax2.set_yticks([])

field_img = None
movable_plot = None
abs_plot = None

plt.rcParams["axes.labelsize"] = 10


def init():
    global field_img, movable_plot, ax, ax2, fig, frozen_plot, legend, abs_plot, re_plot, im_plot, x_out

    # Clear figure and add axes (like in the previous solution)
    ax.cla()
    ax2.cla()

    ax2.set_yticks([])

    # Get data for the first frame
    first_frame_data = precomputed_frames[0]

    imshow_kwargs = {
        "vmin": global_vmin,
        "vmax": global_vmax,
        "cmap": "viridis",
    }

    plots = sim.visualize_intensity(
        title="",
        ax=ax,
        imshow_kwargs=imshow_kwargs,
        dpi=DPI,
        use_colorbar=False,
        legend_kwargs=dict(fontsize=12),
    )
    ax.axvline(x=4, c="r", ls="--")
    ax2.axvline(x=0, c="r", ls="--")
    field_img = plots["field_img"]
    movable_plot = plots["movable_plot"]
    frozen_plot = plots["frozen_plot"]
    legend = plots["legend"]

    E_out = first_frame_data["E_total"][:, -1]
    x_out = X_grid_lambda[:, -1]
    re_plot = ax2.plot(np.real(E_out), x_out, label="Real part of output")[0]
    im_plot = ax2.plot(np.imag(E_out), x_out, label="Imag. part of output")[0]
    # abs_plot = ax2.plot(np.abs(E_out), x_out, -np.abs(E_out), x_out)[0]
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(np.min(sim.x_vis), np.max(sim.x_vis))
    ax2.set_xticks([])
    # ax2.set_yticks([])
    ax2.legend(loc="upper right", fontsize=12, framealpha=1)
    # ax.set_aspect('equal')
    # fig.subplots_adjust(wspace=0)
    plt.tight_layout(pad=0.0)

    return (field_img, movable_plot, frozen_plot, legend, abs_plot)


def update(i):
    # Get the pre-computed data for this frame
    frame_data = precomputed_frames[i]

    field_img.set_array(frame_data["intensity"])

    # # Update the positions of the movable scatterers
    movable_pos_lambda = frame_data["movable_pos"].T / sim.lambda_0
    movable_plot.set_offsets(movable_pos_lambda)

    E_out = frame_data["E_total"][:, -1]
    re_plot.set_data(np.real(E_out), x_out)
    im_plot.set_data(np.imag(E_out), x_out)
    # abs_plot.set_data(np.abs(E_out), x_out, -np.abs(E_out), x_out)

    # fig.subplots_adjust(wspace=0)

    return (
        field_img,
        movable_plot,
        frozen_plot,
        legend,
        abs_plot,
        re_plot,
        im_plot,
    )


# Create the animation
# blit=True is the key to smoothness!

frames = np.arange(2 * len(precomputed_frames) - 2)
frames = np.min([frames, len(frames) - frames], axis=0)
# frames = np.fft.fftshift(frames) # start at zero deviation
ani = matplotlib.animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=frames,
    blit=False,
    interval=40,
    repeat=True,
    repeat_delay=0,
)

# To see it in a notebook, you might use:
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# Or to save it:
if save_plots:
    animation_time_s = 3
    fps = int(np.round(len(frames) / animation_time_s))

    fig_name = f"sensitivity_animation_{wavefront}"
    optimized_path = save_folder / Path(f"sensitivity_anim_{project_name}")
    optimized_path.parents[0].mkdir(exist_ok=True)
    optimized_path.mkdir(exist_ok=True)
    DPI = 150
    ani.save(optimized_path / f"{fig_name}.gif", fps=fps, dpi=DPI)
    ani.save(optimized_path / f"{fig_name}.mp4", fps=fps, dpi=DPI)
    print(f"Saved sensitivity animation to {optimized_path}/{fig_name}")
