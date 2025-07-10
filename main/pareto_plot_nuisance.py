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
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

sys.exit()

# %% Loading data from disk
data_folder = "../data/quasirandom"

sim = CoupledDipoleSimulator.load(folder=data_folder)
fisher_calc = FisherMatrix.load(folder=data_folder)
loss_functions_dict = LossFunctionsDict.load(folder=data_folder)


# %%


def pareto_plot(
    loss_function_names,
    optimized_crlbs_joint,
    param_indices=(0, 1),
    xmax=None,
    ymax=None,
    lowerisbetter=True,
    dpi=100,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    points = []
    texts = []
    # p_front = calc_pareto_front(results, lowerisbetter=lowerisbetter)
    # p_front_names = [result[0].name for result in p_front]
    paramx, paramy = param_indices
    for name, crlb in zip(loss_function_names, optimized_crlbs_joint):
        # loss_function, data = result
        x = crlb[paramx]  # x
        y = crlb[paramy]  # z

        marker = "x"
        color = "blue"
        label = "Non-efficient"

        # if loss_function.name in p_front_names:
        #     color = "r"
        #     label = "Efficient"

        x_out_bounds = not xmax is None and x > xmax
        y_out_bounds = not ymax is None and y > ymax

        if x_out_bounds:
            x = xmax
            marker = matplotlib.markers.CARETRIGHT
            color = "black"
            label = None
        if y_out_bounds:
            y = ymax
            marker = matplotlib.markers.CARETUP
            color = "black"
            label = None
        if x_out_bounds and y_out_bounds:
            marker = "D"

        point = ax.scatter(x, y, s=80, marker=marker, color=color, label=label)
        text = ax.annotate(
            name,
            (x, y),
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=1.0
            ),
            zorder=10,
        )
        points.append(point)
        texts.append(text)
    # plt.scatter(4e-5, 6e-5, color='red', marker='v')
    ax.set_xlabel("joint CRLB of x")
    ax.set_ylabel("joint CRLB of z")
    plt.xlim(None, xmax)
    plt.ylim(None, ymax)
    ax.set_title("Pareto Front of Cramér-Rao Lower bounds")

    ax.grid(True)

    # delete duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    adjust_text(
        texts,
        ax=ax,
        # only_move={'text':'xy', 'explode':'xy'}, # allow text to move in both directions, points only in y
        arrowprops=dict(arrowstyle="-", lw=0.5, alpha=1.0, zorder=0),
        expand=(2.5, 2.5),  # how much to "inflate" the text boxes,
        avoid_self=True,
        force_text=0.5,  # strength of repulsion
        force_static=0.1,
        lim=10,  # max iterations
    )

    plt.show()


# %%
include_lossfunctions = [
    "x",
    "z",
    "phi",
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

optimized_crlbs_joint, optimized_crlbs_scalar = optimized_crlbs
loss_function_names = [
    lf.shortname for lf in loss_functions_dict_plotting.values()
]
pareto_plot(loss_function_names, optimized_crlbs_joint)

# %% 10) Assessment of CRLB


wavefront_labels_latex = {
    # Single-parameter MIS
    "MIS_shiftX": r"MIS$_{x}$",
    "MIS_shiftZ": r"MIS$_{z}$",
    "MIS_rotation": r"MIS$_{\theta}$",
    # Full FIM Optimality Criteria
    "TraceJ_norm_eig": r"T-Opt. ($\mathcal{J}$)",
    "DetJ_torch": r"D-Opt. ($\mathcal{J}$)",
    "SpecJ_torch": r"E-Opt. ($\mathcal{J}$)",
    "TraceJinv_torch": r"A-Opt. ($\mathcal{J}$)",
    # Heuristic Nuisance Strategies
    "LogLin_torch": "LogLin",
    "OffDiag_torch": "OffDiag",
    "SubDet_torch": "D-Opt. ($\mathcal{J}_{\psi|\psi}$)",
    # Marginal FIM Optimality Criteria (Principled Nuisance-Aware)
    "DetJ_marginal_n2_torch": r"D-Opt. ($\mathcal{J}_{\psi|\phi}$)",
    "TraceJinv_marginal_n2_torch": r"A-Opt. ($\mathcal{J}_{\psi|\phi}$)",
    "SpecJ_marginal_n2_torch": r"E-Opt. ($\mathcal{J}_{\psi|\phi}$)",
}

param_map = {
    "MIS_shiftX": [0],
    "MIS_shiftZ": [1],
    "MIS_rotation": [2],
    "TraceJ_norm_eig": "all",  # Special keyword for all parameters
    "DetJ_torch": "all",
    "SpecJ_torch": "all",
    "TraceJinv_torch": "all",
    "SubDet_torch": [0, 1],  # Optimized for x and z
    "DetJ_marginal_n2_torch": [0, 1],
    "TraceJinv_marginal_n2_torch": [0, 1],
    "SpecJ_marginal_n2_torch": [0, 1],
}

legend_params = {"title": "Parameters", "loc": "upper left"}
legend_status = {
    "title": "Optimization Status",
    "loc": "upper left",
    "bbox_to_anchor": (0.08, 1),  # Adjust this x-value to move it left/right
}

# Selecting wavefronts via removal
wavefronts_for_plotting = optimized_wavefronts.copy()

# wavefronts_for_plotting.pop("TraceJ_norm_eig", None)
# wavefronts_for_plotting.pop("DetJ_marginal_n2_torch", None)
# wavefronts_for_plotting.pop("TraceJinv_marginal_n2_torch", None)
# wavefronts_for_plotting.pop("SpecJ_marginal_n2_torch", None)
# wavefronts_for_plotting.pop("SubDet_torch", None)

crlbanalysis.visualize_crlb_comparison(
    optimized_wavefronts=wavefronts_for_plotting,
    F_operator_tensor=optimizer.F_op,
    photon_budget=photon_budget,
    param_names=param_names,
    sim_kwargs=sim_kwargs,
    kx_array=kx_array,
    figure_title="",
    show_plane_waves=True,
    show_random_wavefronts=True,
    num_random_samples=1000,
    optimized_param_map=param_map,
    ylim=(0.9e-3, 1.25e-1),
    force_linear_scale=False,  # Force the y-axis to be linear
    legend_params_kwargs=legend_params,
    legend_status_kwargs=legend_status,
    xtick_labels_map=wavefront_labels_latex,
    xtick_label_rotation=0,
    custom_wavefronts_label=r"Shaped Wavefront $E^\mathrm{in}$",
    broad_xlabel_yoffset=-0.07,
    # filepath="figures/final_crlb_comparison.png",
    # dpi=400,
)


# %% 11) Finally, generate Pareto frontier plot

wavefront_labels_latex = {
    # Single-parameter MIS
    "MIS_shiftX": r"MIS$_{x}$",
    "MIS_shiftZ": r"MIS$_{z}$",
    "MIS_rotation": r"MIS$_{\theta}$",
    # Full FIM Optimality Criteria
    "TraceJ_norm_eig": r"T-Opt. ($\mathcal{J}$)",
    "DetJ_torch": r"D-Opt. ($\mathcal{J}$)",
    "SpecJ_torch": r"E-Opt. ($\mathcal{J}$)",
    "TraceJinv_torch": r"A-Opt. ($\mathcal{J}$)",
    # Heuristic Nuisance Strategies
    "LogLin_torch": "LogLin",
    "OffDiag_torch": "OffDiag",
    "SubDet_torch": "D-Opt. ($\mathcal{J}_{\psi|\psi}$)",
    # Marginal FIM Optimality Criteria (Principled Nuisance-Aware)
    "DetJ_marginal_n2_torch": r"D-Opt. ($\mathcal{J}_{\psi|\phi}$)",
    "TraceJinv_marginal_n2_torch": r"A-Opt. ($\mathcal{J}_{\psi|\phi}$)",
    "SpecJ_marginal_n2_torch": r"E-Opt. ($\mathcal{J}_{\psi|\phi}$)",
}

# 1. Create wavefront_categories dictionary
wavefront_categories_map = {}
for key in optimized_wavefronts.keys():
    if key.startswith("MIS_") or "TraceJ_norm_eig" in key:
        wavefront_categories_map[key] = "MIS"
    elif key in ["DetJ_torch", "SpecJ_torch", "TraceJinv_torch"]:
        wavefront_categories_map[key] = "TorchFull"
    elif key in ["LogLin_torch", "OffDiag_torch", "SubDet_torch"]:
        wavefront_categories_map[key] = "NuisanceHeuristic"
    elif (
        key.startswith("DetJ_marginal_")
        or key.startswith("TraceJinv_marginal_")
        or key.startswith("SpecJ_marginal_")
    ):
        wavefront_categories_map[key] = "TorchMarginal"
    else:
        wavefront_categories_map[key] = "Other"

for key in optimized_wavefronts.keys():
    if key not in wavefront_labels_latex:
        wavefront_labels_latex[key] = key.replace("_", " ").replace(
            "torch", ""
        )

# 3. Define category_styles with new, clearer labels
category_styles_def = {
    "MIS": {
        "marker": "x",
        "color": "purple",
        "label": "Single-Param. MIS",
    },
    "n2_torch": {
        "marker": "o",
        "color": "blue",
        "label": "Two-Param. MIS, nuisance-aware",
    },
    "_torch": {
        "marker": "s",
        "color": "green",
        "label": "Multi-Param. MIS",
    },
    "SubDet_torch": {
        "marker": "P",
        "color": "red",
        "label": "Two-Param. MIS",
    },
    "Other": {"marker": "d", "color": "grey", "label": "Other"},
    "default": {"marker": "*", "color": "black", "label": "Default"},
}

# 4. Create a copy of the wavefronts dictionary for plotting
wavefronts_for_plotting = optimized_wavefronts.copy()

# Remove the 'TraceJ_norm_eig' wavefront as requested
wavefronts_for_plotting.pop("TraceJ_norm_eig", None)

# wavefronts_for_plotting.pop("DetJ_marginal_n2_torch", None)
# wavefronts_for_plotting.pop("TraceJinv_marginal_n2_torch", None)
# wavefronts_for_plotting.pop("SpecJ_marginal_n2_torch", None)
# wavefronts_for_plotting.pop("SubDet_torch", None)

print("\n'TraceJ_norm_eig' wavefront removed from the following plots.")


# 5. Call the plotting function with the modified dictionary
print("\nPlotting CRLB_x vs CRLB_z...")
crlbanalysis.plot_pareto_frontiers(
    optimized_wavefronts_dict=wavefronts_for_plotting,
    F_operator_tensor_np=optimizer.F_op,
    photon_budget_val=photon_budget,
    param_names_list=param_names,
    params_to_plot_indices=[0, 1],
    wavefront_categories_style=category_styles_def,
    wavefront_labels_latex_map=wavefront_labels_latex,
    filepath="figures/pareto_frontier_x_vs_z.png",
    dpi=300,
)
