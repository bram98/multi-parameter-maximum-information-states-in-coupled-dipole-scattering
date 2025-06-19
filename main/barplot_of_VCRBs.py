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


def visualize_crlb_comparison(
    loss_function_names: list,
    param_legends: list,
    optimized_crlbs_joint: np.ndarray = None,
    optimized_crlbs_scalar: np.ndarray = None,
    plane_wave_crlbs_joint: np.ndarray = None,
    plane_wave_crlbs_scalar: np.ndarray = None,
    random_wavefronts_crlbs_joint: np.ndarray = None,
    random_wavefronts_crlbs_scalar: np.ndarray = None,
    figure_title: str = "CRLB Comparison",
    show_plane_waves: bool = True,
    show_random_wavefronts: bool = True,
    num_random_samples: int = 100,
    ax: matplotlib.axes._axes.Axes = None,
    bar_width: float = 0.2,
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

    if ax is None:
        # Adjust figure width based on number of boxplot groups
        fig_width = (
            7 + 2.5 * num_boxplot_groups + 0.5 * num_optimized_wavefronts
        )
        fig, ax = plt.subplots(
            figsize=(max(10, fig_width), 7)
        )  # Ensure a minimum width

    cur_x_position = 0
    xtick_positions = np.array([], dtype=float)

    # Colors for parameters (e.g., CRLB_x, CRLB_z, CRLB_theta)
    # Ensure enough colors if num_params > 3
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    param_colors = [colors[i % len(colors)] for i in range(num_params)]

    if optimized_crlbs_joint is not None:

        x_positions_optimized = cur_x_position + np.arange(
            num_optimized_wavefronts
        )
        xtick_positions = np.append(xtick_positions, x_positions_optimized)
        cur_x_position += len(x_positions_optimized)

        joint_alpha = 0.7 if optimized_crlbs_scalar is not None else 1.0

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
            )

    if optimized_crlbs_scalar is not None:
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
            )

    # # Bar charts for custom wavefronts
    # if num_custom_wavefronts > 0:
    #     for i in range(num_params):  # For each parameter (x, z, theta)
    #         # Offset for each parameter's bar within a group
    #         offset = (i - (num_params - 1) / 2) * bar_width
    #         ax.bar(
    #             x_idx_custom + offset,
    #             data_custom[:, i],
    #             width=bar_width,
    #             label=crlb_plot_labels[i],
    #             color=param_colors[i],
    #             alpha=1.0,
    #         )

    # Determine positions for boxplots
    current_xtick_pos = (
        num_optimized_wavefronts - 1
    )  # Last position of custom bars
    # xtick_positions = list(x_positions_optimized)
    xtick_labels = loss_function_names

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
        for patch, color in zip(bp_plane["boxes"], param_colors):
            patch.set_facecolor(color)
            patch.set_alpha(1.0)

        # xtick_positions.append(
        #     np.mean(box_positions_plane)
        # )  # Tick in middle of group
        xtick_labels.append("Plane Waves")

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
        for patch, color in zip(bp_random["boxes"], param_colors):
            patch.set_facecolor(color)
            patch.set_alpha(1.0)

        # xtick_positions.append(
        #     np.mean(box_positions_random)
        # )  # Tick in middle of group
        xtick_labels.append(f"Random WFs\n({num_random_samples} samples)")

    # # Boxplots for random wavefronts
    # if show_random_wavefronts and random_wavefront_data_for_boxplot:
    #     # Adjust current_xtick_pos based on whether plane waves were plotted
    #     if (
    #         show_plane_waves and plane_wave_data_for_boxplot
    #     ):  # If plane waves were plotted
    #         current_xtick_pos = (
    #             np.mean(box_positions_plane)
    #             + (num_params / 2 * bar_width * 1.2)
    #             + 0.5
    #         )  # position after plane wave boxes
    #     else:  # If only custom bars were plotted or no plane waves
    #         current_xtick_pos = (
    #             num_custom_wavefronts - 1
    #         ) + 1.5  # position after custom bars or start fresh

    #     box_positions_random = [
    #         current_xtick_pos + (i - (num_params - 1) / 2) * bar_width * 1.2
    #         for i in range(num_params)
    #     ]

    #     bp_random = ax.boxplot(
    #         random_wavefront_data_for_boxplot,
    #         positions=box_positions_random,
    #         widths=bar_width * 1.1,
    #         patch_artist=True,
    #         showfliers=False,
    #         medianprops={"color": "black", "linewidth": 1.5},
    #         label=crlb_plot_labels,
    #     )
    #     for patch, color in zip(bp_random["boxes"], param_colors):
    #         patch.set_facecolor(color)
    #         patch.set_alpha(1.0)

    #     xtick_positions.append(np.mean(box_positions_random))
    #     xtick_labels.append(
    #         f"Random WFs\n({num_random_samples} samples)"
    #     )  # Corrected label
    # Final plot adjustments
    ax.set_xticks(xtick_positions)
    # Rotate labels only if there are many or they are long
    if len(xtick_labels) > 5 or any(len(label) > 10 for label in xtick_labels):
        ax.set_xticklabels(xtick_labels, rotation=25, ha="right", fontsize=10)
    else:
        ax.set_xticklabels(xtick_labels, fontsize=10)

    ax.set_title(figure_title, fontsize=14)
    ax.set_ylabel("Estimation Error (CRLB)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=1.0)

    # if (
    #     show_plane_waves or show_random_wavefronts
    # ):  # or data_custom.size > 0: # Check if data_custom is not empty
    #     # ax.set_yscale("log")
    #     pass

    #     plt.axhline(
    #         np.max(joint_CRB_custom),
    #         c="r",
    #         linestyle="--",
    #         label="Largest of optimized",
    #     )
    #     plt.axhline(
    #         np.min(joint_CRB_custom),
    #         c="k",
    #         linestyle="--",
    #         label="Smallest of optimized",
    #     )
    # # Create legend - only add labels for parameters once.
    # legend_handles = []
    # if (
    #     num_custom_wavefronts == 0
    #     and (show_plane_waves or show_random_wavefronts)
    #     and num_params > 0
    # ):
    #     for i in range(num_params):
    #         # Create dummy artists for legend if no bars were plotted for custom wavefronts
    #         legend_handles.append(
    #             plt.Rectangle(
    #                 (0, 0),
    #                 1,
    #                 1,
    #                 color=param_colors[i],
    #                 label=crlb_plot_labels[i],
    #             )
    #         )
    #     ax.legend(handles=legend_handles, title="Parameters", fontsize=10)
    # elif num_custom_wavefronts > 0:  # Legend from bar plot labels
    #     ax.legend(title="Parameters", fontsize=10)
    # ax.legend(title="Parameters", fontsize=10)

    # Prevent label duplicates
    handles, labels = ax.get_legend_handles_labels()
    labels_unique = dict(zip(labels, handles))
    ax.legend(
        labels_unique.values(),
        labels_unique.keys(),
        title="Parameters",
        fontsize=10,
        ncols=2,
    )

    plt.tight_layout()
    plt.show()


include_lossfunctions = [
    "x",
    "z",
    "theta",
    # "tr",
    "tr_norm",
    "tr_norm_nuissance",
    "det",
    "spec_rad",
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

fig, axes = plt.subplots(ncols=2)

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
    show_plane_waves=True,
    show_random_wavefronts=True,
    ax=axes[0],
)

visualize_crlb_comparison(
    loss_function_names=loss_function_names,
    param_legends=[
        "CRLB of x",
        "CRLB of z",
        r"CRLB of $\phi$",
    ],
    # optimized_crlbs_joint=optimized_crlbs_joint,
    # optimized_crlbs_scalar=optimized_crlbs_scalar,
    plane_wave_crlbs_joint=plane_wave_crlbs_joint,
    plane_wave_crlbs_scalar=plane_wave_crlbs_scalar,
    random_wavefronts_crlbs_joint=random_wavefronts_crlbs_joint,
    random_wavefronts_crlbs_scalar=random_wavefronts_crlbs_scalar,
    show_plane_waves=True,
    show_random_wavefronts=True,
    ax=axes[0],
)
