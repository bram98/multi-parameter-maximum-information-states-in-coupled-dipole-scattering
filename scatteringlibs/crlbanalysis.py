import matplotlib.pyplot as plt
import numpy as np
import torch  # For type hinting if F_op comes as torch tensor
from adjustText import adjust_text

from scatteringlibs.scatteringsimulation import (  # If needed for plane waves
    _flux_amplitude_2d,
)

# Potential import from optimizer if we decide to use its J calculation directly
# from .optimizer import Optimizer


def compute_crlb_from_J(J_matrix: np.ndarray) -> np.ndarray:
    """Calculates CRLB from a given Fisher Information Matrix J.
    CRLB_i = sqrt((J_inv)_ii).

    Args:
        J_matrix (np.ndarray): The Fisher Information Matrix (nparams, nparams).

    Returns:
        np.ndarray: Array of CRLB values for each parameter (nparams,).
    """
    if J_matrix.shape[0] != J_matrix.shape[1]:
        raise ValueError("J_matrix must be square.")
    try:
        J_inv = np.linalg.inv(J_matrix)
        crlb_values = np.sqrt(np.diag(J_inv))
    except np.linalg.LinAlgError:
        print(
            "Warning: J_matrix is singular or near singular. CRLBs will be Inf."
        )
        crlb_values = np.full(J_matrix.shape[0], np.inf)
    return crlb_values


def _compute_J_for_wavefront_numpy(
    wavefront_coeffs: np.ndarray, F_operator_tensor_np: np.ndarray
) -> np.ndarray:
    """
    Calculates the Fisher Information Matrix (J) for a given wavefront
    using a numpy Fisher operator tensor.

    Args:
        wavefront_coeffs (np.ndarray): Input wavefront (N_in,).
        F_operator_tensor_np (np.ndarray): Fisher operator tensor (nparams, nparams, N_in, N_in).

    Returns:
        np.ndarray: The (nparams, nparams) Fisher Information Matrix.
    """
    nparams = F_operator_tensor_np.shape[0]
    J_mat = np.zeros((nparams, nparams), dtype=float)
    E_c = wavefront_coeffs.conj()
    for mu in range(nparams):
        for nu in range(nparams):
            F_mu_nu_op = F_operator_tensor_np[mu, nu, :, :]
            val = np.dot(E_c, np.dot(F_mu_nu_op, wavefront_coeffs))
            J_mat[mu, nu] = val.real
    return J_mat


def normalize_wavefront(
    wavefront_coeffs: np.ndarray, photon_budget: float = 1.0
) -> np.ndarray:
    """
    Normalizes a wavefront (array of coefficients) to a given photon budget.
    (Copied from optimizer.py for self-containment if needed, or could be imported)
    """
    current_norm_sq = np.sum(np.abs(wavefront_coeffs) ** 2)
    if current_norm_sq == 0:  # Avoid division by zero for zero wavefront
        return wavefront_coeffs
    scale_factor = np.sqrt(photon_budget / current_norm_sq)
    return wavefront_coeffs * scale_factor


def compute_crlb_for_wavefront(
    wavefront_coeffs: np.ndarray,
    F_operator_tensor: np.ndarray,  # Expects numpy array
    photon_budget: float = 1.0,
) -> np.ndarray:
    """
    Calculates CRLB for a given wavefront.

    Args:
        wavefront_coeffs (np.ndarray): Input wavefront (N_in,).
        F_operator_tensor (np.ndarray): Fisher operator tensor (nparams, nparams, N_in, N_in).
                                        Assumed to be a NumPy array.
        photon_budget (float, optional): Photon budget for normalization. Defaults to 1.0.

    Returns:
        np.ndarray: Array of CRLB values for each parameter.
    """
    if isinstance(F_operator_tensor, torch.Tensor):
        # Convert to numpy if it's a torch tensor. Consider implications.
        F_operator_tensor = F_operator_tensor.detach().cpu().numpy()

    normalized_wf = normalize_wavefront(wavefront_coeffs, photon_budget)
    J_matrix = _compute_J_for_wavefront_numpy(normalized_wf, F_operator_tensor)
    return compute_crlb_from_J(J_matrix)


def compute_crlb_for_planewaves(
    sim_kwargs: dict,
    F_operator_tensor: np.ndarray,  # Expects numpy array
    photon_budget: float,
    kx_array: np.ndarray,
) -> np.ndarray:
    """
    Computes CRLB for a series of plane wave inputs across angles_deg.
    Adapted from the original script.
    """
    if isinstance(F_operator_tensor, torch.Tensor):
        F_operator_tensor = F_operator_tensor.detach().cpu().numpy()

    all_crlbs = []
    k0 = 2 * np.pi / sim_kwargs["lambda_0"]
    N_in = len(kx_array)

    for kx_val in kx_array:
        amplitude = _flux_amplitude_2d(k0, kx_val) * np.sqrt(photon_budget)
        idx = np.argmin(np.abs(kx_array - kx_val))
        E_plane_wave = np.zeros(N_in, dtype=complex)
        E_plane_wave[idx] = amplitude

        crlb_vals = compute_crlb_for_wavefront(
            E_plane_wave, F_operator_tensor, photon_budget
        )
        all_crlbs.append(crlb_vals)
    return np.array(all_crlbs)


def compute_crlb_for_random_wavefronts(
    F_operator_tensor: np.ndarray,  # Expects numpy array
    photon_budget: float,
    num_samples: int,
    N_in: int,
) -> np.ndarray:
    """
    Computes CRLB for a number of random wavefronts.
    Adapted from the original script.
    """
    if isinstance(F_operator_tensor, torch.Tensor):
        F_operator_tensor = F_operator_tensor.detach().cpu().numpy()

    all_crlbs = []
    for _ in range(num_samples):
        E_random = np.random.randn(N_in) + 1j * np.random.randn(N_in)
        # Normalization is handled by compute_crlb_for_wavefront
        crlb_vals = compute_crlb_for_wavefront(
            E_random, F_operator_tensor, photon_budget
        )
        all_crlbs.append(crlb_vals)
    return np.array(all_crlbs)


def get_crlb_data(
    optimized_wavefronts: dict,
    F_operator_tensor: np.ndarray,
    photon_budget: float,
    param_names: list,  # Though not directly used in this function, it's good for consistency
) -> dict:
    """
    Computes CRLB values for a dictionary of optimized wavefronts.

    Args:
        optimized_wavefronts (dict): Dictionary of {wavefront_name: wavefront_coeffs_np_array}.
        F_operator_tensor (np.ndarray): Fisher operator tensor (nparams, nparams, N_in, N_in), numpy format.
        photon_budget (float): Photon budget for normalization.
        param_names (list): List of parameter names (for API consistency, not used in current logic).

    Returns:
        dict: Dictionary of {wavefront_name: crlb_values_np_array}.
    """
    all_crlbs = {}
    for wf_name, wf_coeffs in optimized_wavefronts.items():
        crlbs = compute_crlb_for_wavefront(
            wf_coeffs, F_operator_tensor, photon_budget
        )
        all_crlbs[wf_name] = crlbs
    return all_crlbs


def visualize_crlb_comparison(
    optimized_wavefronts: dict,  # Dict of {name: wavefront_coeffs}
    F_operator_tensor: np.ndarray,  # The (nparams, nparams, N_in, N_in) Fisher op tensor (numpy)
    photon_budget: float,
    param_names: list,  # List of parameter names, e.g., ["shiftX", "shiftZ", "rotation"]
    sim_kwargs: dict = None,
    kx_array: np.ndarray = None,
    figure_title: str = "CRLB Comparison",
    show_plane_waves: bool = True,
    show_random_wavefronts: bool = True,
    num_random_samples: int = 100,
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
    if isinstance(F_operator_tensor, torch.Tensor):
        # Ensure F_operator_tensor is a NumPy array for internal functions
        F_operator_tensor = F_operator_tensor.detach().cpu().numpy()

    num_params = len(param_names)

    # Generate CRLB labels like "CRLB x", "CRLB z", "CRLB θ"
    crlb_plot_labels = []
    for p_name in param_names:
        if "shift" in p_name.lower():
            label_suffix = p_name.replace("shift", "").lower()
        elif "rotation" in p_name.lower():
            label_suffix = r"\theta"  # Use LaTeX for theta
        else:
            label_suffix = p_name.lower()
        crlb_plot_labels.append(f"CRLB of ${label_suffix}$")

    wave_names_custom = list(optimized_wavefronts.keys())
    data_custom = []
    for wname in wave_names_custom:
        wf_coeffs = optimized_wavefronts[wname]
        # compute_crlb_for_wavefront handles normalization internally
        crlb_vals = compute_crlb_for_wavefront(
            wf_coeffs, F_operator_tensor, photon_budget
        )
        data_custom.append(crlb_vals)

    if (
        data_custom
    ):  # Ensure data_custom is not empty before converting to numpy array
        data_custom = np.array(
            data_custom
        )  # Shape: (num_custom_wavefronts, num_params)
    else:
        data_custom = np.empty((0, num_params))

    # Prepare data for plane waves if requested
    plane_wave_data_for_boxplot = None
    if show_plane_waves:
        if not all([sim_kwargs, kx_array is not None]):
            print(
                "Warning: Missing sim_kwargs, or kx_array for plane wave CRLB calculation. Skipping."
            )
            show_plane_waves = False  # Disable if inputs missing
        else:
            plane_wave_crlbs_all_angles = compute_crlb_for_planewaves(
                sim_kwargs,
                F_operator_tensor,
                photon_budget,
                kx_array,
            )
            plane_wave_data_for_boxplot = [
                plane_wave_crlbs_all_angles[:, i] for i in range(num_params)
            ]

    # Prepare data for random wavefronts if requested
    random_wavefront_data_for_boxplot = None
    if show_random_wavefronts:
        if kx_array is None:  # N_in is derived from kx_array
            print(
                "Warning: Missing kx_array (for N_in) for random wavefront CRLB calculation. Skipping."
            )
            show_random_wavefronts = False
        else:
            N_in = len(kx_array)
            random_wavefront_crlbs_all_samples = (
                compute_crlb_for_random_wavefronts(
                    F_operator_tensor, photon_budget, num_random_samples, N_in
                )
            )  # Shape: (num_samples, num_params)
            random_wavefront_data_for_boxplot = [
                random_wavefront_crlbs_all_samples[:, i]
                for i in range(num_params)
            ]

    # --- Plotting ---
    num_custom_wavefronts = len(wave_names_custom)
    num_boxplot_groups = 0
    if show_plane_waves:
        num_boxplot_groups += 1
    if show_random_wavefronts:
        num_boxplot_groups += 1

    # Adjust figure width based on number of boxplot groups
    fig_width = 7 + 2.5 * num_boxplot_groups + 0.5 * num_custom_wavefronts
    fig, ax = plt.subplots(
        figsize=(max(10, fig_width), 7)
    )  # Ensure a minimum width

    x_idx_custom = np.arange(num_custom_wavefronts)
    bar_width = 0.2  # Width of individual bars for each parameter

    # Colors for parameters (e.g., CRLB_x, CRLB_z, CRLB_theta)
    # Ensure enough colors if num_params > 3
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    param_colors = [colors[i % len(colors)] for i in range(num_params)]

    # Bar charts for custom wavefronts
    if num_custom_wavefronts > 0:
        for i in range(num_params):  # For each parameter (x, z, theta)
            # Offset for each parameter's bar within a group
            offset = (i - (num_params - 1) / 2) * bar_width
            ax.bar(
                x_idx_custom + offset,
                data_custom[:, i],
                width=bar_width,
                label=crlb_plot_labels[i],
                color=param_colors[i],
                alpha=1.0,
            )

    # Determine positions for boxplots
    current_xtick_pos = (
        num_custom_wavefronts - 1
    )  # Last position of custom bars
    xtick_positions = list(x_idx_custom)
    xtick_labels = list(wave_names_custom)

    # Boxplots for plane waves
    if show_plane_waves and plane_wave_data_for_boxplot:
        current_xtick_pos += 1.0  # Spacing before boxplot group
        box_positions_plane = [
            current_xtick_pos + (i - (num_params - 1) / 2) * bar_width * 1.2
            for i in range(num_params)
        ]

        bp_plane = ax.boxplot(
            plane_wave_data_for_boxplot,
            positions=box_positions_plane,
            widths=bar_width * 1.1,  # Slightly wider boxes
            patch_artist=True,  # Fill with color
            showfliers=False,  # Hide outliers for cleaner plot
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bp_plane["boxes"], param_colors):
            patch.set_facecolor(color)
            patch.set_alpha(1.0)

        xtick_positions.append(
            np.mean(box_positions_plane)
        )  # Tick in middle of group
        xtick_labels.append("Plane Waves")

    # Boxplots for random wavefronts
    if show_random_wavefronts and random_wavefront_data_for_boxplot:
        # Adjust current_xtick_pos based on whether plane waves were plotted
        if (
            show_plane_waves and plane_wave_data_for_boxplot
        ):  # If plane waves were plotted
            current_xtick_pos = (
                np.mean(box_positions_plane)
                + (num_params / 2 * bar_width * 1.2)
                + 0.5
            )  # position after plane wave boxes
        else:  # If only custom bars were plotted or no plane waves
            current_xtick_pos = (
                num_custom_wavefronts - 1
            ) + 1.5  # position after custom bars or start fresh

        box_positions_random = [
            current_xtick_pos + (i - (num_params - 1) / 2) * bar_width * 1.2
            for i in range(num_params)
        ]

        bp_random = ax.boxplot(
            random_wavefront_data_for_boxplot,
            positions=box_positions_random,
            widths=bar_width * 1.1,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bp_random["boxes"], param_colors):
            patch.set_facecolor(color)
            patch.set_alpha(1.0)

        xtick_positions.append(np.mean(box_positions_random))
        xtick_labels.append(
            f"Random WFs\n({num_random_samples} samples)"
        )  # Corrected label

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

    if (
        show_plane_waves or show_random_wavefronts
    ):  # or data_custom.size > 0: # Check if data_custom is not empty
        ax.set_yscale("log")

    # Create legend - only add labels for parameters once.
    legend_handles = []
    if (
        num_custom_wavefronts == 0
        and (show_plane_waves or show_random_wavefronts)
        and num_params > 0
    ):
        for i in range(num_params):
            # Create dummy artists for legend if no bars were plotted for custom wavefronts
            legend_handles.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    color=param_colors[i],
                    label=crlb_plot_labels[i],
                )
            )
        ax.legend(handles=legend_handles, title="Parameters", fontsize=10)
    elif num_custom_wavefronts > 0:  # Legend from bar plot labels
        ax.legend(title="Parameters", fontsize=10)

    plt.tight_layout()
    plt.show()


def _plot_single_pareto_frontier(
    ax,
    crlb_data_dict: dict,
    param_idx1: int,
    param_idx2: int,
    param_names_list: list,
    wavefront_categories_style: dict,
    wavefront_labels_latex_map: dict,
    plot_title: str,
    xlabel: str,
    ylabel: str,
):
    """Helper function to plot a single CRLB vs CRLB Pareto-like frontier."""
    texts = []
    plotted_categories = set()

    for wf_name, crlbs_values in crlb_data_dict.items():
        val1 = crlbs_values[param_idx1]
        val2 = crlbs_values[param_idx2]

        # Determine style and category label for legend (similar to plot_pareto_frontiers_individual)
        style_info = {}
        legend_label_for_category = wf_name  # Default

        if wf_name in wavefront_categories_style:
            style_info = wavefront_categories_style[wf_name]
            legend_label_for_category = style_info.get("label", wf_name)
        else:
            found_category = False
            for cat_key, cat_style_info in wavefront_categories_style.items():
                if (
                    cat_key in wf_name
                ):  # e.g., "SLM" is in "SLM_Optimized_Mode1"
                    style_info = cat_style_info
                    legend_label_for_category = style_info.get(
                        "label", cat_key
                    )
                    found_category = True
                    break
            if not found_category:
                style_info = wavefront_categories_style.get(
                    "default",
                    {"marker": "x", "color": "grey", "label": "Other"},
                )
                legend_label_for_category = style_info.get("label", "Other")

        marker = style_info.get("marker", "o")
        color = style_info.get("color", "k")

        current_legend_label = None
        if legend_label_for_category not in plotted_categories:
            current_legend_label = legend_label_for_category
            plotted_categories.add(legend_label_for_category)

        ax.plot(
            val2,  # X-axis (param_idx2)
            val1,  # Y-axis (param_idx1)
            marker=marker,
            color=color,
            linestyle="None",
            markersize=8,
            label=current_legend_label,
        )
        label_text = wavefront_labels_latex_map.get(
            wf_name, wf_name.replace("_", " ")
        )
        texts.append(ax.text(val2, val1, label_text, color=color, fontsize=9))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(plot_title)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    if plotted_categories:  # Only show legend if there's something to label
        ax.legend(title="Wavefront Types")

    # Ensure adjust_text is available (imported at the top of the file)
    if (
        "adjust_text" in globals()
        and callable(globals()["adjust_text"])
        and texts
    ):
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
        )
    elif not texts:
        pass  # No text to adjust
    else:
        print(
            "Warning: adjust_text function not available for text label adjustment."
        )


def plot_crlb_vs_crlb(
    crlb_data_dict: dict,
    param_idx1: int,
    param_idx2: int,
    param_names_list: list,
    wavefront_categories_style: dict,
    wavefront_labels_latex_map: dict,
    figure_title_prefix: str = "",
):
    """
    Plots one CRLB parameter against another for a set of wavefronts.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    p1_name_full = param_names_list[param_idx1]
    p2_name_full = param_names_list[param_idx2]

    # Create short names for title and labels, e.g., "x" from "shiftX", "θ" from "rotation"
    p1_name_short = p1_name_full.replace("shift", "").replace("rotation", "θ")
    p2_name_short = p2_name_full.replace("shift", "").replace("rotation", "θ")

    # Construct title and labels
    title = f"{figure_title_prefix}CRLB$_{{{p1_name_short}}}$ vs CRLB$_{{{p2_name_short}}}$"
    ylabel_text = f"CRLB$_{{{p1_name_short}}}$ ({p1_name_full})"
    xlabel_text = f"CRLB$_{{{p2_name_short}}}$ ({p2_name_full})"

    _plot_single_pareto_frontier(
        ax,
        crlb_data_dict,
        param_idx1,
        param_idx2,
        param_names_list,  # Pass along for potential future use in _plot_single_pareto_frontier
        wavefront_categories_style,
        wavefront_labels_latex_map,
        title,
        xlabel_text,
        ylabel_text,
    )

    plt.tight_layout()
    plt.show()


def plot_pareto_frontiers(
    optimized_wavefronts_dict: dict,
    F_operator_tensor_np: np.ndarray,
    photon_budget_val: float,
    param_names_list: list,
    params_to_plot_indices: list,
    wavefront_categories_style: dict,
    wavefront_labels_latex_map: dict,
    filepath: str = None,
    dpi: int = 300,
):
    """
    Calculates CRLBs and plots a specific Pareto frontier based on two selected parameter indices.

    Args:
        optimized_wavefronts_dict (dict): Wavefronts {name: coeffs}.
        F_operator_tensor_np (np.ndarray): Fisher operator tensor.
        photon_budget_val (float): Photon budget.
        param_names_list (list): List of all parameter names.
        params_to_plot_indices (list): List containing exactly two indices of the parameters to plot.
        wavefront_categories_style (dict): Styling info for wavefront categories.
        wavefront_labels_latex_map (dict): LaTeX labels for wavefront points.
    """
    if len(params_to_plot_indices) != 2:
        raise ValueError(
            "params_to_plot_indices must contain exactly two indices."
        )

    idx1, idx2 = params_to_plot_indices[0], params_to_plot_indices[1]

    # Calculate CRLBs for all wavefronts
    all_crlbs = get_crlb_data(
        optimized_wavefronts_dict,
        F_operator_tensor_np,
        photon_budget_val,
        param_names_list,
    )

    # Plot the specified CRLB vs CRLB graph
    plot_crlb_vs_crlb(
        all_crlbs,
        idx1,
        idx2,
        param_names_list,
        wavefront_categories_style,
        wavefront_labels_latex_map,
        figure_title_prefix="Pareto: ",  # Standard prefix for these plots
    )

    if filepath:
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_pareto_frontiers_individual(
    optimized_wavefronts: dict,  # {name: wavefront_coeffs_np_array}
    F_operator_tensor: np.ndarray,  # Fisher op tensor (nparams, nparams, N_in, N_in), numpy
    photon_budget: float,
    param_names: list,  # List of param names e.g. ["shiftX", "shiftZ", "rotation"]
    wavefront_categories: dict,  # {wf_name_or_category: {'marker': 'o', 'color': 'r', 'category_label': 'Category A'}}
    wavefront_labels_latex: dict,  # {wf_name: "latex_label_for_wf"}
):
    """
    Plots two Pareto frontiers for optimized wavefronts:
    1. (CRLB_x + CRLB_z) vs. CRLB_theta
    2. CRLB_x vs. CRLB_z

    Args:
        optimized_wavefronts (dict): Dictionary of wavefronts {name: coefficients}.
        F_operator_tensor (np.ndarray): Fisher operator tensor.
        photon_budget (float): Photon budget for normalization.
        param_names (list): List of parameter names.
        wavefront_categories (dict): Styling for wavefronts based on name or category.
                                     Example: {'SLM': {'marker': 's', 'color': 'blue', 'label': 'SLM WFs'},
                                               'Optimized': {'marker': 'o', 'color': 'red', 'label': 'Optimized WFs'}}
                                     Wavefront names in optimized_wavefronts should map to a key in this dict,
                                     or this dict can contain direct styling for wf_names.
                                     For simplicity, we'll assume wavefront_categories contains a key for each
                                     wavefront name in optimized_wavefronts or a category it belongs to.
        wavefront_labels_latex (dict): LaTeX labels for each wavefront point.
    """
    # Parameter indices - assuming standard order for x, z, theta from param_names
    # This needs to be robust. Let's find indices based on names.
    idx_x, idx_z, idx_theta = -1, -1, -1
    for i, name in enumerate(param_names):
        if "x" in name.lower():  # Assuming "shiftX" or similar
            idx_x = i
        elif "z" in name.lower():  # Assuming "shiftZ" or similar
            idx_z = i
        elif (
            "theta" in name.lower() or "rotation" in name.lower()
        ):  # Assuming "rotation" or "theta"
            idx_theta = i

    if idx_x == -1 or idx_z == -1 or idx_theta == -1:
        print(
            "Warning: Could not reliably determine indices for x, z, or theta parameters "
            "from param_names. Please check input. Using first three params if available."
        )
        # Fallback logic, might not be correct for all cases.
        if len(param_names) >= 1:
            idx_x = 0
        if len(param_names) >= 2:
            idx_z = 1
        if len(param_names) >= 3:
            idx_theta = 2
        if idx_x == -1 or idx_z == -1 or idx_theta == -1:  # Still not found
            raise ValueError(
                "Parameter names for x, z, theta not found or insufficient parameters."
            )

    all_crlbs = {}
    for wf_name, wf_coeffs in optimized_wavefronts.items():
        crlbs = compute_crlb_for_wavefront(
            wf_coeffs, F_operator_tensor, photon_budget
        )
        all_crlbs[wf_name] = crlbs

    # --- Plot 1: (CRLB_x + CRLB_z) vs CRLB_theta ---
    plt.figure(figsize=(8, 6))
    texts = []
    plotted_categories = set()  # For legend

    for wf_name, crlbs_values in all_crlbs.items():
        crlb_x = crlbs_values[idx_x]
        crlb_z = crlbs_values[idx_z]
        crlb_theta = crlbs_values[idx_theta]

        # Determine category and style
        style = {}
        category_label_for_legend = (
            wf_name  # Default legend label to wavefront name
        )

        # Attempt to find style:
        # 1. Direct match for wf_name in wavefront_categories
        if wf_name in wavefront_categories:
            style = wavefront_categories[wf_name]
            category_label_for_legend = style.get(
                "label", wf_name
            )  # Use provided label if exists
        else:  # 2. Check if any category key is part of the wf_name (e.g. wf_name="SLM_optimized_1", category_key="SLM")
            found_category = False
            for cat_key, cat_style in wavefront_categories.items():
                if cat_key in wf_name:
                    style = cat_style
                    category_label_for_legend = style.get("label", cat_key)
                    found_category = True
                    break
            if not found_category:  # 3. Use a default style if no match
                style = wavefront_categories.get(
                    "default",
                    {"marker": "o", "color": "gray", "label": "Other"},
                )
                category_label_for_legend = style.get("label", "Other")

        marker = style.get("marker", "o")
        color = style.get("color", "k")

        legend_label = None
        if category_label_for_legend not in plotted_categories:
            legend_label = category_label_for_legend
            plotted_categories.add(category_label_for_legend)

        plt.plot(
            crlb_theta,
            crlb_x + crlb_z,
            marker=marker,
            color=color,
            linestyle="None",
            markersize=8,
            label=legend_label,
        )
        label = wavefront_labels_latex.get(wf_name, wf_name.replace("_", " "))
        texts.append(
            plt.text(
                crlb_theta, crlb_x + crlb_z, label, color=color, fontsize=9
            )
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(
        f"CRLB$_{{\\theta}}$ ({param_names[idx_theta].replace('rotation', 'θ').replace('shift', '')})"
    )
    plt.ylabel(
        f"CRLB$_x$ ({param_names[idx_x].replace('shift', '')}) + "
        f"CRLB$_z$ ({param_names[idx_z].replace('shift', '')})"
    )
    plt.title("Pareto Frontier: Combined Positional vs. Rotational Error")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(title="Wavefront Types")

    if adjust_text and texts:
        adjust_text(
            texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.5)
        )

    plt.tight_layout()
    plt.show()

    # --- Plot 2: CRLB_x vs CRLB_z ---
    plt.figure(figsize=(8, 6))
    texts_plot2 = []
    plotted_categories_plot2 = set()

    for wf_name, crlbs_values in all_crlbs.items():
        crlb_x = crlbs_values[idx_x]
        crlb_z = crlbs_values[idx_z]

        style = {}
        category_label_for_legend = wf_name
        if wf_name in wavefront_categories:
            style = wavefront_categories[wf_name]
            category_label_for_legend = style.get("label", wf_name)
        else:
            found_category = False
            for cat_key, cat_style in wavefront_categories.items():
                if cat_key in wf_name:
                    style = cat_style
                    category_label_for_legend = style.get("label", cat_key)
                    found_category = True
                    break
            if not found_category:
                style = wavefront_categories.get(
                    "default",
                    {"marker": "o", "color": "gray", "label": "Other"},
                )
                category_label_for_legend = style.get("label", "Other")

        marker = style.get("marker", "o")
        color = style.get("color", "k")

        legend_label_plot2 = None
        if category_label_for_legend not in plotted_categories_plot2:
            legend_label_plot2 = category_label_for_legend
            plotted_categories_plot2.add(category_label_for_legend)

        plt.plot(
            crlb_z,  # X-axis for this plot
            crlb_x,  # Y-axis for this plot
            marker=marker,
            color=color,
            linestyle="None",
            markersize=8,
            label=legend_label_plot2,
        )
        label = wavefront_labels_latex.get(wf_name, wf_name.replace("_", " "))
        texts_plot2.append(
            plt.text(crlb_z, crlb_x, label, color=color, fontsize=9)
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(f"CRLB$_z$ ({param_names[idx_z].replace('shift', '')})")
    plt.ylabel(f"CRLB$_x$ ({param_names[idx_x].replace('shift', '')})")
    plt.title("Pareto Frontier: Positional Error X vs. Z")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(title="Wavefront Types")

    if adjust_text and texts_plot2:
        adjust_text(
            texts_plot2, arrowprops=dict(arrowstyle="-", color="black", lw=0.5)
        )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------
# -- Newly added functions for Fisher Matrix and Condition Number Visualization --
# ------------------------------------------------------------------------------------


def _get_plot_param_labels(param_names_list):
    """Helper to generate short labels for plot ticks from full parameter names."""
    plot_labels = []
    for p_name in param_names_list:
        if "shiftx" in p_name.lower():
            plot_labels.append("x")
        elif "shiftz" in p_name.lower():
            plot_labels.append("z")
        elif "rotation" in p_name.lower():
            plot_labels.append(r"$\theta$")
        else:
            plot_labels.append(p_name[:3])  # Fallback to first 3 chars
    return plot_labels


def plot_fisher_comparison(
    Js_map,
    J_invs_map,
    cond_nums_map,
    param_names_global,
    wavefronts_to_show=None,
    figure_title="Fisher Matrices, Inverses, and Condition Numbers",
):
    """
    Visualizes Fisher Information matrices (J), their inverses (J^-1), and condition numbers
    for a selection of wavefronts.
    """
    if wavefronts_to_show is None:
        # Plot all wavefronts for which J is available
        wavefronts_to_plot = list(Js_map.keys())
    else:
        # Filter to plot only requested and available wavefronts
        wavefronts_to_plot = [wf for wf in wavefronts_to_show if wf in Js_map]

    if not wavefronts_to_plot:
        print(
            "No valid wavefronts selected or available to plot in plot_fisher_comparison."
        )
        return

    n_wave = len(wavefronts_to_plot)
    plot_param_labels = _get_plot_param_labels(param_names_global)
    n_params = len(param_names_global)

    # --- Figure 1: J and J^-1 matrices ---
    fig_matrices = plt.figure(
        figsize=(10, 3 * n_wave + 0.5)
    )  # +0.5 for title space

    try:
        import matplotlib.gridspec as gridspec
    except ImportError:
        print(
            "matplotlib.gridspec not found. Please ensure matplotlib is installed correctly."
        )
        return

    gs_main = gridspec.GridSpec(
        n_wave, 1, figure=fig_matrices, hspace=0.5 if n_wave > 1 else 0.3
    )

    for i, wf_name in enumerate(wavefronts_to_plot):
        J_mat = Js_map.get(wf_name)
        inv_mat = J_invs_map.get(wf_name)
        cond_num_val = cond_nums_map.get(wf_name, float("nan"))

        row_gs = gs_main[i].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.4)

        # Plot J
        axJ = fig_matrices.add_subplot(row_gs[0])
        if J_mat is not None and not np.all(np.isnan(J_mat)):
            imJ = axJ.imshow(
                J_mat, cmap="viridis", aspect="equal", vmin=None, vmax=None
            )
            axJ.set_xticks(np.arange(n_params))
            axJ.set_yticks(np.arange(n_params))
            axJ.set_xticklabels(plot_param_labels, fontsize=8)
            axJ.set_yticklabels(plot_param_labels, fontsize=8)
            axJ.set_title(f"J: {wf_name}", fontsize=10)
            cbJ = plt.colorbar(imJ, ax=axJ, fraction=0.046, pad=0.04)
            cbJ.formatter.set_powerlimits((-2, 2))
            cbJ.ax.tick_params(labelsize=8)
            cbJ.update_ticks()
        else:
            axJ.text(
                0.5,
                0.5,
                "J N/A",
                ha="center",
                va="center",
                transform=axJ.transAxes,
                fontsize=9,
            )
            axJ.set_xticks([])
            axJ.set_yticks([])
            axJ.set_title(f"J: {wf_name} (N/A)", fontsize=10)

        # Plot J^-1
        axInv = fig_matrices.add_subplot(row_gs[1])
        if inv_mat is not None and not np.all(np.isnan(inv_mat)):
            imInv = axInv.imshow(
                inv_mat, cmap="magma", aspect="equal", vmin=None, vmax=None
            )
            axInv.set_xticks(np.arange(n_params))
            axInv.set_yticks(np.arange(n_params))
            axInv.set_xticklabels(plot_param_labels, fontsize=8)
            axInv.set_yticklabels(plot_param_labels, fontsize=8)
            title_str = f"$J^{{-1}}$: {wf_name}"
            if not np.isnan(cond_num_val):
                title_str += f"\nCond: {cond_num_val:.1e}"
            axInv.set_title(title_str, fontsize=10)
            cbInv = plt.colorbar(imInv, ax=axInv, fraction=0.046, pad=0.04)
            cbInv.formatter.set_powerlimits((-2, 2))
            cbInv.ax.tick_params(labelsize=8)
            cbInv.update_ticks()
        else:
            axInv.text(
                0.5,
                0.5,
                "$J^{-1}$ N/A",
                ha="center",
                va="center",
                transform=axInv.transAxes,
                fontsize=9,
            )
            axInv.set_xticks([])
            axInv.set_yticks([])
            title_str = f"$J^{{-1}}$: {wf_name} (N/A)"
            if not np.isnan(cond_num_val) and np.isinf(cond_num_val):
                title_str += f"\nCond: $\infty$"
            axInv.set_title(title_str, fontsize=10)

    if figure_title:
        fig_matrices.suptitle(figure_title, fontsize=14, y=0.99)

    plt.subplots_adjust(top=0.92 if figure_title else 0.95)

    # --- Figure 2: Condition Numbers ---
    fig_cond, ax_cond = plt.subplots(figsize=(max(6, n_wave * 0.6), 5))

    cond_values_to_plot = []
    valid_wf_names_for_cond = []
    for wf_name in wavefronts_to_plot:
        cond_val = cond_nums_map.get(wf_name)
        if cond_val is not None and not np.isnan(cond_val):
            if np.isinf(cond_val):
                cond_values_to_plot.append(
                    1e18
                )  # Placeholder for Inf on log scale
            else:
                cond_values_to_plot.append(cond_val)
            valid_wf_names_for_cond.append(wf_name)
        else:
            cond_values_to_plot.append(np.nan)
            valid_wf_names_for_cond.append(wf_name + " (N/A)")

    x_pos = np.arange(len(valid_wf_names_for_cond))
    finite_cond_indices = [
        idx
        for idx, val in enumerate(cond_values_to_plot)
        if np.isfinite(val) and val > 0
    ]

    if not finite_cond_indices:
        ax_cond.text(
            0.5,
            0.5,
            "No valid finite condition numbers to plot.",
            ha="center",
            va="center",
            transform=ax_cond.transAxes,
        )
        ax_cond.set_title("Condition Numbers (No Data)")
    else:
        plot_x_pos = np.array(x_pos)[finite_cond_indices]
        plot_cond_values = np.array(cond_values_to_plot)[finite_cond_indices]
        plot_names = np.array(valid_wf_names_for_cond)[finite_cond_indices]

        bars = ax_cond.bar(
            plot_x_pos, plot_cond_values, color="teal", width=0.6
        )
        ax_cond.set_xticks(plot_x_pos)
        ax_cond.set_xticklabels(
            plot_names, rotation=45, ha="right", fontsize=9
        )
        ax_cond.set_ylabel("Condition Number of J (log scale)")
        ax_cond.set_title("Condition Numbers of Fisher Matrices")
        try:
            ax_cond.set_yscale("log")
            for bar_idx, bar_val in enumerate(plot_cond_values):
                ax_cond.text(
                    plot_x_pos[bar_idx],
                    bar_val * 1.1,
                    f"{bar_val:.1e}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )
        except ValueError as e:
            ax_cond.set_yscale("linear")
            ax_cond.set_ylabel("Condition Number of J (linear scale)")
            print(
                f"Warning: Could not use log scale for condition numbers due to data values (e.g. zeros/NaNs not fully filtered): {e}. Using linear scale."
            )
            for bar_idx, bar_val in enumerate(plot_cond_values):
                ax_cond.text(
                    plot_x_pos[bar_idx],
                    bar_val,
                    f"{bar_val:.1e}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

    ax_cond.grid(True, which="both", ls=":", alpha=0.7)
    fig_cond.tight_layout()

    plt.show()
