import numpy as np
import torch # For type hinting if F_op comes as torch tensor
from scatteringlibs.scatteringsimulation import _flux_amplitude_2d # If needed for plane waves
import matplotlib.pyplot as plt
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
        print("Warning: J_matrix is singular or near singular. CRLBs will be Inf.")
        crlb_values = np.full(J_matrix.shape[0], np.inf)
    return crlb_values

def _compute_J_for_wavefront_numpy(
    wavefront_coeffs: np.ndarray,
    F_operator_tensor_np: np.ndarray
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
    if current_norm_sq == 0: # Avoid division by zero for zero wavefront
        return wavefront_coeffs
    scale_factor = np.sqrt(photon_budget / current_norm_sq)
    return wavefront_coeffs * scale_factor

def compute_crlb_for_wavefront(
    wavefront_coeffs: np.ndarray,
    F_operator_tensor: np.ndarray, # Expects numpy array
    photon_budget: float = 1.0
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
    # base_dipoles_combined: np.ndarray, # This seems not directly used by original func for CRLB calc itself once F_op is given
    angles_deg: np.ndarray,
    F_operator_tensor: np.ndarray, # Expects numpy array
    photon_budget: float,
    kx_array: np.ndarray
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

    # Make sure _flux_amplitude_2d is available, e.g. by importing from scatteringsimulation
    # from scatteringlibs.scatteringsimulation import _flux_amplitude_2d # (Ensure this import is at top of file)

    for angle in angles_deg:
        theta_rad = np.deg2rad(angle)
        kx_val = k0 * np.sin(theta_rad) # Corrected variable name

        # Use _flux_amplitude_2d from scatteringsimulation
        # This function might need to be exposed or copied if not directly importable
        # For now, assuming it can be imported or is made available.
        # If CoupledDipoleSimulator is needed for this, the design changes.
        # The original script's _flux_amplitude_2d doesn't rely on an instance of CoupledDipoleSimulator.
        amplitude = _flux_amplitude_2d(k0, kx_val) * np.sqrt(photon_budget)

        idx = np.argmin(np.abs(kx_array - kx_val)) # Corrected variable name
        E_plane_wave = np.zeros(N_in, dtype=complex)
        E_plane_wave[idx] = amplitude

        crlb_vals = compute_crlb_for_wavefront(E_plane_wave, F_operator_tensor, photon_budget)
        all_crlbs.append(crlb_vals)
    return np.array(all_crlbs)

def compute_crlb_for_random_wavefronts(
    F_operator_tensor: np.ndarray, # Expects numpy array
    photon_budget: float,
    num_samples: int,
    N_in: int
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
        crlb_vals = compute_crlb_for_wavefront(E_random, F_operator_tensor, photon_budget)
        all_crlbs.append(crlb_vals)
    return np.array(all_crlbs)

def visualize_crlb_comparison(
    optimized_wavefronts: dict,      # Dict of {name: wavefront_coeffs}
    F_operator_tensor: np.ndarray,   # The (nparams, nparams, N_in, N_in) Fisher op tensor (numpy)
    photon_budget: float,
    param_names: list,               # List of parameter names, e.g., ["shiftX", "shiftZ", "rotation"]
    # For plane/random waves (optional parts)
    sim_kwargs: dict = None,
    # base_dipoles_combined: np.ndarray = None, # Not directly used if F_op is global for these calcs
    angles_deg: np.ndarray = None,
    kx_array: np.ndarray = None,
    # Plotting options
    figure_title: str = "CRLB Comparison",
    show_plane_waves: bool = True,
    show_random_wavefronts: bool = True,
    num_random_samples: int = 100
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
            label_suffix = r"\theta" # Use LaTeX for theta
        else:
            label_suffix = p_name.lower()
        crlb_plot_labels.append(f"CRLB {label_suffix}")


    wave_names_custom = list(optimized_wavefronts.keys())
    data_custom = []
    for wname in wave_names_custom:
        wf_coeffs = optimized_wavefronts[wname]
        # compute_crlb_for_wavefront handles normalization internally
        crlb_vals = compute_crlb_for_wavefront(wf_coeffs, F_operator_tensor, photon_budget)
        data_custom.append(crlb_vals)

    if data_custom: # Ensure data_custom is not empty before converting to numpy array
        data_custom = np.array(data_custom) # Shape: (num_custom_wavefronts, num_params)
    else:
        data_custom = np.empty((0, num_params))


    # Prepare data for plane waves if requested
    plane_wave_data_for_boxplot = None
    if show_plane_waves:
        if not all([sim_kwargs, angles_deg is not None, kx_array is not None]):
            print("Warning: Missing sim_kwargs, angles_deg, or kx_array for plane wave CRLB calculation. Skipping.")
            show_plane_waves = False # Disable if inputs missing
        else:
            plane_wave_crlbs_all_angles = compute_crlb_for_planewaves(
                sim_kwargs, angles_deg, F_operator_tensor, photon_budget, kx_array
            ) # Shape: (num_angles, num_params)
            # For boxplot, need list of arrays, one per parameter
            plane_wave_data_for_boxplot = [plane_wave_crlbs_all_angles[:, i] for i in range(num_params)]

    # Prepare data for random wavefronts if requested
    random_wavefront_data_for_boxplot = None
    if show_random_wavefronts:
        if kx_array is None: # N_in is derived from kx_array
             print("Warning: Missing kx_array (for N_in) for random wavefront CRLB calculation. Skipping.")
             show_random_wavefronts = False
        else:
            N_in = len(kx_array)
            random_wavefront_crlbs_all_samples = compute_crlb_for_random_wavefronts(
                F_operator_tensor, photon_budget, num_random_samples, N_in
            ) # Shape: (num_samples, num_params)
            random_wavefront_data_for_boxplot = [random_wavefront_crlbs_all_samples[:, i] for i in range(num_params)]

    # --- Plotting ---
    num_custom_wavefronts = len(wave_names_custom)
    num_boxplot_groups = 0
    if show_plane_waves: num_boxplot_groups += 1
    if show_random_wavefronts: num_boxplot_groups += 1

    # Adjust figure width based on number of boxplot groups
    fig_width = 7 + 2.5 * num_boxplot_groups + 0.5 * num_custom_wavefronts
    fig, ax = plt.subplots(figsize=(max(10, fig_width), 6)) # Ensure a minimum width

    x_idx_custom = np.arange(num_custom_wavefronts)
    bar_width = 0.2  # Width of individual bars for each parameter

    # Colors for parameters (e.g., CRLB_x, CRLB_z, CRLB_theta)
    # Ensure enough colors if num_params > 3
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    param_colors = [colors[i % len(colors)] for i in range(num_params)]

    # Bar charts for custom wavefronts
    if num_custom_wavefronts > 0 :
        for i in range(num_params): # For each parameter (x, z, theta)
            # Offset for each parameter's bar within a group
            offset = (i - (num_params - 1) / 2) * bar_width
            ax.bar(
                x_idx_custom + offset,
                data_custom[:, i],
                width=bar_width,
                label=crlb_plot_labels[i],
                color=param_colors[i],
                alpha=0.8
            )

    # Determine positions for boxplots
    current_xtick_pos = num_custom_wavefronts -1 # Last position of custom bars
    xtick_positions = list(x_idx_custom)
    xtick_labels = list(wave_names_custom)

    # Boxplots for plane waves
    if show_plane_waves and plane_wave_data_for_boxplot:
        current_xtick_pos += 1.5 # Spacing before boxplot group
        box_positions_plane = [current_xtick_pos + (i - (num_params - 1) / 2) * bar_width * 1.2 for i in range(num_params)]

        bp_plane = ax.boxplot(
            plane_wave_data_for_boxplot,
            positions=box_positions_plane,
            widths=bar_width * 1.1, # Slightly wider boxes
            patch_artist=True, # Fill with color
            showfliers=False, # Hide outliers for cleaner plot
            medianprops={'color': 'black', 'linewidth': 1.5}
        )
        for patch, color in zip(bp_plane['boxes'], param_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        xtick_positions.append(np.mean(box_positions_plane)) # Tick in middle of group
        xtick_labels.append("Plane Waves")

    # Boxplots for random wavefronts
    if show_random_wavefronts and random_wavefront_data_for_boxplot:
        # Adjust current_xtick_pos based on whether plane waves were plotted
        if show_plane_waves and plane_wave_data_for_boxplot: # If plane waves were plotted
             current_xtick_pos = np.mean(box_positions_plane) + (num_params/2 * bar_width * 1.2) + 0.5 # position after plane wave boxes
        else: # If only custom bars were plotted or no plane waves
             current_xtick_pos = (num_custom_wavefronts -1) + 1.5 # position after custom bars or start fresh

        box_positions_random = [current_xtick_pos + (i - (num_params - 1) / 2) * bar_width * 1.2 for i in range(num_params)]

        bp_random = ax.boxplot(
            random_wavefront_data_for_boxplot,
            positions=box_positions_random,
            widths=bar_width * 1.1,
            patch_artist=True,
            showfliers=False,
            medianprops={'color': 'black', 'linewidth': 1.5}
        )
        for patch, color in zip(bp_random['boxes'], param_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        xtick_positions.append(np.mean(box_positions_random))
        xtick_labels.append(f"Random WFs\n({num_random_samples} samples)") # Corrected label

    # Final plot adjustments
    ax.set_xticks(xtick_positions)
    # Rotate labels only if there are many or they are long
    if len(xtick_labels) > 5 or any(len(label) > 10 for label in xtick_labels):
        ax.set_xticklabels(xtick_labels, rotation=25, ha="right", fontsize=10)
    else:
        ax.set_xticklabels(xtick_labels, fontsize=10)

    ax.set_title(figure_title, fontsize=14)
    ax.set_ylabel("Estimation Error (CRLB)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.7)

    if show_plane_waves or show_random_wavefronts: # or data_custom.size > 0: # Check if data_custom is not empty
        ax.set_yscale("log")

    # Create legend - only add labels for parameters once.
    legend_handles = []
    if num_custom_wavefronts == 0 and (show_plane_waves or show_random_wavefronts) and num_params > 0 :
        for i in range(num_params):
            # Create dummy artists for legend if no bars were plotted for custom wavefronts
            legend_handles.append(plt.Rectangle((0,0),1,1, color=param_colors[i], label=crlb_plot_labels[i]))
        ax.legend(handles=legend_handles, title="Parameters", fontsize=10)
    elif num_custom_wavefronts > 0 : # Legend from bar plot labels
        ax.legend(title="Parameters", fontsize=10)
    # If no custom WFs and no boxplots, legend might not be needed or handled differently.
    # Current logic: if custom WFs, bars create labels. If no custom WFs but boxplots, dummy artists.
    # If nothing is plotted, legend is not explicitly created, which is fine.

    plt.tight_layout()
    plt.show()
