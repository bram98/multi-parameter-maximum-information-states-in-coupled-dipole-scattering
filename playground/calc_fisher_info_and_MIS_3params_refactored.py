# playground/calc_fisher_info_and_MIS_3params_refactored.py

# %% Imports and functions
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs import placedipoles  # Import the whole module
from scatteringlibs import crlbanalysis
from scatteringlibs.crlbanalysis import plot_fisher_comparison
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator
from scatteringlibs.lossfunctions import determinant_of_J, maximize_min_eig_J, minimize_trace_of_J_inv, \
                                         loglin_loss, offdiag_loss, subdet_loss, \
                                         determinant_of_marginal_J, minimize_trace_of_marginal_J_inv, \
                                         maximize_min_eig_marginal_J


# %% 1) define dipole configurations
movable_dipoles_line = placedipoles.place_line_of_dipoles(
    center=(5.0, 0.0), number=7
)

frozen_dipoles_line = placedipoles.place_line_of_dipoles(
    center=(3.0, 0.0), number=7
)

# %% 2) simulator geometry and input space
total_Ns_sim = movable_dipoles_line.shape[1]
if frozen_dipoles_line is not None and frozen_dipoles_line.size > 0:
    total_Ns_sim += frozen_dipoles_line.shape[1]

sim_kwargs = dict(
    lambda_0=1.0,
    Lz=8.0,
    Lx=8.0,
    Ns=total_Ns_sim,  # Ns in sim_kwargs should be the total number of dipoles
    seed=123,
    grid_points_x=300,  # Number of points for field calculation (output basis for S-matrix)
)
max_angle = 80  # degrees
angles_deg = np.linspace(-max_angle, max_angle, 100)  # Input angle channels

k0 = 2 * np.pi / sim_kwargs["lambda_0"]
kx_array = k0 * np.sin(np.deg2rad(angles_deg))

# %% 3) Build partial derivatives for each param => dS[mu]
param_names = ["shiftX", "shiftZ", "rotation"]
delta_param = 1e-3

print("Calculating dS for 3 parameters...")
dS_tensor = placedipoles.calculate_dS_for_3_parameters(
    sim_kwargs,
    angles_deg,
    movable_dipoles=movable_dipoles_line,
    frozen_dipoles=frozen_dipoles_line,
    delta_param=delta_param,
    flux_normalize=True,
)
print(f"Built dS tensor with shape: {dS_tensor.shape}")
# %% 4) Build the Fisher-operator array and set global variables
fisher_calc = FisherMatrix(
    dS=dS_tensor, param_names=param_names, delta=delta_param
)
F_tensor_operator = fisher_calc.F

print(
    "Built F[mu,nu,:,:], shape =",
    F_tensor_operator.shape,
    f" (should be 3,3,{len(angles_deg)},{len(angles_deg)})",
)

# placeholder for base_sim for visualization
_sim_params_vis = sim_kwargs.copy()

# Correctly combine frozen and movable dipoles for visualization.
if frozen_dipoles_line is not None and frozen_dipoles_line.size > 0:
    _scatterers_vis = np.hstack((frozen_dipoles_line, movable_dipoles_line))
else:
    # If there are no frozen dipoles, just use the movable ones.
    _scatterers_vis = movable_dipoles_line

base_sim_for_vis = CoupledDipoleSimulator(**_sim_params_vis)
base_sim_for_vis.scatterers = _scatterers_vis

# Initialize Optimizer (used for Blocks 5, 6, and 7 onwards)
optimizer_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for Optimizer: {optimizer_device}")
optimizer = Optimizer(
    F_operator_tensor=F_tensor_operator, device=optimizer_device
)

# Photon budget for normalization (can be defined once, used across sections)
photon_budget = 1.0

# Global optimization run settings (e.g., 10x3000 iterations)
multi_run_count = 5
single_run_iterations = 3000

# Globally store wavefronts and their J matrices
optimized_wavefronts = {}
fisher_matrices_for_wavefronts = {}

# %% 5) Single-parameter MIS wavefronts and their Fisher Information Matrices

print("\n--- Section 5: Single-Parameter MIS Wavefronts ---")

# Get single-parameter MIS wavefronts
single_mis_results = optimizer.get_single_parameter_mis(
    param_names=param_names, photon_budget=photon_budget
)

print(
    "\nCalculating and storing Fisher Information Matrix for each single-parameter MIS:"
)
for name, result in single_mis_results.items():
    wavefront = result["wavefront"]
    optimized_wavefronts[f"MIS_{name}"] = wavefront

    # Calculate the 3x3 Fisher Information Matrix for this specific wavefront
    J_for_mis = optimizer.J_fisher_info_matrix(wavefront)
    fisher_matrices_for_wavefronts[f"MIS_{name}"] = J_for_mis

    # Visualize the intensity pattern for each single-parameter MIS
    print(f"  Visualizing intensity for MIS_{name}...")
    base_sim_for_vis.reset()  # Reset simulator state
    base_sim_for_vis.set_input_field_kx(
        kx_array, wavefront
    )  # kx_array should be from previous cells
    base_sim_for_vis.solve_fields()
    base_sim_for_vis.visualize_intensity(
        title=f"MIS for {name} (Max Eigenvalue F_{name}{name} = {result['eigenvalue']:.2e})",
        dpi=100,
    )

print("\nSingle-parameter MIS analysis complete.")

# %% 6) Combined wavefront => sum of diagonal blocks (Trace optimization via eigendecomposition)

print(
    "\n--- Section 6: Combined Wavefront (Trace J via Eigendecomposition) ---"
)

# F_sum_op is the operator sum(F_mumu) whose largest eigenvector maximizes Tr(J)
# when J is calculated with that eigenvector.
F_sum_op = np.zeros_like(
    F_tensor_operator[0, 0, :, :]
)  # Initialize as (N_in, N_in)
for i in range(optimizer.nparams):
    F_sum_op += F_tensor_operator[i, i, :, :]

# Find the eigenvector of F_sum_op corresponding to its largest eigenvalue
# This eigenvector is the wavefront that maximizes trace(J) among this class of solutions
trace_opt_eigval, trace_opt_wavefront_eig = optimizer.eig_optimize_matrix(
    F_sum_op, maximize=True, num=1
)
trace_opt_wavefront_eig = Optimizer.normalize_wavefront(
    trace_opt_wavefront_eig, photon_budget
)

optimized_wavefronts["TraceJ_eig"] = trace_opt_wavefront_eig
J_for_trace_opt_eig = optimizer.J_fisher_info_matrix(trace_opt_wavefront_eig)
fisher_matrices_for_wavefronts["TraceJ_eig"] = J_for_trace_opt_eig


print(f"Visualizing intensity for TraceJ_eig...")
base_sim_for_vis.reset()
base_sim_for_vis.set_input_field_kx(kx_array, trace_opt_wavefront_eig)
base_sim_for_vis.solve_fields()
base_sim_for_vis.visualize_intensity(
    title=f"Trace J Optimized Wavefront (Eigendecomposition of sum F_mumu)\nMax Eigenvalue = {trace_opt_eigval:.2e}",
    dpi=100,
)

# %% 7a) D-optimality: Maximize Determinant of J (via torch)

print("\n--- Section 7a: D-Optimality (Maximize det(J)) ---")

# Import the loss function
from scatteringlibs.lossfunctions import determinant_of_J

# Perform the optimization using the multi-run method
# We want to maximize the determinant of J.
det_J_optimization_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=determinant_of_J,
    runs=multi_run_count,
    steps=single_run_iterations,
    lr=1e-3,
    verbose=True,
    maximize=True,
    photon_budget=photon_budget,
    base_seed=42,
)

# Store the best wavefront and its Fisher matrix
wavefront_detJ_opt = det_J_optimization_results["bestvec"]
best_detJ_value = det_J_optimization_results["bestval"]

if wavefront_detJ_opt is not None:
    optimized_wavefronts["DetJ_torch"] = wavefront_detJ_opt
    J_for_detJ_opt = optimizer.J_fisher_info_matrix(wavefront_detJ_opt)
    fisher_matrices_for_wavefronts["DetJ_torch"] = J_for_detJ_opt
    print(f"Optimization complete. Best det(J) = {best_detJ_value:.3e}")

    # Let's plot all loss histories from the multirun
    plt.figure(figsize=(8, 5))
    for i, history in enumerate(
        det_J_optimization_results["all_loss_histories"]
    ):
        plt.plot(history, label=f"Run {i+1}")
    plt.title(
        f"D-Optimality: det(J) Maximization (All {multi_run_count} Runs)"
    )
    plt.xlabel("Iteration")
    plt.ylabel("det(J)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

    # Visualize the intensity pattern
    print("Visualizing intensity for DetJ_torch...")
    base_sim_for_vis.reset()
    base_sim_for_vis.set_input_field_kx(kx_array, wavefront_detJ_opt)
    base_sim_for_vis.solve_fields()
    base_sim_for_vis.visualize_intensity(
        title=f"D-Optimized Wavefront (Max det(J) = {best_detJ_value:.2e})",
        dpi=100,
    )
else:
    print("D-optimality optimization did not yield a valid wavefront.")

print("\nSection 7a (D-Optimality) complete.")

# %% 7b) E-optimality: Maximize smallest eigenvalue of J (Minimize spectral radius of J^-1)

print("\n--- Section 7b: E-Optimality (Maximize min eigenvalue of J) ---")

# Loss function already imported at the top
# from scatteringlibs.lossfunctions import maximize_min_eig_J

# Perform the optimization
# We want to maximize the smallest eigenvalue of J.
min_eig_J_optimization_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=maximize_min_eig_J,
    runs=multi_run_count,
    steps=single_run_iterations,
    lr=1e-3,
    verbose=True,
    maximize=True,
    photon_budget=photon_budget,
    base_seed=42,
)

# Store the best wavefront and its Fisher matrix
wavefront_min_eig_J_opt = min_eig_J_optimization_results["bestvec"]
best_min_eig_J_value = min_eig_J_optimization_results["bestval"]

if wavefront_min_eig_J_opt is not None:
    optimized_wavefronts["SpecJ_torch"] = (
        wavefront_min_eig_J_opt  # Key used in original script for spectral opt
    )
    J_for_min_eig_J_opt = optimizer.J_fisher_info_matrix(
        wavefront_min_eig_J_opt
    )
    fisher_matrices_for_wavefronts["SpecJ_torch"] = J_for_min_eig_J_opt
    print(
        f"Optimization complete. Best min eigenvalue(J) = {best_min_eig_J_value:.3e}"
    )

    # Plot all loss histories from the multirun
    plt.figure(figsize=(8, 5))
    for i, history in enumerate(
        min_eig_J_optimization_results["all_loss_histories"]
    ):
        plt.plot(history, label=f"Run {i+1}")
    plt.title(
        f"E-Optimality: min eigenvalue(J) Maximization (All {multi_run_count} Runs)"
    )
    plt.xlabel("Iteration")
    plt.ylabel("min eigenvalue(J)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

    # Visualize the intensity pattern
    print("Visualizing intensity for SpecJ_torch (max min_eig(J))...")
    base_sim_for_vis.reset()
    base_sim_for_vis.set_input_field_kx(kx_array, wavefront_min_eig_J_opt)
    base_sim_for_vis.solve_fields()
    base_sim_for_vis.visualize_intensity(
        title=f"E-Optimized Wavefront (Max min_eig(J) = {best_min_eig_J_value:.2e})",
        dpi=100,
    )
else:
    print(
        "E-optimality (maximize min_eig(J)) optimization did not yield a valid wavefront."
    )

print("\nSection 7b (E-Optimality) complete.")

# %% 7c) A-optimality: Minimize Trace of J^-1 (sum of variances)

print("\n--- Section 7c: A-Optimality (Minimize Tr(J^-1)) ---")

# Loss function already imported at the top
# from scatteringlibs.lossfunctions import minimize_trace_of_J_inv

# Perform the optimization
# We want to minimize the trace of J_inverse.
trace_J_inv_optimization_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=minimize_trace_of_J_inv,
    runs=multi_run_count,
    steps=single_run_iterations,
    lr=1e-3,
    verbose=True,
    maximize=False,
    photon_budget=photon_budget,
    base_seed=42,
)

# Store the best wavefront and its Fisher matrix
wavefront_trace_J_inv_opt = trace_J_inv_optimization_results["bestvec"]
best_trace_J_inv_value = trace_J_inv_optimization_results["bestval"]

if wavefront_trace_J_inv_opt is not None:
    optimized_wavefronts["TraceJinv_torch"] = wavefront_trace_J_inv_opt
    J_for_trace_J_inv_opt = optimizer.J_fisher_info_matrix(
        wavefront_trace_J_inv_opt
    )
    fisher_matrices_for_wavefronts["TraceJinv_torch"] = J_for_trace_J_inv_opt
    print(
        f"Optimization complete. Best Tr(J^-1) = {best_trace_J_inv_value:.3e}"
    )

    # Plot all loss histories from the multirun
    plt.figure(figsize=(8, 5))
    for i, history in enumerate(
        trace_J_inv_optimization_results["all_loss_histories"]
    ):
        plt.plot(history, label=f"Run {i+1}")
    plt.title(
        f"A-Optimality: Tr(J^-1) Minimization (All {multi_run_count} Runs)"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Tr(J^-1)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

    # Visualize the intensity pattern
    print("Visualizing intensity for TraceJinv_torch (min Tr(J^-1))...")
    base_sim_for_vis.reset()
    base_sim_for_vis.set_input_field_kx(kx_array, wavefront_trace_J_inv_opt)
    base_sim_for_vis.solve_fields()
    base_sim_for_vis.visualize_intensity(
        title=f"A-Optimized Wavefront (Min Tr(J^-1) = {best_trace_J_inv_value:.2e})",
        dpi=100,
    )
else:
    print(
        "A-optimality (minimize Tr(J^-1)) optimization did not yield a valid wavefront."
    )

print("\nSection 7c (A-Optimality) complete.")


# %% 8) Visualize Comparison of Fisher Information Matrix components for all optimized wavefronts
# no need to refactor here because this block is just for inspection
def visualize_multi_param_fisher_comparison(
    wavefronts_dict,
    optimizer_instance,
    param_names_global,
    figure_title="Comparison of Wavefronts",
):
    """
    Visualizes and compares Fisher Information Matrix components (diagonal, trace, determinant)
    for multiple wavefronts.

    Args:
        wavefronts_dict (dict): Dictionary where keys are wavefront names (str) and
                                values are the wavefront vectors (np.ndarray or torch.Tensor).
        optimizer_instance (Optimizer): An instance of the Optimizer class, already configured.
        param_names_global (list): List of parameter names, e.g., ["shiftX", "shiftZ", "rotation"].
        figure_title (str): Overall title for the figure.
    """
    results = {}
    wave_names = list(wavefronts_dict.keys())

    if not wave_names:
        print("No wavefronts provided to visualize.")
        return

    for wname in wave_names:
        wf_cplx = wavefronts_dict[wname]
        if isinstance(
            wf_cplx, torch.Tensor
        ):  # Ensure it's a NumPy array for J calculation
            wf_cplx = wf_cplx.cpu().numpy()

        J_mat = optimizer_instance.J_fisher_info_matrix(wf_cplx)
        results[wname] = {
            "diag": [J_mat[0, 0], J_mat[1, 1], J_mat[2, 2]],
            "trace": np.trace(J_mat),
            "det": np.linalg.det(J_mat),
        }

    diag_data = np.array([results[w]["diag"] for w in wave_names])
    trace_data = np.array([results[w]["trace"] for w in wave_names])
    det_data = np.array([results[w]["det"] for w in wave_names])

    # Prepare plot labels for parameters: J_xx, J_zz, J_θθ
    plot_param_labels = []
    for p_name in param_names_global:
        if "shift" in p_name.lower():
            label = p_name.replace("shift", "").lower()
            plot_param_labels.append(f"{label}{label}")
        elif "rotation" in p_name.lower():
            plot_param_labels.append(
                r"\theta\theta"
            )  # Special case for rotation
        else:
            plot_param_labels.append(p_name.lower() * 2)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6)
    )  # Adjusted figsize slightly

    # First subplot: Diagonal elements
    x_idx = np.arange(len(wave_names))
    width = 0.25

    for i in range(len(param_names_global)):  # Should be 3 for this script
        offset = (i - 1.0) * width  # Centering the group of bars
        axes[0].bar(
            x_idx + offset,
            diag_data[:, i],
            width=width,
            label=f"$J_{{{plot_param_labels[i]}}}$",
        )

    axes[0].set_title("Diagonal Elements of Fisher Matrix (J)", fontsize=12)
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(wave_names, rotation=45, ha="right")
    axes[0].legend(title="Parameters")
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].set_ylabel("Value")

    # Second subplot: Trace and Determinant
    ax_tr = axes[1]
    ax_det = ax_tr.twinx()  # Share the same x-axis

    bar_width_tr_det = 0.35  # Width for these bars

    # Plot Trace
    tr_color = "skyblue"
    ax_tr.bar(
        x_idx - bar_width_tr_det / 2,  # Offset to the left
        trace_data,
        width=bar_width_tr_det,
        color=tr_color,
        label="Tr(J)",
        alpha=0.8,
    )
    ax_tr.set_ylabel("Trace(J)", color=tr_color, fontsize=12)
    ax_tr.tick_params(axis="y", labelcolor=tr_color)

    # Plot Determinant
    det_color = "lightcoral"
    ax_det.bar(
        x_idx + bar_width_tr_det / 2,  # Offset to the right
        det_data,
        width=bar_width_tr_det,
        color=det_color,
        label="det(J)",
        alpha=0.8,
    )
    ax_det.set_ylabel("det(J)", color=det_color, fontsize=12)
    ax_det.tick_params(axis="y", labelcolor=det_color)
    if np.all(det_data >= 0):
        ax_det.set_yscale("log")
    else:
        print(
            "Warning: Some determinant values are negative, using linear scale for det(J)."
        )

    ax_tr.set_title("Trace and Determinant of Fisher Matrix (J)", fontsize=12)
    ax_tr.set_xticks(x_idx)
    ax_tr.set_xticklabels(wave_names, rotation=45, ha="right")
    ax_tr.grid(True, linestyle=":", alpha=0.5)

    handles_tr, labels_tr = ax_tr.get_legend_handles_labels()
    handles_det, labels_det = ax_det.get_legend_handles_labels()
    ax_tr.legend(
        handles_tr + handles_det,
        labels_tr + labels_det,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
    )

    plt.suptitle(figure_title, fontsize=16)
    plt.tight_layout(
        rect=[0, 0.05, 1, 0.96]
    )  # Adjust rect to make space for suptitle and legend
    plt.show()


# Run and visualize comparison
visualize_multi_param_fisher_comparison(
    optimized_wavefronts,
    optimizer,  # The optimizer instance, should be already defined
    param_names,  # The global param_names list, should be already defined
    figure_title="Comparison of All Optimized Wavefronts",
)

print("\nSection 8 (Wavefront Comparison Visualization) complete.")

# %% 9) Compare CRLB, with optional plane wave or random wavefront comparison

print("\n--- Section 9: CRLB Comparison ---")

crlbanalysis.visualize_crlb_comparison(
    optimized_wavefronts=optimized_wavefronts,
    F_operator_tensor=optimizer.F_op,
    photon_budget=photon_budget,
    param_names=param_names,
    sim_kwargs=sim_kwargs,
    angles_deg=angles_deg,
    kx_array=kx_array,
    figure_title="CRLB Comparison for different types of Maximum Information States",
    show_plane_waves=True,
    show_random_wavefronts=True,
    num_random_samples=1000,
)

print("\nSection 9 (CRLB Comparison) complete.")

# %% 10) Fisher and Covariance Matrix Visualization Data & condition numbers
print("\n--- Section 10: Fisher and Covariance Matrix Visualization ---")

Js_for_visualization = {}
J_invs_for_visualization = {}
cond_nums_for_visualization = {}

for wf_name, wf_cplx_tensor in optimized_wavefronts.items():
    # Ensure wavefront is a NumPy array for J calculation
    if isinstance(wf_cplx_tensor, torch.Tensor):
        wf_cplx = wf_cplx_tensor.cpu().numpy()
    else:
        wf_cplx = wf_cplx_tensor

    try:
        J_mat = optimizer.J_fisher_info_matrix(wf_cplx)

        # Handle potential non-invertible J_mat before attempting inversion
        # A very small number compared to 1/cond indicates singularity for practical purposes
        if np.linalg.cond(J_mat) < 1 / np.finfo(J_mat.dtype).eps:
            inv_mat = np.linalg.inv(J_mat)
            # Re-calculate cond_num after successful inversion if desired, or use the one from check
            cond_num = np.linalg.cond(J_mat)
        else:  # Matrix is singular or ill-conditioned
            print(
                f"Warning: Fisher matrix for '{wf_name}' is singular or ill-conditioned. Condition number: {np.linalg.cond(J_mat)}"
            )
            inv_mat = np.full_like(J_mat, np.nan)  # Store NaN matrix
            cond_num = np.inf  # Store infinity for condition number

    except np.linalg.LinAlgError as e:
        print(
            f"Error calculating inverse or condition number for '{wf_name}': {e}"
        )
        # Attempt to get J_mat again if it wasn't calculated before the error, or if it's needed.
        # This line assumes J_fisher_info_matrix itself doesn't raise LinAlgError for non-invertible matrices,
        # which is typical as J can be calculated even if singular.
        if "J_mat" not in locals() or J_mat is None:  # Ensure J_mat exists
            J_mat = optimizer.J_fisher_info_matrix(
                wf_cplx
            )  # Recalculate if necessary

        inv_mat = np.full_like(
            J_mat, np.nan
        )  # Store NaN matrix if inversion fails
        cond_num = (
            np.inf
        )  # Store infinity or a very large number for condition number

    except (
        Exception
    ) as e:  # Catch any other unexpected errors during J calculation or processing
        print(
            f"An unexpected error occurred for '{wf_name}' during Fisher matrix processing: {e}"
        )
        # Define nparams based on typical optimizer structure if available, otherwise use a default like 3
        num_params = optimizer.nparams if hasattr(optimizer, "nparams") else 3
        J_mat = np.full((num_params, num_params), np.nan)  # Store NaN matrix
        inv_mat = np.full((num_params, num_params), np.nan)
        cond_num = np.nan

    Js_for_visualization[wf_name] = J_mat
    J_invs_for_visualization[wf_name] = inv_mat
    cond_nums_for_visualization[wf_name] = cond_num

print(
    "\nCalculated J, J^-1, and condition numbers for all wavefronts for visualization."
)
print("Section 10 (Fisher and Covariance Matrix Visualization Data) complete.")

# Call the visualization function (now imported from crlbanalysis)
plot_fisher_comparison(
    Js_map=Js_for_visualization,
    J_invs_map=J_invs_for_visualization,
    cond_nums_map=cond_nums_for_visualization,
    param_names_global=param_names,  # param_names should be globally defined earlier in the script
    figure_title="Comparison of J, $J^{-1}$, and Condition Numbers for Optimized Wavefronts",
)

print("\nSection 10b (J, J^-1, Condition Number Visualization) complete.")

# %% 11) Multi-objective optimization considering nuisance parameters
print("\n--- Section 11: Nuisance Parameter Optimization Strategies ---")

from scatteringlibs.lossfunctions import loglin_loss, offdiag_loss, subdet_loss

block11_loss_histories = {}
nuisance_param_index = 2  # Corresponds to rotation

strategies = {
    "LogLin": {
        "loss_fn_constructor": lambda: (
            lambda J_mat: loglin_loss(
                J_mat, nuisance_param_index=nuisance_param_index, alpha=0.1
            )
        ),
        "params": {"alpha": 0.1},
    },
    "OffDiag": {
        "loss_fn_constructor": lambda: (
            lambda J_mat: offdiag_loss(
                J_mat, nuisance_param_index=nuisance_param_index, beta=1.0
            )
        ),
        "params": {"beta": 1.0},
    },
    "SubDet": {
        "loss_fn_constructor": lambda: (
            lambda J_mat: subdet_loss(
                J_mat, nuisance_param_index=nuisance_param_index
            )
        ),
        "params": {},
    },
}

for strategy_name, config in strategies.items():
    print(f"\n--- Optimizing for Strategy: {strategy_name} ---")

    loss_fn = config["loss_fn_constructor"]()

    optimization_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=loss_fn,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=False,  # All these are minimization problems
        photon_budget=photon_budget,
        base_seed=42,
    )

    block11_loss_histories[strategy_name] = optimization_results[
        "all_loss_histories"
    ]
    bestvec = optimization_results["bestvec"]
    bestval = optimization_results["bestval"]

    wavefront_key = f"{strategy_name}_torch"

    if bestvec is not None:
        optimized_wavefronts[wavefront_key] = bestvec
        J_for_strategy = optimizer.J_fisher_info_matrix(bestvec)
        fisher_matrices_for_wavefronts[wavefront_key] = J_for_strategy
        print(
            f"Optimization for {strategy_name} complete. Best loss value = {bestval:.3e}"
        )

        # Plot all loss histories from the multirun
        plt.figure(figsize=(8, 5))
        for i, history in enumerate(
            optimization_results["all_loss_histories"]
        ):
            plt.plot(history, label=f"Run {i+1}")
        plt.title(f"{strategy_name} Optimization (All {multi_run_count} Runs)")
        plt.xlabel("Iteration")
        plt.ylabel(f"{strategy_name} Loss")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.show()

        # Visualize the intensity pattern
        print(f"Visualizing intensity for {wavefront_key}...")
        base_sim_for_vis.reset()
        base_sim_for_vis.set_input_field_kx(kx_array, bestvec)
        base_sim_for_vis.solve_fields()
        base_sim_for_vis.visualize_intensity(
            title=f"{strategy_name}-Optimized Wavefront (Loss = {bestval:.2e})",
            dpi=100,
        )
    else:
        print(f"{strategy_name} optimization did not yield a valid wavefront.")

print("\nSection 11 (Nuisance Parameter Optimization) complete.")

# %% 12) skipped

# %% 13) Marginal FIM Optimality (D, A, E)
# Optimization based on the 2x2 marginal Fisher Information Matrix,
# treating one parameter as a nuisance parameter.

# Define the nuisance parameter index (e.g., 2 for rotation)
# This index is used by the loss functions to compute the marginal FIM.
nuisance_param_index_for_marginal = 2 # Rotation
lr_marginal = 1e-3  # Learning rate for these optimizations
base_seed_marginal = 142 # Different seed for this block

print(f"\n--- Starting Marginal FIM Optimizations (Nuisance Param Index: {nuisance_param_index_for_marginal}) ---")

# %% 13a) D-Optimality for Marginal FIM (Maximize det(J_marginal))
print("\n--- Section 13a: D-Optimality (Maximize det(J_marginal)) ---")

loss_fn_det_marginal = lambda J_full_torch: determinant_of_marginal_J(
    optimizer._compute_marginal_fisher_info_matrix(J_full_torch, nuisance_param_index_for_marginal)
)

det_J_marginal_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=loss_fn_det_marginal,
    runs=multi_run_count,
    steps=single_run_iterations,
    lr=lr_marginal,
    verbose=True,
    maximize=True,
    photon_budget=photon_budget,
    base_seed=base_seed_marginal
)

wavefront_detJ_marginal_opt = det_J_marginal_results["bestvec"]
best_detJ_marginal_value = det_J_marginal_results["bestval"]
opt_key_13a = f"DetJ_marginal_n{nuisance_param_index_for_marginal}_torch"

if wavefront_detJ_marginal_opt is not None:
    optimized_wavefronts[opt_key_13a] = wavefront_detJ_marginal_opt
    J_full_for_detJ_marginal = optimizer.J_fisher_info_matrix(wavefront_detJ_marginal_opt)
    fisher_matrices_for_wavefronts[opt_key_13a] = J_full_for_detJ_marginal
    # Also store the marginal J itself for potential direct comparison later
    # marginal_J_det_opt = optimizer._compute_marginal_fisher_info_matrix(torch.from_numpy(J_full_for_detJ_marginal).to(optimizer.device), nuisance_param_index_for_marginal).cpu().numpy()
    # You might want to store this in a new dictionary if needed: marginal_fisher_matrices_for_wavefronts
    print(f"Optimization complete. Best det(J_marginal) = {best_detJ_marginal_value:.3e}")

    plt.figure(figsize=(8, 5))
    for i, history in enumerate(det_J_marginal_results["all_loss_histories"]):
        plt.plot(history, label=f"Run {i+1}")
    plt.title(f"D-Optimality for Marginal FIM (det(J_marginal)) (All {multi_run_count} Runs)")
    plt.xlabel("Iteration")
    plt.ylabel("det(J_marginal)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

    base_sim_for_vis.reset()
    base_sim_for_vis.set_input_field_kx(kx_array, wavefront_detJ_marginal_opt)
    base_sim_for_vis.solve_fields()
    base_sim_for_vis.visualize_intensity(
        title=f"D-Optimized Wavefront for Marginal FIM\nMax det(J_marginal) = {best_detJ_marginal_value:.2e}",
        dpi=100,
    )
else:
    print(f"D-optimality for marginal FIM (det(J_marginal)) did not yield a valid wavefront for {opt_key_13a}.")

print("\nSection 13a (D-Optimality for Marginal FIM) complete.")


# %% 13b) A-Optimality for Marginal FIM (Minimize Tr(J_marginal^-1))
print("\n--- Section 13b: A-Optimality (Minimize Tr(J_marginal^-1)) ---")

loss_fn_trace_inv_marginal = lambda J_full_torch: minimize_trace_of_marginal_J_inv(
    optimizer._compute_marginal_fisher_info_matrix(J_full_torch, nuisance_param_index_for_marginal)
)

trace_J_inv_marginal_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=loss_fn_trace_inv_marginal,
    runs=multi_run_count,
    steps=single_run_iterations,
    lr=lr_marginal,
    verbose=True,
    maximize=False, # Minimization problem
    photon_budget=photon_budget,
    base_seed=base_seed_marginal + 1 # Vary seed slightly
)

wavefront_traceJinv_marginal_opt = trace_J_inv_marginal_results["bestvec"]
best_traceJinv_marginal_value = trace_J_inv_marginal_results["bestval"]
opt_key_13b = f"TraceJinv_marginal_n{nuisance_param_index_for_marginal}_torch"

if wavefront_traceJinv_marginal_opt is not None:
    optimized_wavefronts[opt_key_13b] = wavefront_traceJinv_marginal_opt
    J_full_for_traceJinv_marginal = optimizer.J_fisher_info_matrix(wavefront_traceJinv_marginal_opt)
    fisher_matrices_for_wavefronts[opt_key_13b] = J_full_for_traceJinv_marginal
    print(f"Optimization complete. Best Tr(J_marginal^-1) = {best_traceJinv_marginal_value:.3e}")

    plt.figure(figsize=(8, 5))
    for i, history in enumerate(trace_J_inv_marginal_results["all_loss_histories"]):
        plt.plot(history, label=f"Run {i+1}")
    plt.title(f"A-Optimality for Marginal FIM (Tr(J_marginal^-1)) (All {multi_run_count} Runs)")
    plt.xlabel("Iteration")
    plt.ylabel("Tr(J_marginal^-1)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

    base_sim_for_vis.reset()
    base_sim_for_vis.set_input_field_kx(kx_array, wavefront_traceJinv_marginal_opt)
    base_sim_for_vis.solve_fields()
    base_sim_for_vis.visualize_intensity(
        title=f"A-Optimized Wavefront for Marginal FIM\nMin Tr(J_marginal^-1) = {best_traceJinv_marginal_value:.2e}",
        dpi=100,
    )
else:
    print(f"A-optimality for marginal FIM (Tr(J_marginal^-1)) did not yield a valid wavefront for {opt_key_13b}.")

print("\nSection 13b (A-Optimality for Marginal FIM) complete.")


# %% 13c) E-Optimality for Marginal FIM (Maximize min_eig(J_marginal))
print("\n--- Section 13c: E-Optimality (Maximize min_eig(J_marginal)) ---")

loss_fn_min_eig_marginal = lambda J_full_torch: maximize_min_eig_marginal_J(
    optimizer._compute_marginal_fisher_info_matrix(J_full_torch, nuisance_param_index_for_marginal)
)

min_eig_J_marginal_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=loss_fn_min_eig_marginal,
    runs=multi_run_count,
    steps=single_run_iterations,
    lr=lr_marginal,
    verbose=True,
    maximize=True,
    photon_budget=photon_budget,
    base_seed=base_seed_marginal + 2 # Vary seed slightly
)

wavefront_mineigJ_marginal_opt = min_eig_J_marginal_results["bestvec"]
best_mineigJ_marginal_value = min_eig_J_marginal_results["bestval"]
opt_key_13c = f"SpecJ_marginal_n{nuisance_param_index_for_marginal}_torch"

if wavefront_mineigJ_marginal_opt is not None:
    optimized_wavefronts[opt_key_13c] = wavefront_mineigJ_marginal_opt
    J_full_for_mineigJ_marginal = optimizer.J_fisher_info_matrix(wavefront_mineigJ_marginal_opt)
    fisher_matrices_for_wavefronts[opt_key_13c] = J_full_for_mineigJ_marginal
    print(f"Optimization complete. Best min_eig(J_marginal) = {best_mineigJ_marginal_value:.3e}")

    plt.figure(figsize=(8, 5))
    for i, history in enumerate(min_eig_J_marginal_results["all_loss_histories"]):
        plt.plot(history, label=f"Run {i+1}")
    plt.title(f"E-Optimality for Marginal FIM (min_eig(J_marginal)) (All {multi_run_count} Runs)")
    plt.xlabel("Iteration")
    plt.ylabel("min_eig(J_marginal)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

    base_sim_for_vis.reset()
    base_sim_for_vis.set_input_field_kx(kx_array, wavefront_mineigJ_marginal_opt)
    base_sim_for_vis.solve_fields()
    base_sim_for_vis.visualize_intensity(
        title=f"E-Optimized Wavefront for Marginal FIM\nMax min_eig(J_marginal) = {best_mineigJ_marginal_value:.2e}",
        dpi=100,
    )
else:
    print(f"E-optimality for marginal FIM (min_eig(J_marginal)) did not yield a valid wavefront for {opt_key_13c}.")

print("\nSection 13c (E-Optimality for Marginal FIM) complete.")

print("\nAll Marginal FIM Optimizations (Sections 13a, 13b, 13c) complete.")

# %% Re-run Visualizations if they should include these new results
# It's good practice to re-run summary visualizations if new data points are added.
# For example, Section 8, 9, 10 might be relevant.
# If these are time-consuming or generate too many plots for an intermediate step,
# they can be left to the end of the script or run selectively.

# Example: Re-running Section 8 after adding new wavefronts from Section 13
print("\n--- Re-running Section 8 (Wavefront Comparison Visualization with Marginal Opts) ---")
visualize_multi_param_fisher_comparison(
    optimized_wavefronts,
    optimizer,
    param_names,
    figure_title="Comparison of All Optimized Wavefronts (incl. Marginal)",
)
print("\nRe-run of Section 8 complete.")

# Example: Re-running Section 10b after adding new wavefronts from Section 13
print("\n--- Re-running Section 10b (J, J^-1, Condition Number Visualization with Marginal Opts) ---")
# First, re-calculate J, J_inv, cond_nums for ALL wavefronts in optimized_wavefronts
# This is important because Section 10's data (Js_for_visualization, etc.) was calculated
# before Section 13 results were added.

Js_for_visualization_updated = {}
J_invs_for_visualization_updated = {}
cond_nums_for_visualization_updated = {}

for wf_name, wf_cplx_tensor in optimized_wavefronts.items():
    if isinstance(wf_cplx_tensor, torch.Tensor):
        wf_cplx = wf_cplx_tensor.cpu().numpy()
    else:
        wf_cplx = wf_cplx_tensor
    try:
        J_mat = optimizer.J_fisher_info_matrix(wf_cplx)
        if np.linalg.cond(J_mat) < 1 / np.finfo(J_mat.dtype).eps:
            inv_mat = np.linalg.inv(J_mat)
            cond_num = np.linalg.cond(J_mat)
        else:
            print(f"Warning (Re-run): Fisher matrix for '{wf_name}' is singular. Cond: {np.linalg.cond(J_mat)}")
            inv_mat = np.full_like(J_mat, np.nan)
            cond_num = np.inf
    except np.linalg.LinAlgError as e:
        print(f"Error (Re-run) for '{wf_name}': {e}")
        J_mat = optimizer.J_fisher_info_matrix(wf_cplx) # ensure J_mat is available
        inv_mat = np.full_like(J_mat, np.nan)
        cond_num = np.inf
    except Exception as e:
        print(f"Unexpected error (Re-run) for '{wf_name}': {e}")
        num_params = optimizer.nparams if hasattr(optimizer, "nparams") else 3
        J_mat = np.full((num_params, num_params), np.nan)
        inv_mat = np.full((num_params, num_params), np.nan)
        cond_num = np.nan

    Js_for_visualization_updated[wf_name] = J_mat
    J_invs_for_visualization_updated[wf_name] = inv_mat
    cond_nums_for_visualization_updated[wf_name] = cond_num

plot_fisher_comparison(
    Js_map=Js_for_visualization_updated,
    J_invs_map=J_invs_for_visualization_updated,
    cond_nums_map=cond_nums_for_visualization_updated,
    param_names_global=param_names,
    figure_title="Comparison of J, $J^{-1}$, Cond Nums (incl. Marginal)",
)
print("\nRe-run of Section 10b complete.")


# %% END OF SCRIPT (or further analysis)
# Any final analysis or summary plots would go here.
