# playground/calc_fisher_info_and_MIS_3params_refactored.py

# %% Imports and functions
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs import placedipoles  # Import the whole module
from scatteringlibs import crlbanalysis
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

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

# Import the new loss function
from scatteringlibs.lossfunctions import maximize_min_eig_J

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

# Import the new loss function
from scatteringlibs.lossfunctions import minimize_trace_of_J_inv

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

# %% 10) Save results (optional)
# Storable.save_results(
#     data_to_save={
#         "optimized_wavefronts": optimized_wavefronts,
#         "fisher_matrices_for_wavefronts": fisher_matrices_for_wavefronts,
#         # Add other relevant data like loss histories if needed
#     },
#     filename_prefix="3params_optimization_results",
#     save_dir="data" # Make sure this directory exists or is created
# )
# print("\nSection 10 (Save Results) complete.")


# %% 11) Multi-objective optimization considering nuisance parameters
print("\n--- Section 11: Nuisance Parameter Optimization Strategies ---")

from scatteringlibs.lossfunctions import loglin_loss, offdiag_loss, subdet_loss

block11_loss_histories = {}
nuisance_param_index = 2 # Corresponds to rotation

strategies = {
    "LogLin": {
        "loss_fn_constructor": lambda: (lambda J_mat: loglin_loss(J_mat, nuisance_param_index=nuisance_param_index, alpha=0.1)),
        "params": {"alpha": 0.1}
    },
    "OffDiag": {
        "loss_fn_constructor": lambda: (lambda J_mat: offdiag_loss(J_mat, nuisance_param_index=nuisance_param_index, beta=1.0)),
        "params": {"beta": 1.0}
    },
    "SubDet": {
        "loss_fn_constructor": lambda: (lambda J_mat: subdet_loss(J_mat, nuisance_param_index=nuisance_param_index)),
        "params": {}
    }
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
        maximize=False, # All these are minimization problems
        photon_budget=photon_budget,
        base_seed=42
    )

    block11_loss_histories[strategy_name] = optimization_results["all_loss_histories"]
    bestvec = optimization_results["bestvec"]
    bestval = optimization_results["bestval"]

    wavefront_key = f"{strategy_name}_torch"

    if bestvec is not None:
        optimized_wavefronts[wavefront_key] = bestvec
        J_for_strategy = optimizer.J_fisher_info_matrix(bestvec)
        fisher_matrices_for_wavefronts[wavefront_key] = J_for_strategy
        print(f"Optimization for {strategy_name} complete. Best loss value = {bestval:.3e}")

        # Plot all loss histories from the multirun
        plt.figure(figsize=(8, 5))
        for i, history in enumerate(optimization_results["all_loss_histories"]):
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
# Note: You might want to update the comparison plots in sections 8 and 9
# to include these new wavefronts if desired, by re-running those cells
# or by appending to the optimized_wavefronts dict before they run.
# For now, this script adds them, and they would be included if sections 8/9 are run after this.


# %% 11.1) Plotting "OffDiag" Multi-Run Loss Curves
if "OffDiag" in block11_loss_histories:
    plt.figure(figsize=(10, 6))
    offdiag_losses = block11_loss_histories["OffDiag"]
    num_runs = len(offdiag_losses)
    colors = plt.cm.winter(np.linspace(0, 1, num_runs))
    for i, history in enumerate(offdiag_losses):
        plt.plot(history, label=f"Run {i+1}", color=colors[i])
    plt.title("OffDiag Strategy: All Loss Curves (Multi-Run)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (OffDiag)")
    plt.yscale("symlog", linthresh=1e-3, linscale=1) # Match original script's scaling if appropriate
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()
else:
    print("OffDiag loss histories not found in block11_loss_histories.")

# %% 11.2) Plotting Normalized Loss Curves for All Strategies
plt.figure(figsize=(10, 6))
strategies_to_plot = ["LogLin", "OffDiag", "SubDet"]
for strategy_name in strategies_to_plot:
    if strategy_name in block11_loss_histories:
        # Assuming each history is a list/array of loss values for one run.
        # We'll use the history of the best run if available, or the first run.
        # For simplicity, let's take the first run's history or average history.
        # The prompt implies plotting one curve per strategy from "all_loss_histories"
        # which typically stores multiple runs. Let's use the first run for simplicity.
        loss_history_runs = block11_loss_histories[strategy_name]
        if loss_history_runs: # Check if not empty
            loss_history = np.array(loss_history_runs[0]) # Taking the first run

            min_loss = np.min(loss_history)
            max_loss = np.max(loss_history)

            if max_loss == min_loss:
                normalized_loss = np.zeros_like(loss_history)
            else:
                normalized_loss = (loss_history - min_loss) / (max_loss - min_loss)

            plt.plot(normalized_loss, label=strategy_name)
        else:
            print(f"No loss histories found for strategy: {strategy_name}")

    else:
        print(f"Strategy {strategy_name} not found in block11_loss_histories.")

plt.title("Normalized Loss Curves for Nuisance Parameter Strategies (First Run)")
plt.xlabel("Iteration")
plt.ylabel("Normalized Loss (0 to 1)")
plt.legend()
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.show()

# %% 11.3) Condition Number Comparison
block11_strategy_keys = ["LogLin_torch", "OffDiag_torch", "SubDet_torch"]
plot_keys = []
condition_numbers = []

# Optionally include DetJ_torch
if "DetJ_torch" in fisher_matrices_for_wavefronts:
    plot_keys.append("DetJ_torch")

plot_keys.extend(block11_strategy_keys)

for key in plot_keys:
    if key in fisher_matrices_for_wavefronts:
        J_matrix = fisher_matrices_for_wavefronts[key]
        if isinstance(J_matrix, torch.Tensor):
            J_matrix = J_matrix.cpu().numpy()

        # Ensure matrix is finite and not all zeros before cond
        if np.all(np.isfinite(J_matrix)) and not np.all(J_matrix == 0):
            cond_num = np.linalg.cond(J_matrix)
            condition_numbers.append(cond_num)
        else:
            print(f"Warning: Fisher matrix for {key} is not suitable for condition number calculation (e.g. contains NaNs, Infs, or is zero). Assigning NaN.")
            condition_numbers.append(np.nan) # Or skip
    else:
        print(f"Fisher matrix for {key} not found. Skipping for condition number plot.")

# Filter out keys for which we couldn't get a condition number if NaNs were appended
valid_plot_keys = [key for i, key in enumerate(plot_keys) if i < len(condition_numbers) and not np.isnan(condition_numbers[i])]
valid_condition_numbers = [cn for cn in condition_numbers if not np.isnan(cn)]


if valid_plot_keys:
    plt.figure(figsize=(10, 6))
    plt.bar(valid_plot_keys, valid_condition_numbers)
    plt.title("Condition Number of J for Different Strategies")
    plt.ylabel("Condition Number (log scale)")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log") # Condition numbers can vary wildly
    plt.grid(True, axis='y', linestyle=":")
    plt.tight_layout()
    plt.show()
else:
    print("No valid condition numbers to plot.")


# %% 12) Updated Comprehensive Visualizations (including Block 11 results)
print("\n--- Section 12: Updated Comprehensive Visualizations ---")

# 12.1) Updated Comparison of Fisher Matrix components
print("\nRegenerating Fisher components plot including Block 11 strategies...")
visualize_multi_param_fisher_comparison(
    optimized_wavefronts,
    optimizer,
    param_names,
    figure_title="Updated Comparison of All Optimized Wavefronts (incl. Block 11)"
)
plt.show() # Ensure plot is displayed

# 12.2) Updated CRLB Comparison (Ellipses, etc.)
print("\nRegenerating CRLB comparison plot including Block 11 strategies...")
crlbanalysis.visualize_crlb_comparison(
    optimized_wavefronts=optimized_wavefronts,
    F_operator_tensor=optimizer.F_op, # Assuming F_op is the correct attribute in Optimizer for F_tensor_operator
    photon_budget=photon_budget,
    param_names=param_names,
    sim_kwargs=sim_kwargs,
    angles_deg=angles_deg,
    kx_array=kx_array,
    figure_title="Updated CRLB Comparison (incl. Block 11)",
    show_plane_waves=True,
    show_random_wavefronts=True,
    num_random_samples=1000 # Match earlier call
)
plt.show() # Ensure plot is displayed

# 12.3) Pareto Frontier Plots (adapted from original script's Block 11, sub-point 7)
print("\nGenerating Pareto frontier plots for CRLBs (incl. Block 11 strategies)...")

# Define parameter pairs for CRLB plots
# param_names = ["shiftX", "shiftZ", "rotation"]
param_pairs_indices = [(0, 1), (0, 2), (1, 2)] # (X,Z), (X,Rot), (Z,Rot)
param_pair_labels = [
    (f"$\\sigma$({param_names[p1]})", f"$\\sigma$({param_names[p2]})")
    for p1, p2 in param_pairs_indices
]

num_pairs = len(param_pairs_indices)
fig_pareto, axes_pareto = plt.subplots(1, num_pairs, figsize=(5 * num_pairs, 5))
if num_pairs == 1: # Ensure axes_pareto is always an array
    axes_pareto = [axes_pareto]

# Use a consistent set of markers for wavefront types
# This map can be expanded based on keys in optimized_wavefronts
marker_map = {
    "MIS_": "o",  # MIS for individual params
    "TraceJ_eig": "s",  # Trace J via eigendecomposition
    "DetJ_torch": "D",  # D-optimality
    "SpecJ_torch": "^",  # E-optimality (min_eig_J)
    "TraceJinv_torch": "v",  # A-optimality (trace_J_inv)
    "LogLin_torch": "p",  # LogLin strategy
    "OffDiag_torch": "*",  # OffDiag strategy
    "SubDet_torch": "X",  # SubDet strategy
    "PlaneWave": ".",     # Plane waves
    "Random": ","         # Random wavefronts
}
color_map = plt.cm.get_cmap("tab10", len(optimized_wavefronts))

# Collect CRLBs for all wavefronts
crlb_data_for_pareto = {}

for i_wf, (wf_name, wf_vec) in enumerate(optimized_wavefronts.items()):
    if wf_name not in fisher_matrices_for_wavefronts:
        print(f"Fisher matrix for {wf_name} not found, trying to compute.")
        J_mat_np = optimizer.J_fisher_info_matrix(wf_vec, photon_budget=photon_budget) # ensure using correct photon budget
        if isinstance(J_mat_np, torch.Tensor):
            J_mat_np = J_mat_np.cpu().numpy()
    else:
        J_mat_np = fisher_matrices_for_wavefronts[wf_name]
        if isinstance(J_mat_np, torch.Tensor):
            J_mat_np = J_mat_np.cpu().numpy()

    try:
        # Add small identity matrix for stability before inversion, if J is singular or ill-conditioned
        J_inv = np.linalg.inv(J_mat_np + 1e-9 * np.eye(J_mat_np.shape[0]))
        # CRLBs are the diagonal elements of J_inv, scaled by 1/photon_budget if J was not already scaled
        # optimizer.J_fisher_info_matrix already handles photon budget scaling if photon_budget is passed.
        # If J_fisher_info_matrix does not scale by photon_budget, then J_inv should be scaled.
        # Assuming J from fisher_matrices_for_wavefronts is correctly scaled.
        crlbs_diag = np.diag(J_inv)
        # Variances are directly these diagonal elements. Sigmas are sqrt of these.
        sigmas = np.sqrt(crlbs_diag)
        crlb_data_for_pareto[wf_name] = sigmas
    except np.linalg.LinAlgError:
        print(f"Could not invert J for wavefront {wf_name}. Skipping in Pareto plot.")
        crlb_data_for_pareto[wf_name] = np.full(len(param_names), np.nan)

# Plotting
for ax_idx, (p_indices, p_labels) in enumerate(zip(param_pairs_indices, param_pair_labels)):
    ax = axes_pareto[ax_idx]
    for i_wf, (wf_name, sigmas) in enumerate(crlb_data_for_pareto.items()):
        if not np.any(np.isnan(sigmas)):
            # Determine marker (can be more sophisticated)
            marker_style = "o" # default
            for prefix, m_style in marker_map.items():
                if wf_name.startswith(prefix):
                    marker_style = m_style
                    break

            ax.loglog(
                sigmas[p_indices[0]],
                sigmas[p_indices[1]],
                marker=marker_style,
                linestyle="None", # No lines connecting points for a scatter plot
                label=wf_name if ax_idx == 0 else None,  # Label only on the first subplot
                color=color_map(i_wf % color_map.N) # Cycle through colors
            )

    ax.set_xlabel(p_labels[0])
    ax.set_ylabel(p_labels[1])
    ax.grid(True, which="both", linestyle=":")
    ax.set_aspect('equal', adjustable='box') # Make axes equal for better visual comparison

fig_pareto.suptitle("Pareto Frontiers: CRLB Trade-offs (Updated incl. Block 11)", fontsize=16)
# Place legend outside, adjust layout
handles, labels = [], []
for ax in axes_pareto:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# Create a single legend for the whole figure if labels were only added to the first plot
if num_pairs > 0 and any(axes_pareto[0].get_legend_handles_labels()[1]):
    fig_pareto.legend(axes_pareto[0].get_legend_handles_labels()[0], axes_pareto[0].get_legend_handles_labels()[1], loc='center right', bbox_to_anchor=(1.15, 0.5))

plt.tight_layout(rect=[0, 0, 0.85, 0.95]) # Adjust rect for suptitle and legend if outside
plt.show()

print("\nSection 12 (Updated Visualizations) complete.")
