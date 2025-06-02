# playground/calc_fisher_info_and_MIS_3params_refactored.py

# %% Imports and functions
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs import placedipoles  # Import the whole module
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
