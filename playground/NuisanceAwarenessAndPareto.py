"""
Study the effect of different wavefront shaping objectives using
pareto frontier visualization.
Specficially, do we outperform general objectives using
nuisance awareness?

05.06.2025, j.seifert@uu.nl
"""

# %% Imports and functions

import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

# %% 0) settings
verbose_plotting = True
max_angle = 80  # degrees, max angle for input modes
param_names = ["shiftX", "shiftZ", "rotation"]
delta_param = 1e-3  # delta for building dS

# %% 1) define dipole configurations
movable_dipoles_line = placedipoles.place_line_of_dipoles(
    center=(5.0, 0.0), number=7
)

frozen_dipoles_line = placedipoles.place_line_of_dipoles(
    center=(3.0, 0.0), number=7
)

# %% 2) simulator geometry and input
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
angles_deg = np.linspace(-max_angle, max_angle, 100)  # Input angle channels

k0 = 2 * np.pi / sim_kwargs["lambda_0"]
kx_array = k0 * np.sin(np.deg2rad(angles_deg))

# %% 3) Build partial derivatives for each param => dS[mu]
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

# Correctly combine frozen and movable dipoles for visualization.
if frozen_dipoles_line is not None and frozen_dipoles_line.size > 0:
    _scatterers_vis = np.hstack((frozen_dipoles_line, movable_dipoles_line))
else:
    # If there are no frozen dipoles, just use the movable ones.
    _scatterers_vis = movable_dipoles_line

base_sim_for_vis = CoupledDipoleSimulator(**sim_kwargs)
base_sim_for_vis.scatterers = _scatterers_vis

# Initialize Optimizer (used for Blocks 5, 6, and 7 onwards)
optimizer_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for Optimizer: {optimizer_device}")
optimizer = Optimizer(F_operator_tensor=fisher_calc.F, device=optimizer_device)

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

    if (
        verbose_plotting
    ):  # Visualize the intensity pattern for each single-parameter MIS
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

# %% 6) Combined wavefront via weighted trace eigendecomposition

print(
    "\n--- Section 6: Combined Wavefront (Weighted Trace J via Eigendecomposition) ---"
)

weighted_trace_results = optimizer.get_weighted_trace_mis(
    photon_budget=photon_budget
)

# Extract the results from the returned dictionary
trace_result_data = weighted_trace_results["weighted_trace"]
trace_opt_eigval = trace_result_data["eigenvalue"]
trace_opt_wavefront_eig = trace_result_data["wavefront"]

# Store the optimized wavefront and its corresponding Fisher matrix
optimized_wavefronts["TraceJ_norm_eig"] = trace_opt_wavefront_eig
J_for_trace_opt_eig = optimizer.J_fisher_info_matrix(trace_opt_wavefront_eig)
fisher_matrices_for_wavefronts["TraceJ_norm_eig"] = J_for_trace_opt_eig

if verbose_plotting:
    print(f"Visualizing intensity for TraceJ_norm_eig...")
    base_sim_for_vis.reset()
    base_sim_for_vis.set_input_field_kx(kx_array, trace_opt_wavefront_eig)
    base_sim_for_vis.solve_fields()
    base_sim_for_vis.visualize_intensity(
        title=f"Normalized Trace J Optimized Wavefront (Eigendecomposition of sum w_i * F_mumu)\nMax Eigenvalue = {trace_opt_eigval:.2e}",
        dpi=100,
    )

# %% 7a) D-optimality: Maximize Determinant of J (via torch)

print("\n--- Section 7a: D-Optimality (Maximize det(J)) ---")

# Perform the optimization using the multi-run method
# We want to maximize the determinant of J.
det_J_optimization_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=lossfunctions.determinant_of_J,
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

optimized_wavefronts["DetJ_torch"] = wavefront_detJ_opt
J_for_detJ_opt = optimizer.J_fisher_info_matrix(wavefront_detJ_opt)
fisher_matrices_for_wavefronts["DetJ_torch"] = J_for_detJ_opt
print(f"Optimization complete. Best det(J) = {best_detJ_value:.3e}")


if verbose_plotting:
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

# %% 7b) E-optimality: Maximize smallest eigenvalue of J (Minimize spectral radius of J^-1)

print("\n--- Section 7b: E-Optimality (Maximize min eigenvalue of J) ---")

# Perform the optimization
# We want to maximize the smallest eigenvalue of J.
min_eig_J_optimization_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=lossfunctions.maximize_min_eig_J,
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


optimized_wavefronts["SpecJ_torch"] = (
    wavefront_min_eig_J_opt  # Key used in original script for spectral opt
)
J_for_min_eig_J_opt = optimizer.J_fisher_info_matrix(wavefront_min_eig_J_opt)
fisher_matrices_for_wavefronts["SpecJ_torch"] = J_for_min_eig_J_opt
print(
    f"Optimization complete. Best min eigenvalue(J) = {best_min_eig_J_value:.3e}"
)

if verbose_plotting:
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

# %% 7c) A-optimality: Minimize Trace of J^-1 (sum of variances)

print("\n--- Section 7c: A-Optimality (Minimize Tr(J^-1)) ---")

# Perform the optimization
# We want to minimize the trace of J_inverse.
trace_J_inv_optimization_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=lossfunctions.minimize_trace_of_J_inv,
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

optimized_wavefronts["TraceJinv_torch"] = wavefront_trace_J_inv_opt
J_for_trace_J_inv_opt = optimizer.J_fisher_info_matrix(
    wavefront_trace_J_inv_opt
)
fisher_matrices_for_wavefronts["TraceJinv_torch"] = J_for_trace_J_inv_opt
print(f"Optimization complete. Best Tr(J^-1) = {best_trace_J_inv_value:.3e}")

if verbose_plotting:
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


# %% 8) SubDet

loss_histories = {}
nuisance_param_index = 2  # Corresponds to rotation

strategies = {
    # "LogLin": {
    #     "loss_fn_constructor": lambda J_mat: lossfunctions.loglin_loss(
    #         J_mat, nuisance_param_index=nuisance_param_index, alpha=0.1
    #     ),
    #     "params": {"alpha": 0.1},
    # },
    # "OffDiag": {
    #     "loss_fn_constructor": lambda J_mat: lossfunctions.offdiag_loss(
    #         J_mat, nuisance_param_index=nuisance_param_index, beta=1.0
    #     ),
    #     "params": {"beta": 1.0},
    # },
    "SubDet": {
        "loss_fn_constructor": lambda J_mat: lossfunctions.subdet_loss(
            J_mat, nuisance_param_index=nuisance_param_index
        ),
        "params": {},
    },
}

for strategy_name, config in strategies.items():
    print(f"\n--- Optimizing for Strategy: {strategy_name} ---")

    loss_fn = config["loss_fn_constructor"]

    optimization_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=loss_fn,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=False,
        photon_budget=photon_budget,
        base_seed=42,
    )

    loss_histories[strategy_name] = optimization_results["all_loss_histories"]
    bestvec = optimization_results["bestvec"]
    bestval = optimization_results["bestval"]

    wavefront_key = f"{strategy_name}_torch"

    optimized_wavefronts[wavefront_key] = bestvec
    J_for_strategy = optimizer.J_fisher_info_matrix(bestvec)
    fisher_matrices_for_wavefronts[wavefront_key] = J_for_strategy
    print(
        f"Optimization for {strategy_name} complete. Best loss value = {bestval:.3e}"
    )

    if verbose_plotting:
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


# %% 9a) D-Optimality for Marginal FIM (Maximize det(J_marginal))

nuisance_param_index_for_marginal = 2  # Rotation

loss_fn_det_marginal = (
    lambda J_full_torch: lossfunctions.determinant_of_marginal_J(
        optimizer._compute_marginal_fisher_info_matrix(
            J_full_torch, nuisance_param_index_for_marginal
        )
    )
)

det_J_marginal_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=loss_fn_det_marginal,
    runs=multi_run_count,
    steps=single_run_iterations,
    lr=1e-3,
    verbose=True,
    maximize=True,
    photon_budget=photon_budget,
    base_seed=42,
)

wavefront_detJ_marginal_opt = det_J_marginal_results["bestvec"]
best_detJ_marginal_value = det_J_marginal_results["bestval"]
opt_key_13a = f"DetJ_marginal_n{nuisance_param_index_for_marginal}_torch"

optimized_wavefronts[opt_key_13a] = wavefront_detJ_marginal_opt
J_full_for_detJ_marginal = optimizer.J_fisher_info_matrix(
    wavefront_detJ_marginal_opt
)
fisher_matrices_for_wavefronts[opt_key_13a] = J_full_for_detJ_marginal
print(
    f"Optimization complete. Best det(J_marginal) = {best_detJ_marginal_value:.3e}"
)

if verbose_plotting:
    plt.figure(figsize=(8, 5))
    for i, history in enumerate(det_J_marginal_results["all_loss_histories"]):
        plt.plot(history, label=f"Run {i+1}")
    plt.title(
        f"D-Optimality for Marginal FIM (det(J_marginal)) (All {multi_run_count} Runs)"
    )
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


# %% 9b) A-Optimality for Marginal FIM (Minimize Tr(J_marginal^-1))

loss_fn_trace_inv_marginal = (
    lambda J_full_torch: lossfunctions.minimize_trace_of_marginal_J_inv(
        optimizer._compute_marginal_fisher_info_matrix(
            J_full_torch, nuisance_param_index_for_marginal
        )
    )
)

trace_J_inv_marginal_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=loss_fn_trace_inv_marginal,
    runs=multi_run_count,
    steps=single_run_iterations,
    lr=1e-3,
    verbose=True,
    maximize=False,  # Minimization problem
    photon_budget=photon_budget,
    base_seed=42,  # Vary seed slightly
)

wavefront_traceJinv_marginal_opt = trace_J_inv_marginal_results["bestvec"]
best_traceJinv_marginal_value = trace_J_inv_marginal_results["bestval"]
opt_key_13b = f"TraceJinv_marginal_n{nuisance_param_index_for_marginal}_torch"

optimized_wavefronts[opt_key_13b] = wavefront_traceJinv_marginal_opt
J_full_for_traceJinv_marginal = optimizer.J_fisher_info_matrix(
    wavefront_traceJinv_marginal_opt
)
fisher_matrices_for_wavefronts[opt_key_13b] = J_full_for_traceJinv_marginal
print(
    f"Optimization complete. Best Tr(J_marginal^-1) = {best_traceJinv_marginal_value:.3e}"
)

if verbose_plotting:
    plt.figure(figsize=(8, 5))
    for i, history in enumerate(
        trace_J_inv_marginal_results["all_loss_histories"]
    ):
        plt.plot(history, label=f"Run {i+1}")
    plt.title(
        f"A-Optimality for Marginal FIM (Tr(J_marginal^-1)) (All {multi_run_count} Runs)"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Tr(J_marginal^-1)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

    base_sim_for_vis.reset()
    base_sim_for_vis.set_input_field_kx(
        kx_array, wavefront_traceJinv_marginal_opt
    )
    base_sim_for_vis.solve_fields()
    base_sim_for_vis.visualize_intensity(
        title=f"A-Optimized Wavefront for Marginal FIM\nMin Tr(J_marginal^-1) = {best_traceJinv_marginal_value:.2e}",
        dpi=100,
    )


# %% 9c) E-Optimality for Marginal FIM (Maximize min_eig(J_marginal))

loss_fn_min_eig_marginal = (
    lambda J_full_torch: lossfunctions.maximize_min_eig_marginal_J(
        optimizer._compute_marginal_fisher_info_matrix(
            J_full_torch, nuisance_param_index_for_marginal
        )
    )
)

min_eig_J_marginal_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=loss_fn_min_eig_marginal,
    runs=multi_run_count,
    steps=single_run_iterations,
    lr=1e-3,
    verbose=True,
    maximize=True,
    photon_budget=photon_budget,
    base_seed=42,  # Vary seed slightly
)

wavefront_mineigJ_marginal_opt = min_eig_J_marginal_results["bestvec"]
best_mineigJ_marginal_value = min_eig_J_marginal_results["bestval"]
opt_key_13c = f"SpecJ_marginal_n{nuisance_param_index_for_marginal}_torch"

optimized_wavefronts[opt_key_13c] = wavefront_mineigJ_marginal_opt
J_full_for_mineigJ_marginal = optimizer.J_fisher_info_matrix(
    wavefront_mineigJ_marginal_opt
)
fisher_matrices_for_wavefronts[opt_key_13c] = J_full_for_mineigJ_marginal
print(
    f"Optimization complete. Best min_eig(J_marginal) = {best_mineigJ_marginal_value:.3e}"
)

if verbose_plotting:
    plt.figure(figsize=(8, 5))
    for i, history in enumerate(
        min_eig_J_marginal_results["all_loss_histories"]
    ):
        plt.plot(history, label=f"Run {i+1}")
    plt.title(
        f"E-Optimality for Marginal FIM (min_eig(J_marginal)) (All {multi_run_count} Runs)"
    )
    plt.xlabel("Iteration")
    plt.ylabel("min_eig(J_marginal)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

    base_sim_for_vis.reset()
    base_sim_for_vis.set_input_field_kx(
        kx_array, wavefront_mineigJ_marginal_opt
    )
    base_sim_for_vis.solve_fields()
    base_sim_for_vis.visualize_intensity(
        title=f"E-Optimized Wavefront for Marginal FIM\nMax min_eig(J_marginal) = {best_mineigJ_marginal_value:.2e}",
        dpi=100,
    )


# %% 10) Compare CRLB, with optional plane wave or random wavefront comparison

crlbanalysis.visualize_crlb_comparison(
    optimized_wavefronts=optimized_wavefronts,
    F_operator_tensor=optimizer.F_op,
    photon_budget=photon_budget,
    param_names=param_names,
    sim_kwargs=sim_kwargs,
    kx_array=kx_array,
    figure_title="CRLB Comparison for different types of Maximum Information States",
    show_plane_waves=False,
    show_random_wavefronts=False,
    num_random_samples=100,
)

# %% 11) Generate Pareto frontier plots
print("\n--- Block 17: Generating Pareto frontier plots... ---")

# 1. Create wavefront_categories dictionary
wavefront_categories_map = {}
for key in optimized_wavefronts.keys():
    if key.startswith("MIS_"):
        wavefront_categories_map[key] = "MIS"
    elif "TraceJ_eig" == key:
        wavefront_categories_map[key] = "TraceJ_eig"
    elif key in ["DetJ_torch", "SpecJ_torch", "TraceJinv_torch"]:
        wavefront_categories_map[key] = "TorchFull"
    elif key in ["LogLin_torch", "OffDiag_torch", "SubDet_torch"]:
        wavefront_categories_map[key] = "NuisanceOriginal"
    elif (
        key.startswith("DetJ_marginal_")
        or key.startswith("TraceJinv_marginal_")
        or key.startswith("SpecJ_marginal_")
    ):
        wavefront_categories_map[key] = "TorchMarginal"
    else:
        wavefront_categories_map[key] = "Other"

# 2. Create wavefront_labels_latex dictionary
wavefront_labels_latex = {
    "MIS_shiftX": r"MIS$_x$",
    "MIS_shiftZ": r"MIS$_z$",
    "MIS_rotation": r"MIS$_\theta$",
    "TraceJ_eig": r"Max Tr($\mathcal{J}$)$_{eig}$",
    "DetJ_torch": r"Max Det($\mathcal{J}$)",
    "SpecJ_torch": r"Max MinEig($\mathcal{J}$)",
    "TraceJinv_torch": r"Min Tr($\mathcal{J}^{-1}$)",
    "LogLin_torch": r"LogLin",
    "OffDiag_torch": r"OffDiag",
    "SubDet_torch": r"SubDet",
    "DetJ_marginal_n2_torch": r"Max Det($\mathcal{J}_{\psi|\phi}$)",
    "TraceJinv_marginal_n2_torch": r"Min Tr($\mathcal{J}^{-1}_{\psi|\phi}$)",
    "SpecJ_marginal_n2_torch": r"Max MinEig($\mathcal{J}_{\psi|\phi}$)",
}
# Add default labels for any missing keys
for key in optimized_wavefronts.keys():
    if key not in wavefront_labels_latex:
        wavefront_labels_latex[key] = key.replace("_", " ").replace(
            "torch", ""
        )


# 3. Define category_styles
category_styles_def = {
    "MIS": {
        "marker": "x",
        "color": "purple",
        "label": "MIS via eigenvalue decomposition",
    },
    "TraceJ_eig": {
        "marker": "x",
        "color": "purple",
        "label": "MIS via eigenvalue decomposition",
    },
    "TorchFull": {"marker": "o", "color": "blue", "label": "Full FIM Optim."},
    "NuisanceOriginal": {
        "marker": "s",
        "color": "green",
        "label": "Nuisance-engineered FIM via gradent descent optimization",
    },
    "TorchMarginal": {
        "marker": "P",
        "color": "red",
        "label": "Marginal FIM via gradent descent optimization",
    },
    "Other": {"marker": "d", "color": "grey", "label": "Other"},
    "default": {
        "marker": "*",
        "color": "black",
        "label": "Gradient descent optimization",
    },  # Fallback for the plotting function
}


# Call 1: CRLB_x vs CRLB_z
print("\nPlotting CRLB_x vs CRLB_z...")
crlbanalysis.plot_pareto_frontiers(
    optimized_wavefronts_dict=optimized_wavefronts,
    F_operator_tensor_np=optimizer.F_op,
    photon_budget_val=photon_budget,
    param_names_list=param_names,
    params_to_plot_indices=[0, 1],
    wavefront_categories_style=category_styles_def,
    wavefront_labels_latex_map=wavefront_labels_latex,
    filepath="pareto_frontier_x_vs_z.png",
    dpi=300,
)

# %% 11b) Leaner plot alternative

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

# 2. Create wavefront_labels_latex dictionary with unified typography from previous steps
wavefront_labels_latex = {
    "MIS_shiftX": r"MIS$_{x}$",
    "MIS_shiftZ": r"MIS$_{z}$",
    "MIS_rotation": r"MIS$_{\theta}$",
    "TraceJ_norm_eig": r"$\max\,\mathrm{Tr}(\mathcal{J})$",
    "DetJ_torch": r"$\max\,\det(\mathcal{J})$",
    "SpecJ_torch": r"$\max\,\lambda_{\min}(\mathcal{J})$",
    "TraceJinv_torch": r"$\min\,\mathrm{Tr}(\mathcal{J}^{-1})$",
    "LogLin_torch": r"$\text{LogLin}$",
    "OffDiag_torch": r"$\text{OffDiag}$",
    # "SubDet_torch": r"$\text{SubDet}$",
    "SubDet_torch": r"$\max\,\det(\mathcal{J}_{\psi|\psi})$",
    "DetJ_marginal_n2_torch": r"$\max\,\det(\mathcal{J}_{\psi|\phi})$",
    "TraceJinv_marginal_n2_torch": r"$\min\,\mathrm{Tr}((\mathcal{J}_{\psi|\phi})^{-1})$",
    "SpecJ_marginal_n2_torch": r"$\max\,\lambda_{\min}(\mathcal{J}_{\psi|\phi})$",
}


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
)
