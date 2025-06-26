import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator


# %% -1) define functions
def measure_heterodyne(E, total_photons, nsamples=None, seed=None):
    """
    Models a camera with complex field detection under shot noise, consistent
    with the quantum mechanical model where the variance of a quadrature
    measurement is 1/2.
    """
    signal_field = E * np.sqrt(total_photons)
    # The variance of the noise on each quadrature (real, imag) is 1/2.
    # Therefore, the standard deviation is sqrt(0.5).
    noise_std_dev = np.sqrt(0.5)
    E_real = signal_field.view(np.float64)
    rng = np.random.RandomState(seed)
    if nsamples is None:
        field_samples = stats.norm.rvs(
            loc=E_real, scale=noise_std_dev, random_state=rng
        )
    else:
        field_samples = stats.norm.rvs(
            loc=E_real,
            scale=noise_std_dev,
            size=(nsamples, *E_real.shape),
            random_state=rng,
        )
    return field_samples.view(np.complex128) / np.sqrt(total_photons)


def best_estimate_multi(Z_noisy, E0, dE_vectors, J_inv):
    """
    Optimal multi-parameter linear estimator.
    """
    H = np.asarray(dE_vectors).T
    delta_E = (Z_noisy - E0).reshape(-1, 1)
    theta_est_vector = J_inv @ H.conj().T @ delta_E
    return np.real(theta_est_vector.flatten())


def best_estimate(E, dE_dtheta, E0):
    """
    "Naive" single-parameter estimator.
    """
    theta_est = np.vecdot(dE_dtheta, E - E0) / np.vecdot(dE_dtheta, dE_dtheta)
    theta_est = np.real(theta_est)
    return theta_est


# %% 0) settings
OPTIMALITY_CRITERION = "D"  # Can be 'D', 'A', or 'E'

verbose_plotting = True
max_angle = 80
number_of_input_channels = 100
number_of_output_channels = 300
param_names = ["shiftX", "shiftZ", "rotation"]
delta_param = 1e-3
photon_budget = 1.0
multi_run_count = 5
single_run_iterations = 4000

# %% 1) define dipole configurations
# movable_dipoles_line = placedipoles.place_line_of_dipoles(
#     center=(5.0, 0.0), number=7
# )
# frozen_dipoles_line = placedipoles.place_line_of_dipoles(
#     center=(3.0, 0.0), number=7
# )

### Quasi-random double line
movable_dipoles_line = placedipoles.place_quasi_random_line_of_dipoles(
    center=(5.2, 0.0), number=6, small_dist=0.786, start_index=6
)
frozen_dipoles_line = placedipoles.place_quasi_random_line_of_dipoles(
    center=(3.0, 0.0), number=6, small_dist=0.786, start_index=26
)
# frozen_dipoles_line = None

# ### Square
# movable_dipoles_line = placedipoles.place_square_of_dipoles(
#     center=(5.0, 0.0), side=1.0
# )

# frozen_dipoles_line = placedipoles.place_line_of_dipoles(
#     center=(2.5, 0.0), number=7
# )
# # frozen_dipoles_line = None


# %% 2) simulator geometry and input
total_Ns_sim = movable_dipoles_line.shape[1]
if frozen_dipoles_line is not None and frozen_dipoles_line.size > 0:
    total_Ns_sim += frozen_dipoles_line.shape[1]

sim_kwargs = dict(
    lambda_0=1.0,
    Lz=8.0,
    Lx=8.0,
    Ns=total_Ns_sim,
    seed=123,
    grid_points_x=number_of_output_channels,
)
angles_deg = np.linspace(-max_angle, max_angle, number_of_input_channels)
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
optimizer_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for Optimizer: {optimizer_device}")
optimizer = Optimizer(F_operator_tensor=fisher_calc.F, device=optimizer_device)
optimized_wavefronts = {}
fisher_matrices_for_wavefronts = {}

# %% 5) compute optimal wavefronts

# --- 5.1) Baseline: Single-parameter MIS for X ---
print("\n--- Calculating Single-Parameter MIS for X ---")
F_xx_op = optimizer.F_op[0, 0, :, :]
_, wavefront_mis_x_eig = optimizer.eig_optimize_matrix(
    F_xx_op, maximize=True, num=1
)
wavefront_mis_x = optimizer.normalize_wavefront(
    wavefront_mis_x_eig, photon_budget
)
optimized_wavefronts["MIS_x"] = wavefront_mis_x
J_for_mis_x = optimizer.J_fisher_info_matrix(wavefront_mis_x)
fisher_matrices_for_wavefronts["MIS_x"] = J_for_mis_x

# --- 5.2) Nuisance-Aware (Marginal FIM) Optimization ---
nuisance_param_index_for_marginal = param_names.index("rotation")
marginal_criteria_config = {
    "D": {
        "loss_fn": lambda J_full: lossfunctions.determinant_of_marginal_J(
            optimizer._compute_marginal_fisher_info_matrix(
                J_full, nuisance_param_index_for_marginal
            )
        ),
        "maximize": True,
        "name": "Maximize det(J_marginal_XZ|theta)",
    },
    "A": {
        "loss_fn": lambda J_full: lossfunctions.minimize_trace_of_marginal_J_inv(
            optimizer._compute_marginal_fisher_info_matrix(
                J_full, nuisance_param_index_for_marginal
            )
        ),
        "maximize": False,
        "name": "Minimize Tr((J_marginal_XZ|theta)^-1)",
    },
    "E": {
        "loss_fn": lambda J_full: lossfunctions.maximize_min_eig_marginal_J(
            optimizer._compute_marginal_fisher_info_matrix(
                J_full, nuisance_param_index_for_marginal
            )
        ),
        "maximize": True,
        "name": "Maximize min_eig(J_marginal_XZ|theta)",
    },
}
selected_marginal_config = marginal_criteria_config[OPTIMALITY_CRITERION]
print(
    f"\n--- {OPTIMALITY_CRITERION}-Optimality for Nuisance-Aware Marginal FIM ({selected_marginal_config['name']}) ---"
)
nuisance_aware_opt_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=selected_marginal_config["loss_fn"],
    runs=multi_run_count,
    steps=single_run_iterations,
    lr=1e-3,
    verbose=True,
    maximize=selected_marginal_config["maximize"],
    photon_budget=photon_budget,
    base_seed=42,
)
marginal_result_key = f"NuisanceAware_{OPTIMALITY_CRITERION}opt_Marginal"
optimized_wavefronts[marginal_result_key] = nuisance_aware_opt_results[
    "bestvec"
]
fisher_matrices_for_wavefronts[marginal_result_key] = (
    optimizer.J_fisher_info_matrix(nuisance_aware_opt_results["bestvec"])
)

# --- 5.3) Nuisance-Ignorant (Sub-matrix FIM) Optimization ---
nuisance_param_index_for_subdet = param_names.index("rotation")
subdet_criteria_config = {
    "D": {
        "loss_fn": lambda J_full: lossfunctions.subdet_loss(
            J_full, nuisance_param_index=nuisance_param_index_for_subdet
        ),
        "maximize": False,
        "name": "Maximize det(J_XZ)",
    },
    "A": {
        "loss_fn": lambda J_full: lossfunctions.trace_of_J_block_inv(
            lossfunctions.compute_J_SubDet_given_nuisance(
                J_full, nuisance_param_index_for_subdet
            )
        ),
        "maximize": False,
        "name": "Minimize Tr((J_XZ)^-1)",
    },
    "E": {
        "loss_fn": lambda J_full: lossfunctions.min_eigenvalue_of_J_block(
            lossfunctions.compute_J_SubDet_given_nuisance(
                J_full, nuisance_param_index_for_subdet
            )
        ),
        "maximize": True,
        "name": "Maximize min_eig(J_XZ)",
    },
}
selected_subdet_config = subdet_criteria_config[OPTIMALITY_CRITERION]
print(
    f"\n--- {OPTIMALITY_CRITERION}-Optimality for Sub-determinant FIM ({selected_subdet_config['name']}) ---"
)
subdet_xz_opt_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=selected_subdet_config["loss_fn"],
    runs=multi_run_count,
    steps=single_run_iterations,
    lr=1e-3,
    verbose=True,
    maximize=selected_subdet_config["maximize"],
    photon_budget=photon_budget,
    base_seed=123,
)
subdet_result_key = f"Subdet_XZ_{OPTIMALITY_CRITERION}opt"
optimized_wavefronts[subdet_result_key] = subdet_xz_opt_results["bestvec"]
fisher_matrices_for_wavefronts[subdet_result_key] = (
    optimizer.J_fisher_info_matrix(subdet_xz_opt_results["bestvec"])
)

# --- 5.4) Measure-Everything (Full FIM) Optimization ---
full_fim_criteria_config = {
    "D": {
        "loss_fn": lossfunctions.determinant_of_J,
        "maximize": True,
        "name": "Maximize det(J_full)",
    },
    "A": {
        "loss_fn": lossfunctions.minimize_trace_of_J_inv,
        "maximize": False,
        "name": "Minimize Tr(J_full^-1)",
    },
    "E": {
        "loss_fn": lossfunctions.maximize_min_eig_J,
        "maximize": True,
        "name": "Maximize min_eig(J_full)",
    },
}
selected_full_fim_config = full_fim_criteria_config[OPTIMALITY_CRITERION]
print(
    f"\n--- {OPTIMALITY_CRITERION}-Optimality for Full FIM ({selected_full_fim_config['name']}) ---"
)
full_fim_opt_results = optimizer.torch_optimize_loss_multirun(
    loss_function_of_J=selected_full_fim_config["loss_fn"],
    runs=multi_run_count,
    steps=single_run_iterations,
    lr=1e-3,
    verbose=True,
    maximize=selected_full_fim_config["maximize"],
    photon_budget=photon_budget,
    base_seed=88,
)
full_fim_result_key = f"FullFIM_{OPTIMALITY_CRITERION}opt"
optimized_wavefronts[full_fim_result_key] = full_fim_opt_results["bestvec"]
fisher_matrices_for_wavefronts[full_fim_result_key] = (
    optimizer.J_fisher_info_matrix(full_fim_opt_results["bestvec"])
)


# %% 6) Run simulation with the multi-parameter estimator
total_photons = 1e6
N_estimations = int(1e3)
theta0_rad = 0.0
x_true_pos = 0.0
z_true_pos = 0.0
N_deviations = 25
max_deviation = 0.1
theta_deviations = np.linspace(-max_deviation, max_deviation, N_deviations)

# Lists to store results for each strategy
all_estimates_nuisance_aware = []
all_estimates_subdet_xz = []
all_estimates_full_fim = []
all_estimates_mis_x = []

print("\nPre-calculating estimator components...")
sim = CoupledDipoleSimulator(**sim_kwargs)
nominal_movable_dipoles = placedipoles.shift_and_rotate_dipoles(
    movable_dipoles_line, 0, 0, 0
)
if frozen_dipoles_line is not None and frozen_dipoles_line.size > 0:
    sim.scatterers = np.hstack((frozen_dipoles_line, nominal_movable_dipoles))
else:
    sim.scatterers = nominal_movable_dipoles

if verbose_plotting:
    sim.visualize_intensity()
    sim.reset()

# Define keys based on optimality criterion
nuisance_aware_wf_key = f"NuisanceAware_{OPTIMALITY_CRITERION}opt_Marginal"
subdet_xz_wf_key = f"Subdet_XZ_{OPTIMALITY_CRITERION}opt"
full_fim_wf_key = f"FullFIM_{OPTIMALITY_CRITERION}opt"

# --- Estimator Components Setup ---
# Nuisance-Aware (Marginal FIM)
print(f"\nSetting up components for: {nuisance_aware_wf_key}")
wf_nuisance_aware = optimized_wavefronts[nuisance_aware_wf_key]
dE_nuisance_aware = [
    dS_tensor[i] @ wf_nuisance_aware for i in range(len(param_names))
]
sim.set_input_field_kx(kx_array, wf_nuisance_aware)
E0_nuisance_aware = sim.compute_zL_fields_direct()
J_nuisance_aware = fisher_matrices_for_wavefronts[nuisance_aware_wf_key]
J_inv_nuisance_aware = np.linalg.inv(J_nuisance_aware)

# Sub-determinant XZ
print(f"Setting up components for: {subdet_xz_wf_key}")
wf_subdet_xz = optimized_wavefronts[subdet_xz_wf_key]
dE_subdet_xz_est = [
    dS_tensor[param_names.index("shiftX")] @ wf_subdet_xz,
    dS_tensor[param_names.index("shiftZ")] @ wf_subdet_xz,
]
sim.set_input_field_kx(kx_array, wf_subdet_xz)
E0_subdet_xz = sim.compute_zL_fields_direct()
J_subdet_xz_full = fisher_matrices_for_wavefronts[subdet_xz_wf_key]
idx_x, idx_z = param_names.index("shiftX"), param_names.index("shiftZ")
J_subdet_xz_2x2 = J_subdet_xz_full[np.ix_([idx_x, idx_z], [idx_x, idx_z])]
J_inv_subdet_xz_2x2 = np.linalg.inv(J_subdet_xz_2x2)

# Full FIM
print(f"Setting up components for: {full_fim_wf_key}")
wf_full_fim = optimized_wavefronts[full_fim_wf_key]
dE_full_fim = [dS_tensor[i] @ wf_full_fim for i in range(len(param_names))]
sim.set_input_field_kx(kx_array, wf_full_fim)
E0_full_fim = sim.compute_zL_fields_direct()
J_full_fim = fisher_matrices_for_wavefronts[full_fim_wf_key]
J_inv_full_fim = np.linalg.inv(J_full_fim)

# MIS_x
print("Setting up components for: MIS_x")
wf_mis_x = optimized_wavefronts["MIS_x"]
dE_mis_x = dS_tensor[param_names.index("shiftX")] @ wf_mis_x
sim.set_input_field_kx(kx_array, wf_mis_x)
E0_mis_x = sim.compute_zL_fields_direct()
J_mis_x_full = fisher_matrices_for_wavefronts["MIS_x"]

print("\nStarting estimation loop...")
for i, theta_dev in enumerate(theta_deviations):
    print(
        f"\rSimulating deviation {i+1}/{len(theta_deviations)}: {theta_dev:.4f} rad",
        end="     ",
    )
    theta_true = theta0_rad + theta_dev
    true_movable_dipoles = placedipoles.shift_and_rotate_dipoles(
        movable_dipoles_line, x_true_pos, z_true_pos, theta_true
    )
    if frozen_dipoles_line is not None and frozen_dipoles_line.size > 0:
        sim.scatterers = np.hstack((frozen_dipoles_line, true_movable_dipoles))
    else:
        sim.scatterers = true_movable_dipoles
    sim.reset()

    # --- Run Estimations for Each Strategy ---
    # Nuisance-Aware
    sim.set_input_field_kx(kx_array, wf_nuisance_aware)
    E_true = sim.compute_zL_fields_direct()
    Z_noisy = measure_heterodyne(
        E_true, total_photons, nsamples=N_estimations, seed=i * 10
    )
    estimates = np.array(
        [
            best_estimate_multi(
                Z, E0_nuisance_aware, dE_nuisance_aware, J_inv_nuisance_aware
            )
            for Z in Z_noisy
        ]
    )
    all_estimates_nuisance_aware.append(estimates)

    # Sub-determinant XZ
    sim.set_input_field_kx(kx_array, wf_subdet_xz)
    E_true = sim.compute_zL_fields_direct()
    Z_noisy = measure_heterodyne(
        E_true, total_photons, nsamples=N_estimations, seed=i * 10 + 1
    )
    estimates = np.array(
        [
            best_estimate_multi(
                Z, E0_subdet_xz, dE_subdet_xz_est, J_inv_subdet_xz_2x2
            )
            for Z in Z_noisy
        ]
    )
    all_estimates_subdet_xz.append(estimates)

    # Full FIM
    sim.set_input_field_kx(kx_array, wf_full_fim)
    E_true = sim.compute_zL_fields_direct()
    Z_noisy = measure_heterodyne(
        E_true, total_photons, nsamples=N_estimations, seed=i * 10 + 2
    )
    estimates = np.array(
        [
            best_estimate_multi(Z, E0_full_fim, dE_full_fim, J_inv_full_fim)
            for Z in Z_noisy
        ]
    )
    all_estimates_full_fim.append(estimates)

    # MIS_x
    sim.set_input_field_kx(kx_array, wf_mis_x)
    E_true = sim.compute_zL_fields_direct()
    Z_noisy = measure_heterodyne(
        E_true, total_photons, nsamples=N_estimations, seed=i * 10 + 3
    )
    estimates = np.array(
        [best_estimate(Z, dE_mis_x, E0_mis_x) for Z in Z_noisy]
    )
    all_estimates_mis_x.append(estimates)

print("\nSimulation loop finished.")


# %% 7) Plotting Bias and Precision for X and Z Parameters
def plot_performance_on_axes(axes, param_name, idx_in_3param, idx_in_2param):
    """
    Plots bias and precision for a given parameter on a set of two axes.
    `axes` should be a tuple or list of two axes: (ax_bias, ax_precision).
    """
    ax1, ax2 = axes

    # Calculate Bias
    bias_nuisance_aware = [
        np.mean(est[:, idx_in_3param]) - x_true_pos
        for est in all_estimates_nuisance_aware
    ]
    bias_subdet_xz = [
        np.mean(est[:, idx_in_2param]) - x_true_pos
        for est in all_estimates_subdet_xz
        if idx_in_2param is not None
    ]
    bias_full_fim = [
        np.mean(est[:, idx_in_3param]) - x_true_pos
        for est in all_estimates_full_fim
    ]
    bias_mis_x = (
        [np.mean(est) - x_true_pos for est in all_estimates_mis_x]
        if param_name == "X"
        else None
    )

    # Calculate Standard Deviation
    std_nuisance_aware = [
        np.std(est[:, idx_in_3param]) for est in all_estimates_nuisance_aware
    ]
    std_subdet_xz = [
        np.std(est[:, idx_in_2param])
        for est in all_estimates_subdet_xz
        if idx_in_2param is not None
    ]
    std_full_fim = [
        np.std(est[:, idx_in_3param]) for est in all_estimates_full_fim
    ]
    std_mis_x = (
        [np.std(est) for est in all_estimates_mis_x]
        if param_name == "X"
        else None
    )

    # Calculate CRLBs for each deviation
    crlb_prefactor = 1 / np.sqrt(2)
    crlb_nuisance_aware_list = []
    crlb_subdet_xz_list = []
    crlb_full_fim_list = []
    crlb_mis_x_list = []

    print(
        f"\nRecalculating CRLB for each nuisance deviation for {param_name}-parameter..."
    )
    for i, theta_dev in enumerate(theta_deviations):
        print(
            f"\rCalculating CRLB for deviation {i+1}/{len(theta_deviations)}: {theta_dev:.4f} rad",
            end="     ",
        )
        # Recalculate dS for the current deviation
        dS_tensor_dev = placedipoles.calculate_dS_for_3_parameters(
            sim_kwargs,
            angles_deg,
            movable_dipoles=placedipoles.shift_and_rotate_dipoles(
                movable_dipoles_line, 0, 0, theta_dev
            ),
            frozen_dipoles=frozen_dipoles_line,
            delta_param=delta_param,
            flux_normalize=True,
        )
        fisher_calc_dev = FisherMatrix(
            dS=dS_tensor_dev, param_names=param_names, delta=delta_param
        )
        optimizer_dev = Optimizer(
            F_operator_tensor=fisher_calc_dev.F, device=optimizer_device
        )

        # Nuisance-Aware
        J_nuisance_aware_dev = optimizer_dev.J_fisher_info_matrix(
            wf_nuisance_aware
        )
        J_inv_nuisance_aware_dev = np.linalg.inv(J_nuisance_aware_dev)
        crlb_nuisance_aware_list.append(
            crlb_prefactor
            * np.sqrt(J_inv_nuisance_aware_dev[idx_in_3param, idx_in_3param])
        )

        # Sub-determinant XZ
        if idx_in_2param is not None:
            J_subdet_xz_full_dev = optimizer_dev.J_fisher_info_matrix(
                wf_subdet_xz
            )
            J_subdet_xz_2x2_dev = J_subdet_xz_full_dev[
                np.ix_([idx_x, idx_z], [idx_x, idx_z])
            ]
            J_inv_subdet_xz_2x2_dev = np.linalg.inv(J_subdet_xz_2x2_dev)
            crlb_subdet_xz_list.append(
                crlb_prefactor
                * np.sqrt(
                    J_inv_subdet_xz_2x2_dev[idx_in_2param, idx_in_2param]
                )
            )

        # Full FIM
        J_full_fim_dev = optimizer_dev.J_fisher_info_matrix(wf_full_fim)
        J_inv_full_fim_dev = np.linalg.inv(J_full_fim_dev)
        crlb_full_fim_list.append(
            crlb_prefactor
            * np.sqrt(J_inv_full_fim_dev[idx_in_3param, idx_in_3param])
        )

        # MIS_x
        if param_name == "X":
            J_mis_x_full_dev = optimizer_dev.J_fisher_info_matrix(wf_mis_x)
            crlb_mis_x_list.append(
                crlb_prefactor * np.sqrt(1 / J_mis_x_full_dev[idx_x, idx_x])
            )
    if crlb_subdet_xz_list:
        crlb_subdet_xz = np.array(crlb_subdet_xz_list)
    else:
        crlb_subdet_xz = None

    if crlb_mis_x_list:
        crlb_mis_x = np.array(crlb_mis_x_list)
    else:
        crlb_mis_x = None

    crlb_nuisance_aware = np.array(crlb_nuisance_aware_list)
    crlb_full_fim = np.array(crlb_full_fim_list)

    # --- Define Plotting Styles ---
    # Inspired by result_Pareto_frontier.py
    opt_criterion_label = f"{OPTIMALITY_CRITERION}-Opt."
    styles = {
        "FullFIM": {
            "label": rf"{opt_criterion_label} ($\mathcal{{J}}$)",
            "color": "#1f77b4",  # Blue
            "marker": "o",
        },
        "Subdet_XZ": {
            "label": rf"{opt_criterion_label} ($\mathcal{{J}}_{{\psi}})$",
            "color": "#ff7f0e",  # Orange
            "marker": "s",
        },
        "NuisanceAware": {
            "label": rf"{opt_criterion_label} ($\mathcal{{J}}_{{\psi|\phi}})$",
            "color": "#2ca02c",  # Green
            "marker": "d",
        },
        "MIS_x": {
            "label": r"MIS$_x$",
            "color": "#d62728",  # Red
            "marker": "^",
        },
    }

    # --- Bias Plot ---
    # Order: FullFIM, Subdet, Nuisance-Aware, MIS_x
    ax1.plot(
        theta_deviations,
        bias_full_fim,
        marker=styles["FullFIM"]["marker"],
        linestyle="-",
        label=styles["FullFIM"]["label"],
        color=styles["FullFIM"]["color"],
        markersize=8,
        lw=2,
    )
    if bias_subdet_xz:
        ax1.plot(
            theta_deviations,
            bias_subdet_xz,
            marker=styles["Subdet_XZ"]["marker"],
            linestyle="-",
            label=styles["Subdet_XZ"]["label"],
            color=styles["Subdet_XZ"]["color"],
            markersize=8,
            lw=2,
        )
    ax1.plot(
        theta_deviations,
        bias_nuisance_aware,
        marker=styles["NuisanceAware"]["marker"],
        linestyle="-",
        label=styles["NuisanceAware"]["label"],
        color=styles["NuisanceAware"]["color"],
        markersize=8,
        lw=2,
    )
    if bias_mis_x:
        ax1.plot(
            theta_deviations,
            bias_mis_x,
            marker=styles["MIS_x"]["marker"],
            linestyle="-",
            label=styles["MIS_x"]["label"],
            color=styles["MIS_x"]["color"],
            markersize=8,
            lw=2,
        )
    ax1.axhline(0, color="k", linestyle="--", label="Unbiased")
    ax1.set_ylabel(
        f"Estimation Bias of $\hat{{{param_name.lower()}}}$", fontsize=14
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # --- Precision Plot ---
    # Order: FullFIM, Subdet, Nuisance-Aware, MIS_x
    ax2.plot(
        theta_deviations,
        np.mean(std_full_fim)
        * np.ones_like(
            theta_deviations
        ),  # estimator is only dependent on noise and therefore invariant for changes in nuisance deviation
        linestyle="--",
        label=styles["FullFIM"]["label"],
        color=styles["FullFIM"]["color"],
        lw=2,
    )
    if std_subdet_xz:
        ax2.plot(
            theta_deviations,
            np.mean(std_subdet_xz)
            * np.ones_like(
                theta_deviations
            ),  # estimator is only dependent on noise and therefore invariant for changes in nuisance deviation
            linestyle="--",
            label=styles["Subdet_XZ"]["label"],
            color=styles["Subdet_XZ"]["color"],
            lw=2,
        )
    ax2.plot(
        theta_deviations,
        np.mean(std_nuisance_aware)
        * np.ones_like(
            theta_deviations
        ),  # estimator is only dependent on noise and therefore invariant for changes in nuisance deviation
        linestyle="--",
        label=styles["NuisanceAware"]["label"],
        color=styles["NuisanceAware"]["color"],
        lw=2,
    )
    if std_mis_x:
        ax2.plot(
            theta_deviations,
            np.mean(std_mis_x)
            * np.ones_like(
                theta_deviations
            ),  # estimator is only dependent on noise and therefore invariant for changes in nuisance deviation
            linestyle="--",
            label=styles["MIS_x"]["label"],
            color=styles["MIS_x"]["color"],
            lw=2,
        )

    # --- CRLB Plotting ---
    ax2.plot(
        theta_deviations,
        crlb_full_fim / np.sqrt(total_photons),
        color=styles["FullFIM"]["color"],
        marker=styles["FullFIM"]["marker"],
        linestyle="-",
        label=f"CRLB " + styles["FullFIM"]["label"],
    )
    if crlb_subdet_xz is not None:
        ax2.plot(
            theta_deviations,
            crlb_subdet_xz / np.sqrt(total_photons),
            color=styles["Subdet_XZ"]["color"],
            marker=styles["Subdet_XZ"]["marker"],
            linestyle="-",
            label=f"CRLB " + styles["Subdet_XZ"]["label"],
        )
    ax2.plot(
        theta_deviations,
        crlb_nuisance_aware / np.sqrt(total_photons),
        color=styles["NuisanceAware"]["color"],
        marker=styles["NuisanceAware"]["marker"],
        linestyle="-",
        label=f"CRLB " + styles["NuisanceAware"]["label"],
    )
    if crlb_mis_x is not None:
        ax2.plot(
            theta_deviations,
            crlb_mis_x / np.sqrt(total_photons),
            color=styles["MIS_x"]["color"],
            marker=styles["MIS_x"]["marker"],
            linestyle="-",
            label="CRLB " + styles["MIS_x"]["label"],
        )

    ax2.set_xlabel(
        "Nuisance Parameter Deviation ($\\theta_{true} - \\theta_0$) [rad]",
        fontsize=14,
    )
    ax2.set_ylabel(
        f"Estimation Error (Std. Dev. of $\hat{{{param_name.lower()}}}$)",
        fontsize=14,
    )
    ax2.legend(fontsize=12)
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax2.set_ylim(bottom=0)


# Main plotting block
fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=False)
# fig.suptitle(
#     f"Estimator Performance ({OPTIMALITY_CRITERION}-Optimality, N_est={N_estimations})",
#     fontsize=16,
# )

# Plot for X parameter on the left column
plot_performance_on_axes(axes[:, 0], "X", idx_x, 0)
axes[0, 0].set_title("Bias (X-parameter)", fontsize=14)
axes[1, 0].set_title("Precision (X-parameter)", fontsize=14)

# Plot for Z parameter on the right column
plot_performance_on_axes(axes[:, 1], "Z", idx_z, 1)
axes[0, 1].set_title("Bias (Z-parameter)", fontsize=14)
axes[1, 1].set_title("Precision (Z-parameter)", fontsize=14)


# Final adjustments
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %% 8) Print J for one of the optimal states
# Choose which J to print, e.g., NuisanceAware or Subdet_XZ
# For example, printing the NuisanceAware FIM:
# J_to_print_key = f"NuisanceAware_{OPTIMALITY_CRITERION}opt_Marginal"
# Or the Subdet_XZ FIM
J_to_print_key = f"Subdet_XZ_{OPTIMALITY_CRITERION}opt"
# J_to_print_key = 'FullFIM_Dopt'

J_to_print = fisher_matrices_for_wavefronts[J_to_print_key]
# J_to_print = np.linalg.inv(J_to_print)
print("\n" + "=" * 50)
print(
    f"Fisher Information Matrix J for the {OPTIMALITY_CRITERION}-Optimal Wavefront: {J_to_print_key}"
)
print("=" * 50)
print(f"{'':>6} | {'x':^12} | {'z':^12} | {'theta':^12}")
print("-" * 50)
for i, row in enumerate(J_to_print):
    print(
        f"{param_names[i]:>6} | {row[0]:>12.3e} | {row[1]:>12.3e} | {row[2]:>12.3e}"
    )
print("=" * 50)

# %% 9) produce nuisance wiggle animation

if False:  # false because it's very slow to generate
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    # --- 1. Animation Setup ---
    # Select the wavefront to visualize.
    wavefront_to_animate = optimized_wavefronts["Marginal_FIM_Optimal"]
    animation_filename = "marginal_d_optimal_loop_animation.gif"
    print(
        f"Preparing animation for Marginal FIM ({OPTIMALITY_CRITERION}-Optimality)..."
    )

    # Create a seamless looping path for the animation
    # The path goes from 0 -> -max -> +max -> 0
    n_sweep_points = len(theta_deviations)
    p1 = np.linspace(0, -max_deviation, n_sweep_points // 2, endpoint=False)
    p2 = np.linspace(
        -max_deviation, max_deviation, n_sweep_points, endpoint=False
    )
    p3 = np.linspace(max_deviation, 0, n_sweep_points // 2, endpoint=True)
    animation_thetas = np.concatenate((p1, p2, p3))
    num_frames = len(animation_thetas)

    # Create a figure for the animation
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))

    # Pre-compute the total field for colormap scaling
    sim.set_input_field_kx(kx_array, wavefront_to_animate)
    sim.compute_total_field()
    vmax = np.max(np.abs(sim.E_total) ** 2)
    sim.reset()

    # --- 2. Animation Function ---
    def animate(i):
        ax_anim.clear()

        # Get the rotation from the new looping animation path
        theta_dev = animation_thetas[i]

        # Update the scatterer positions
        true_movable_dipoles = placedipoles.shift_and_rotate_dipoles(
            movable_dipoles_line, x_true_pos, z_true_pos, theta_dev
        )
        if frozen_dipoles_line is not None and frozen_dipoles_line.size > 0:
            sim.scatterers = np.hstack(
                (frozen_dipoles_line, true_movable_dipoles)
            )
        else:
            sim.scatterers = true_movable_dipoles
        sim.reset()

        # Set the input field and compute the total scattered field
        sim.set_input_field_kx(kx_array, wavefront_to_animate)
        sim.compute_total_field()
        intensity = np.abs(sim.E_total) ** 2

        # Define plot extents in units of wavelength
        extents = [
            sim.z[0] / sim.lambda_0,
            sim.z[-1] / sim.lambda_0,
            sim.x[0] / sim.lambda_0,
            sim.x[-1] / sim.lambda_0,
        ]

        # BUG FIX: Plot the intensity without transposing it (.T is removed)
        im = ax_anim.imshow(
            intensity,
            vmax=vmax,
            extent=extents,
            origin="lower",
            cmap="inferno",
            interpolation="spline16",
            aspect="equal",
        )

        # Overlay the dipole positions
        if frozen_dipoles_line is not None and frozen_dipoles_line.size > 0:
            ax_anim.scatter(
                frozen_dipoles_line[0, :] / sim.lambda_0,
                frozen_dipoles_line[1, :] / sim.lambda_0,
                c="cyan",
                s=25,
                label="Fixed Dipoles",
                marker="o",
            )
        ax_anim.scatter(
            true_movable_dipoles[0, :] / sim.lambda_0,
            true_movable_dipoles[1, :] / sim.lambda_0,
            c="white",
            s=25,
            label="Movable Dipoles",
            marker="x",
        )

        # Set labels and title for the current frame
        ax_anim.set_xlabel("z / λ₀")
        ax_anim.set_ylabel("x / λ₀")
        ax_anim.set_title(
            f"Nuisance Deviation: {np.rad2deg(theta_dev):.2f} degrees"
        )
        ax_anim.legend(loc="upper right")

        return [im]

    # --- 3. Create and Save the Animation ---
    print(
        f"Creating animation ({num_frames} frames)... This will take a moment."
    )
    # Note: blit=True can be faster but is sometimes buggy depending on the Matplotlib backend.
    # If you experience issues, setting blit=False is a safe alternative.
    anim = FuncAnimation(
        fig_anim, animate, frames=num_frames, interval=50, blit=False
    )

    # Save the animation as a GIF
    writer = PillowWriter(fps=15)
    anim.save(animation_filename, writer=writer)
    plt.close(fig_anim)

    print(f"Animation successfully saved as '{animation_filename}'")


# %% 10) Data Preparation for XZ Deviation Impact (Optimal vs. Random with Fair Estimators)

# Settings for the XZ deviation analysis
N_xz_deviations = 51
max_xz_deviation = 0.12
x_dev_range = np.linspace(-max_xz_deviation, max_xz_deviation, N_xz_deviations)
z_dev_range = np.linspace(-max_xz_deviation, max_xz_deviation, N_xz_deviations)

# --- 1. Components for Nuisance-Aware Optimized Wavefront ---
wavefront_key_optimal = f"NuisanceAware_{OPTIMALITY_CRITERION}opt_Marginal"
wf_optimal = optimized_wavefronts[wavefront_key_optimal]
E0_estimator_optimal = E0_nuisance_aware
dE_vectors_estimator_optimal = dE_nuisance_aware
J_inv_estimator_optimal = J_inv_nuisance_aware
absolute_bias_grid_optimal = np.zeros((N_xz_deviations, N_xz_deviations))

# --- 2. Components for Flux-Normalized Random Wavefront ---
np.random.seed(42)  # for reproducibility
wf_random = np.random.randn(number_of_input_channels) + 1j * np.random.randn(
    number_of_input_channels
)
wf_random = optimizer.normalize_wavefront(wf_random, photon_budget)

# **FAIR COMPARISON STEP**: Calibrate an estimator specifically for the random wavefront
print("\nCalibrating a separate, fair estimator for the random wavefront...")
# a) Calculate the nominal field E0 for the random wavefront
sim.scatterers = (
    np.hstack((frozen_dipoles_line, movable_dipoles_line))
    if frozen_dipoles_line is not None
    else movable_dipoles_line
)
sim.reset()
sim.set_input_field_kx(kx_array, wf_random)
E0_estimator_random = sim.compute_zL_fields_direct()
# b) Calculate the derivative vectors for the random wavefront
dE_vectors_estimator_random = [
    dS_tensor[i] @ wf_random for i in range(len(param_names))
]
# c) Calculate the Fisher Matrix and its inverse for the random wavefront
J_matrix_random = optimizer.J_fisher_info_matrix(wf_random)
J_inv_estimator_random = np.linalg.inv(J_matrix_random)
absolute_bias_grid_random = np.zeros((N_xz_deviations, N_xz_deviations))


print(
    f"\nStarting XZ deviation simulation for two wavefronts with fair estimators..."
)
for i, x_dev in enumerate(x_dev_range):
    for j, z_dev in enumerate(z_dev_range):
        print(
            f"\rSimulating XZ deviation: x={x_dev:.3f}, z={z_dev:.3f} ({i*N_xz_deviations + j + 1}/{N_xz_deviations**2})",
            end="",
        )

        current_movable_dipoles = placedipoles.shift_and_rotate_dipoles(
            movable_dipoles_line, x_dev, z_dev, 0.0
        )
        sim.scatterers = (
            np.hstack((frozen_dipoles_line, current_movable_dipoles))
            if frozen_dipoles_line is not None
            else current_movable_dipoles
        )

        # --- Process Optimal Wavefront ---
        sim.reset()
        sim.set_input_field_kx(kx_array, wf_optimal)
        E_true_optimal = sim.compute_zL_fields_direct()
        Z_noisy_optimal = measure_heterodyne(
            E_true_optimal,
            total_photons,
            nsamples=N_estimations,
            seed=(i * N_xz_deviations + j),
        )
        estimates_optimal = np.array(
            [
                best_estimate_multi(
                    Z,
                    E0_estimator_optimal,
                    dE_vectors_estimator_optimal,
                    J_inv_estimator_optimal,
                )
                for Z in Z_noisy_optimal
            ]
        )
        absolute_bias_grid_optimal[i, j] = np.abs(
            np.mean(estimates_optimal[:, param_names.index("shiftX")]) - x_dev
        )

        # --- Process Random Wavefront ---
        sim.reset()
        sim.set_input_field_kx(kx_array, wf_random)
        E_true_random = sim.compute_zL_fields_direct()
        Z_noisy_random = measure_heterodyne(
            E_true_random,
            total_photons,
            nsamples=N_estimations,
            seed=(i * N_xz_deviations + j) + N_xz_deviations**2,
        )
        # Use the estimator calibrated for the random wavefront
        estimates_random = np.array(
            [
                best_estimate_multi(
                    Z,
                    E0_estimator_random,
                    dE_vectors_estimator_random,
                    J_inv_estimator_random,
                )
                for Z in Z_noisy_random
            ]
        )
        absolute_bias_grid_random[i, j] = np.abs(
            np.mean(estimates_random[:, param_names.index("shiftX")]) - x_dev
        )

print("\nXZ deviation simulation for both wavefronts finished.")

# %% 11) Combined Contour Plot of Absolute Estimation Bias

# Prepare meshgrid and data for plotting
X_dev_grid, Z_dev_grid = np.meshgrid(x_dev_range, z_dev_range)
plot_data_optimal = absolute_bias_grid_optimal.T
plot_data_random = absolute_bias_grid_random.T

# Calculate the difference in bias (performance gain)
performance_difference = plot_data_random - plot_data_optimal

plt.figure(figsize=(12, 10))

# Use pcolormesh to show the performance difference
vmax_diff = 0.005
pcm = plt.pcolormesh(
    X_dev_grid,
    Z_dev_grid,
    performance_difference,
    cmap="coolwarm",
    shading="gouraud",
    vmin=-vmax_diff,
    vmax=vmax_diff,
)
cbar = plt.colorbar(pcm, extend="both")
cbar.set_label(
    r"Bias Reduction: $\epsilon_{random} - \epsilon_{optimal}$ [$\lambda$]",
    fontsize=14,
)

# Define contour levels
contour_levels = [0.001, 0.01]
fmt_dict = {level: f"{level:.3f} $\\lambda$" for level in contour_levels}

# Overlay contours for the OPTIMAL wavefront (solid white lines)
cs_optimal = plt.contour(
    X_dev_grid,
    Z_dev_grid,
    plot_data_optimal,
    levels=contour_levels,
    colors="white",
    linewidths=2.5,
    linestyles="solid",
)
plt.clabel(cs_optimal, inline=True, fontsize=12, fmt=fmt_dict, colors="white")

# Overlay contours for the RANDOM wavefront (dashed black lines)
cs_random = plt.contour(
    X_dev_grid,
    Z_dev_grid,
    plot_data_random,
    levels=contour_levels,
    colors="black",
    linewidths=2.5,
    linestyles="dashed",
)
plt.clabel(cs_random, inline=True, fontsize=12, fmt=fmt_dict, colors="black")

# Add plot details and labels
plt.xlabel(
    r"True Deviation in x-position ($\Delta x$) [$\lambda$]", fontsize=14
)
plt.ylabel(
    r"True Deviation in z-position ($\Delta z$) [$\lambda$]", fontsize=14
)
plt.title(r"Absolute Estimation Bias ($\epsilon$) for $\hat{x}$", fontsize=16)
plt.axhline(0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
plt.axvline(0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
plt.scatter(
    [0],
    [0],
    marker="o",
    color="magenta",
    s=80,
    zorder=5,
    edgecolors="black",
    label="Nominal (0,0)",
)

# Create custom legend for the contour styles
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        color="white",
        lw=2.5,
        linestyle="-",
        label=f'Optimal WF ({wavefront_key_optimal.replace("_", " ")})',
    ),
    Line2D([0], [0], color="black", lw=2.5, linestyle="--", label="Random WF"),
    plt.scatter(
        [],
        [],
        marker="o",
        color="magenta",
        s=80,
        edgecolors="black",
        label="Nominal (0,0)",
    ),
]
plt.legend(handles=legend_elements, loc="upper right", fontsize=12)

plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.show()
