import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator


# %% -1) define functions
def measure_heterodyne(E, total_photons, nsamples=None):
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
    if nsamples is None:
        field_samples = stats.norm.rvs(loc=E_real, scale=noise_std_dev)
    else:
        field_samples = stats.norm.rvs(
            loc=E_real,
            scale=noise_std_dev,
            size=(nsamples, *E_real.shape),
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
OPTIMALITY_CRITERION = "D"  # Can be 'D' or 'E', or 'A'

verbose_plotting = True
max_angle = 80
number_of_input_channels = 100
number_of_output_channels = 300
param_names = ["shiftX", "shiftZ", "rotation"]
delta_param = 1e-3
photon_budget = 1.0
multi_run_count = 10
single_run_iterations = 3000

# %% 1) define dipole configurations
# movable_dipoles_line = placedipoles.place_line_of_dipoles(
#     center=(5.0, 0.0), number=7
# )
# frozen_dipoles_line = placedipoles.place_line_of_dipoles(
#     center=(3.0, 0.0), number=7
# )

### Quasi-random double line
movable_dipoles_line = placedipoles.place_quasi_random_line_of_dipoles(
    center=(5.2, 0.0), number=6, small_dist=0.786, start_index=5
)
frozen_dipoles_line = placedipoles.place_quasi_random_line_of_dipoles(
    center=(3.0, 0.0), number=6, small_dist=0.786, start_index=22
)

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

# Add the single-parameter MIS for X as a baseline
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

# Conditional optimization based on the toggle
if OPTIMALITY_CRITERION == "D":
    print(
        f"\n--- {OPTIMALITY_CRITERION}-Optimality for Nuisance-Aware Marginal FIM (Maximize det(J_marginal_XZ|theta)) ---"
    )
    nuisance_param_index_for_marginal = param_names.index(
        "rotation"
    )  # Index for theta
    loss_fn_nuisance_aware = (
        lambda J_full: lossfunctions.determinant_of_marginal_J(
            optimizer._compute_marginal_fisher_info_matrix(
                J_full, nuisance_param_index_for_marginal
            )
        )
    )
    nuisance_aware_opt_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=loss_fn_nuisance_aware,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=True,
        photon_budget=photon_budget,
        base_seed=42,
    )
    optimized_wavefronts["NuisanceAware_Dopt_Marginal"] = (
        nuisance_aware_opt_results["bestvec"]
    )
    fisher_matrices_for_wavefronts["NuisanceAware_Dopt_Marginal"] = (
        optimizer.J_fisher_info_matrix(nuisance_aware_opt_results["bestvec"])
    )

    print(
        f"\n--- {OPTIMALITY_CRITERION}-Optimality for Sub-determinant FIM (Maximize det(J_XZ)) ---"
    )
    # Indices for X (0) and Z (1)
    subdet_indices_xz = [
        param_names.index("shiftX"),
        param_names.index("shiftZ"),
    ]
    loss_fn_subdet_xz = (
        lambda J_full: lossfunctions.determinant_of_J_submatrix(
            J_full, subdet_indices_xz
        )
    )
    subdet_xz_opt_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=loss_fn_subdet_xz,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=True,
        photon_budget=photon_budget,
        base_seed=123,  # Different seed for variety
    )
    optimized_wavefronts["Subdet_XZ_Dopt"] = subdet_xz_opt_results["bestvec"]
    fisher_matrices_for_wavefronts["Subdet_XZ_Dopt"] = (
        optimizer.J_fisher_info_matrix(subdet_xz_opt_results["bestvec"])
    )

elif OPTIMALITY_CRITERION == "A":
    print(
        f"\n--- {OPTIMALITY_CRITERION}-Optimality for Nuisance-Aware Marginal FIM (Minimize Tr((J_marginal_XZ|theta)^-1)) ---"
    )
    nuisance_param_index_for_marginal = param_names.index("rotation")
    loss_fn_nuisance_aware = lambda J_full: lossfunctions.minimize_trace_of_marginal_J_inv(
        optimizer._compute_marginal_fisher_info_matrix(
            J_full, nuisance_param_index_for_marginal
        )
    )
    nuisance_aware_opt_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=loss_fn_nuisance_aware,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=False,
        photon_budget=photon_budget,
        base_seed=42,
    )
    optimized_wavefronts["NuisanceAware_Aopt_Marginal"] = (
        nuisance_aware_opt_results["bestvec"]
    )
    fisher_matrices_for_wavefronts["NuisanceAware_Aopt_Marginal"] = (
        optimizer.J_fisher_info_matrix(nuisance_aware_opt_results["bestvec"])
    )

    print(
        f"\n--- {OPTIMALITY_CRITERION}-Optimality for Sub-determinant FIM (Minimize Tr((J_XZ)^-1)) ---"
    )
    subdet_indices_xz = [
        param_names.index("shiftX"),
        param_names.index("shiftZ"),
    ]
    loss_fn_subdet_xz = (
        lambda J_full: lossfunctions.minimize_trace_of_J_inv_submatrix(
            J_full, subdet_indices_xz
        )
    )
    subdet_xz_opt_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=loss_fn_subdet_xz,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=False,
        photon_budget=photon_budget,
        base_seed=123,
    )
    optimized_wavefronts["Subdet_XZ_Aopt"] = subdet_xz_opt_results["bestvec"]
    fisher_matrices_for_wavefronts["Subdet_XZ_Aopt"] = (
        optimizer.J_fisher_info_matrix(subdet_xz_opt_results["bestvec"])
    )


elif OPTIMALITY_CRITERION == "E":
    print(
        f"\n--- {OPTIMALITY_CRITERION}-Optimality for Nuisance-Aware Marginal FIM (Maximize min_eig(J_marginal_XZ|theta)) ---"
    )
    nuisance_param_index_for_marginal = param_names.index("rotation")
    loss_fn_nuisance_aware = (
        lambda J_full: lossfunctions.maximize_min_eig_marginal_J(
            optimizer._compute_marginal_fisher_info_matrix(
                J_full, nuisance_param_index_for_marginal
            )
        )
    )
    nuisance_aware_opt_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=loss_fn_nuisance_aware,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=True,
        photon_budget=photon_budget,
        base_seed=42,
    )
    optimized_wavefronts["NuisanceAware_Eopt_Marginal"] = (
        nuisance_aware_opt_results["bestvec"]
    )
    fisher_matrices_for_wavefronts["NuisanceAware_Eopt_Marginal"] = (
        optimizer.J_fisher_info_matrix(nuisance_aware_opt_results["bestvec"])
    )

    print(
        f"\n--- {OPTIMALITY_CRITERION}-Optimality for Sub-determinant FIM (Maximize min_eig(J_XZ)) ---"
    )
    subdet_indices_xz = [
        param_names.index("shiftX"),
        param_names.index("shiftZ"),
    ]
    loss_fn_subdet_xz = (
        lambda J_full: lossfunctions.maximize_min_eig_J_submatrix(
            J_full, subdet_indices_xz
        )
    )
    subdet_xz_opt_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=loss_fn_subdet_xz,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=True,
        photon_budget=photon_budget,
        base_seed=123,
    )
    optimized_wavefronts["Subdet_XZ_Eopt"] = subdet_xz_opt_results["bestvec"]
    fisher_matrices_for_wavefronts["Subdet_XZ_Eopt"] = (
        optimizer.J_fisher_info_matrix(subdet_xz_opt_results["bestvec"])
    )

else:
    raise ValueError("OPTIMALITY_CRITERION must be 'D', 'A', or 'E'")

# %% 6) Run simulation with the CORRECT multi-parameter estimator
total_photons = 1e6
N_estimations = 10000
theta0_rad = 0.0
x_true_pos = 0.0
z_true_pos = 0.0
N_deviations = 51
max_deviation = 0.02 * 5
theta_deviations = np.linspace(-max_deviation, max_deviation, N_deviations)

all_estimates_nuisance_aware = []
all_estimates_subdet_xz = []
all_estimates_mis_x = []

print("\nPre-calculating estimator components...")
sim = CoupledDipoleSimulator(**sim_kwargs)
nominal_movable_dipoles = placedipoles.shift_and_rotate_dipoles(
    movable_dipoles_line, 0, 0, theta0_rad
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

# Components for Nuisance-Aware (Marginal FIM) optimal wavefront (3-parameter estimator)
print(f"\nSetting up components for: {nuisance_aware_wf_key}")
wf_nuisance_aware = optimized_wavefronts[nuisance_aware_wf_key]
dE_vectors_nuisance_aware = [
    dS_tensor[i] @ wf_nuisance_aware for i in range(len(param_names))
]
sim.set_input_field_kx(kx_array, wf_nuisance_aware)
E0_nuisance_aware = sim.compute_zL_fields_direct()
J_nuisance_aware = fisher_matrices_for_wavefronts[nuisance_aware_wf_key]
J_inv_nuisance_aware = np.linalg.inv(J_nuisance_aware) # 3x3 inverse

# Components for Sub-determinant XZ optimal wavefront (2-parameter estimator for X, Z)
print(f"Setting up components for: {subdet_xz_wf_key}")
wf_subdet_xz = optimized_wavefronts[subdet_xz_wf_key]
dE_vectors_subdet_xz_full = [ # For calculating E0 and for naive estimator if needed
    dS_tensor[i] @ wf_subdet_xz for i in range(len(param_names))
]
dE_vectors_subdet_xz_for_estimation = [ # Only dEx and dEz for 2-param estimator
    dE_vectors_subdet_xz_full[param_names.index("shiftX")],
    dE_vectors_subdet_xz_full[param_names.index("shiftZ")],
]
sim.set_input_field_kx(kx_array, wf_subdet_xz)
E0_subdet_xz = sim.compute_zL_fields_direct()
J_subdet_xz_full = fisher_matrices_for_wavefronts[subdet_xz_wf_key] # Full 3x3 FIM
# Extract 2x2 FIM for X and Z for the 2-parameter estimator
idx_x = param_names.index("shiftX")
idx_z = param_names.index("shiftZ")
J_subdet_xz_2x2 = J_subdet_xz_full[np.ix_([idx_x, idx_z], [idx_x, idx_z])]
J_inv_subdet_xz_2x2 = np.linalg.inv(J_subdet_xz_2x2) # 2x2 inverse for 2-param estimator

# Components for MIS_x wavefront (1-parameter estimator for X)
print("Setting up components for: MIS_x")
wf_mis_x = optimized_wavefronts["MIS_x"]
dE_vector_x_for_mis_x = dS_tensor[param_names.index("shiftX")] @ wf_mis_x # Only dEx for 1-param estimator
dE_vectors_mis_x_full = [ # All dE vectors for MIS_x, e.g. for naive estimator plots
    dS_tensor[i] @ wf_mis_x for i in range(len(param_names))
]
sim.set_input_field_kx(kx_array, wf_mis_x)
E0_mis_x = sim.compute_zL_fields_direct()
J_mis_x_full = fisher_matrices_for_wavefronts["MIS_x"] # Full 3x3 FIM
J_inv_mis_x_full = np.linalg.inv(J_mis_x_full) # Full 3x3 inverse for CRLB (param X from 3-param context)


print("\nStarting estimation loop...")
for i, theta_dev in enumerate(theta_deviations):
    print(
        f"\rSimulating deviation {i+1}/{len(theta_deviations)}: {theta_dev:.4f} rad",
        end="",
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

    # Nuisance-Aware (e.g., Marginal D-Optimal) Estimation (3-parameter estimator)
    sim.set_input_field_kx(kx_array, wf_nuisance_aware)
    E_true_nuisance_aware = sim.compute_zL_fields_direct()
    Z_noisy_samples_nuisance_aware = measure_heterodyne(
        E_true_nuisance_aware, total_photons, nsamples=N_estimations
    )
    estimates_nuisance_aware = np.array(
        [
            best_estimate_multi(
                Z, E0_nuisance_aware, dE_vectors_nuisance_aware, J_inv_nuisance_aware
            ) # estimates all 3 params
            for Z in Z_noisy_samples_nuisance_aware
        ]
    )
    all_estimates_nuisance_aware.append(estimates_nuisance_aware)

    # Sub-determinant XZ Optimal Estimation (2-parameter estimator for X, Z)
    sim.set_input_field_kx(kx_array, wf_subdet_xz)
    E_true_subdet_xz = sim.compute_zL_fields_direct()
    Z_noisy_samples_subdet_xz = measure_heterodyne(
        E_true_subdet_xz, total_photons, nsamples=N_estimations
    )
    estimates_subdet_xz = np.array(
        [
            best_estimate_multi(
                Z, E0_subdet_xz, dE_vectors_subdet_xz_for_estimation, J_inv_subdet_xz_2x2
            ) # estimates 2 params (X,Z)
            for Z in Z_noisy_samples_subdet_xz
        ]
    )
    all_estimates_subdet_xz.append(estimates_subdet_xz)

    # MIS_x Estimation (1-parameter estimator for X)
    sim.set_input_field_kx(kx_array, wf_mis_x)
    E_true_mis_x = sim.compute_zL_fields_direct()
    Z_noisy_samples_mis_x = measure_heterodyne(
        E_true_mis_x, total_photons, nsamples=N_estimations
    )
    estimates_mis_x = np.array(
        [
            best_estimate(Z, dE_vector_x_for_mis_x, E0_mis_x) # estimates 1 param (X)
            for Z in Z_noisy_samples_mis_x
        ]
    )
    all_estimates_mis_x.append(estimates_mis_x)

print("\nSimulation loop finished.")

# %% 7) Plotting Bias and Precision for X-parameter
# Note: idx_x is param_names.index("shiftX"), defined in section 6 component setup.

bias_x_nuisance_aware = [
    np.mean(estimates[:, idx_x]) - x_true_pos # NuisanceAware estimates 3 params
    for estimates in all_estimates_nuisance_aware
]
bias_x_subdet_xz = [
    np.mean(estimates[:, 0]) - x_true_pos # Subdet_XZ estimates 2 params (X,Z), X is at index 0
    for estimates in all_estimates_subdet_xz
]
bias_x_mis_x = [
    np.mean(estimates) - x_true_pos # MIS_x estimates 1 param (X)
    for estimates in all_estimates_mis_x
]

std_x_nuisance_aware = [
    np.std(estimates[:, idx_x])
    for estimates in all_estimates_nuisance_aware
]
std_x_subdet_xz = [
    np.std(estimates[:, 0])
    for estimates in all_estimates_subdet_xz
]
std_x_mis_x = [
    np.std(estimates)
    for estimates in all_estimates_mis_x
]

crlb_prefactor = 1 / np.sqrt(2) # Heterodyne detection specific
# CRLB for X from 3-param estimator using NuisanceAware wavefront (J_inv_nuisance_aware is 3x3)
crlb_std_x_nuisance_aware = crlb_prefactor * np.sqrt(J_inv_nuisance_aware[idx_x, idx_x])
# CRLB for X from 2-param estimator using Subdet_XZ wavefront (J_inv_subdet_xz_2x2 is 2x2 for X,Z)
crlb_std_x_subdet_xz = crlb_prefactor * np.sqrt(J_inv_subdet_xz_2x2[0, 0]) # X is param 0 in the 2x2 matrix
# CRLB for X from 1-param estimator using MIS_x wavefront (J_mis_x_full is 3x3)
# For a single parameter estimator, CRLB_i = 1/J_ii
crlb_std_x_mis_x = crlb_prefactor * np.sqrt(1 / J_mis_x_full[idx_x, idx_x])


fig_x, (ax1_x, ax2_x) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig_x.suptitle(
    f"Estimator Performance for X-Parameter ({OPTIMALITY_CRITERION}-Optimality, N_est={N_estimations})",
    fontsize=16,
)

# Bias Plot for X
ax1_x.plot(
    theta_deviations,
    bias_x_nuisance_aware,
    "o-",
    label=f"Nuisance-Aware {OPTIMALITY_CRITERION}-Opt (3-param est.)", # Was Marginal
    color="#ff7f0e", # Orange
    markersize=8,
    lw=2,
)
ax1_x.plot(
    theta_deviations,
    bias_x_subdet_xz,
    "s-",
    label=f"Subdet XZ {OPTIMALITY_CRITERION}-Opt (2-param est.)", # Was Full
    color="#1f77b4", # Blue
    markersize=8,
    lw=2,
)
ax1_x.plot(
    theta_deviations,
    bias_x_mis_x,
    "^-",
    label="MIS_x (1-param est.)",
    color="#2ca02c", # Green
    markersize=8,
    lw=2,
)
ax1_x.axhline(0, color="k", linestyle="--", label="Unbiased")
ax1_x.set_ylabel("Estimation Bias of $\\hat{x}$", fontsize=14)
ax1_x.legend(fontsize=11)
ax1_x.grid(True, which="both", linestyle=":", alpha=0.6)
ax1_x.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax1_x.set_title("Accuracy (Bias)", fontsize=14)

# Standard Deviation Plot for X
ax2_x.plot(
    theta_deviations,
    std_x_nuisance_aware,
    "o-",
    label=f"Nuisance-Aware {OPTIMALITY_CRITERION}-Opt (3-param est.)",
    color="#ff7f0e",
    markersize=8,
    lw=2,
)
ax2_x.plot(
    theta_deviations,
    std_x_subdet_xz,
    "s-",
    label=f"Subdet XZ {OPTIMALITY_CRITERION}-Opt (2-param est.)",
    color="#1f77b4",
    markersize=8,
    lw=2,
)
ax2_x.plot(
    theta_deviations,
    std_x_mis_x,
    "^-",
    label="MIS_x (1-param est.)",
    color="#2ca02c",
    markersize=8,
    lw=2,
)
ax2_x.axhline(
    y=crlb_std_x_nuisance_aware / np.sqrt(total_photons),
    color="#ff7f0e",
    linestyle="--",
    label=f"CRLB Nuisance-Aware {OPTIMALITY_CRITERION}-Opt",
)
ax2_x.axhline(
    y=crlb_std_x_subdet_xz / np.sqrt(total_photons),
    color="#1f77b4",
    linestyle="--",
    label=f"CRLB Subdet XZ {OPTIMALITY_CRITERION}-Opt",
)
ax2_x.axhline(
    y=crlb_std_x_mis_x / np.sqrt(total_photons),
    color="#2ca02c",
    linestyle="--",
    label="CRLB MIS_x (1-param)",
)
ax2_x.set_xlabel(
    "Nuisance Parameter Deviation ($\\theta_{true} - \\theta_0$) [rad]",
    fontsize=14,
)
ax2_x.set_ylabel("Estimation Error (Std. Dev. of $\\hat{x}$)", fontsize=14)
ax2_x.legend(fontsize=12)
ax2_x.grid(True, which="both", linestyle=":", alpha=0.6)
ax2_x.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax2_x.set_title("Precision (Standard Deviation)", fontsize=14)
ax2_x.set_ylim(bottom=0)
plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.savefig('out.png', dpi=300)
plt.show()

# %% 8) Plotting Bias and Precision for Z-parameter
# Note: idx_z is param_names.index("shiftZ"), defined in section 6 component setup.

bias_z_nuisance_aware = [
    np.mean(estimates[:, idx_z]) - z_true_pos # NuisanceAware estimates 3 params
    for estimates in all_estimates_nuisance_aware
]
bias_z_subdet_xz = [
    np.mean(estimates[:, 1]) - z_true_pos # Subdet_XZ estimates 2 params (X,Z), Z is at index 1
    for estimates in all_estimates_subdet_xz
]
# MIS_x is not plotted for Z as it's a 1-parameter estimator for X

std_z_nuisance_aware = [
    np.std(estimates[:, idx_z])
    for estimates in all_estimates_nuisance_aware
]
std_z_subdet_xz = [
    np.std(estimates[:, 1])
    for estimates in all_estimates_subdet_xz
]
# MIS_x is not plotted for Z

# CRLB for Z from 3-param estimator using NuisanceAware wavefront (J_inv_nuisance_aware is 3x3)
crlb_std_z_nuisance_aware = crlb_prefactor * np.sqrt(J_inv_nuisance_aware[idx_z, idx_z])
# CRLB for Z from 2-param estimator using Subdet_XZ wavefront (J_inv_subdet_xz_2x2 is 2x2 for X,Z)
crlb_std_z_subdet_xz = crlb_prefactor * np.sqrt(J_inv_subdet_xz_2x2[1, 1]) # Z is param 1 in the 2x2 matrix
# MIS_x is not plotted for Z

fig_z, (ax1_z, ax2_z) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig_z.suptitle(
    f"Estimator Performance for Z-Parameter ({OPTIMALITY_CRITERION}-Optimality, N_est={N_estimations})",
    fontsize=16,
)

# Bias Plot for Z
ax1_z.plot(
    theta_deviations,
    bias_z_nuisance_aware,
    "o-",
    label=f"Nuisance-Aware {OPTIMALITY_CRITERION}-Opt (3-param est.)",
    color="#ff7f0e", # Orange
    markersize=8,
    lw=2,
)
ax1_z.plot(
    theta_deviations,
    bias_z_subdet_xz,
    "s-",
    label=f"Subdet XZ {OPTIMALITY_CRITERION}-Opt (2-param est.)",
    color="#1f77b4", # Blue
    markersize=8,
    lw=2,
)
ax1_z.axhline(0, color="k", linestyle="--", label="Unbiased")
ax1_z.set_ylabel("Estimation Bias of $\\hat{z}$", fontsize=14)
ax1_z.legend(fontsize=11)
ax1_z.grid(True, which="both", linestyle=":", alpha=0.6)
ax1_z.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax1_z.set_title("Accuracy (Bias)", fontsize=14)

# Standard Deviation Plot for Z
ax2_z.plot(
    theta_deviations,
    std_z_nuisance_aware,
    "o-",
    label=f"Nuisance-Aware {OPTIMALITY_CRITERION}-Opt (3-param est.)",
    color="#ff7f0e",
    markersize=8,
    lw=2,
)
ax2_z.plot(
    theta_deviations,
    std_z_subdet_xz,
    "s-",
    label=f"Subdet XZ {OPTIMALITY_CRITERION}-Opt (2-param est.)",
    color="#1f77b4",
    markersize=8,
    lw=2,
)
ax2_z.axhline(
    y=crlb_std_z_nuisance_aware / np.sqrt(total_photons),
    color="#ff7f0e",
    linestyle="--",
    label=f"CRLB Nuisance-Aware {OPTIMALITY_CRITERION}-Opt",
)
ax2_z.axhline(
    y=crlb_std_z_subdet_xz / np.sqrt(total_photons),
    color="#1f77b4",
    linestyle="--",
    label=f"CRLB Subdet XZ {OPTIMALITY_CRITERION}-Opt",
)
ax2_z.set_xlabel(
    "Nuisance Parameter Deviation ($\\theta_{true} - \\theta_0$) [rad]",
    fontsize=14,
)
ax2_z.set_ylabel("Estimation Error (Std. Dev. of $\\hat{z}$)", fontsize=14)
ax2_z.legend(fontsize=12)
ax2_z.grid(True, which="both", linestyle=":", alpha=0.6)
ax2_z.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax2_z.set_title("Precision (Standard Deviation)", fontsize=14)
ax2_z.set_ylim(bottom=0)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("out.png", dpi=300)
plt.show()


# %% 9) Revisiting the Naive Estimator to Visualize its Bias for X and Z
# This section applies a naive single-parameter estimator to each wavefront
# to see how much bias is introduced by nuisance parameter mismatch if the
# estimator is not aware of other parameters.

naive_bias_x_nuisance_aware = []
naive_bias_x_subdet_xz = []
naive_bias_x_mis_x = []

naive_bias_z_nuisance_aware = []
naive_bias_z_subdet_xz = []
naive_bias_z_mis_x = []


# dE vectors for X, for each primary wavefront type (using the _full versions from sec 6)
dEx_nuisance_aware = dE_vectors_nuisance_aware[idx_x]
dEx_subdet_xz = dE_vectors_subdet_xz_full[idx_x]
dEx_mis_x = dE_vectors_mis_x_full[idx_x] # This is same as dE_vector_x_for_mis_x

# dE vectors for Z, for each primary wavefront type
dEz_nuisance_aware = dE_vectors_nuisance_aware[idx_z]
dEz_subdet_xz = dE_vectors_subdet_xz_full[idx_z]
dEz_mis_x = dE_vectors_mis_x_full[idx_z]


print(
    "\nApplying naive single-parameter estimator to noise-free data for X and Z..."
)
for i, theta_dev in enumerate(theta_deviations):
    print(
        f"\rSimulating deviation {i+1}/{len(theta_deviations)}: {theta_dev:.4f} rad",
        end="",
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

    # Nuisance-Aware Wavefront with Naive Estimator
    sim.set_input_field_kx(kx_array, wf_nuisance_aware)
    E_true_nuisance_aware = sim.compute_zL_fields_direct()
    naive_bias_x_nuisance_aware.append(
        best_estimate(E_true_nuisance_aware, dEx_nuisance_aware, E0_nuisance_aware) - x_true_pos
    )
    naive_bias_z_nuisance_aware.append(
        best_estimate(E_true_nuisance_aware, dEz_nuisance_aware, E0_nuisance_aware) - z_true_pos
    )

    # Sub-determinant XZ Wavefront with Naive Estimator
    sim.set_input_field_kx(kx_array, wf_subdet_xz)
    E_true_subdet_xz = sim.compute_zL_fields_direct()
    naive_bias_x_subdet_xz.append(
        best_estimate(E_true_subdet_xz, dEx_subdet_xz, E0_subdet_xz) - x_true_pos
    )
    naive_bias_z_subdet_xz.append(
        best_estimate(E_true_subdet_xz, dEz_subdet_xz, E0_subdet_xz) - z_true_pos
    )

    # MIS_x Wavefront with Naive Estimator
    sim.set_input_field_kx(kx_array, wf_mis_x)
    E_true_mis_x = sim.compute_zL_fields_direct()
    naive_bias_x_mis_x.append(
        best_estimate(E_true_mis_x, dEx_mis_x, E0_mis_x) - x_true_pos
    )
    naive_bias_z_mis_x.append(
        best_estimate(E_true_mis_x, dEz_mis_x, E0_mis_x) - z_true_pos
    )

print("\nBias calculation for naive estimators finished.")

fig_naive, (ax_x_naive, ax_z_naive) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig_naive.suptitle(
    f"Bias of Naive Single-Parameter Estimator vs. Nuisance Mismatch ({OPTIMALITY_CRITERION}-Optimality Wavefronts)",
    fontsize=16,
)
plt.subplots_adjust(wspace=0.05) # Adjusted for shared Y

# --- Left Plot: Bias of X-hat (Naive) ---
ax_x_naive.plot(
    theta_deviations,
    naive_bias_x_nuisance_aware,
    "o-",
    label=f"Nuisance-Aware {OPTIMALITY_CRITERION}-Opt WF",
    color="#ff7f0e",
    markersize=8,
)
ax_x_naive.plot(
    theta_deviations,
    naive_bias_x_subdet_xz,
    "s-",
    label=f"Subdet XZ {OPTIMALITY_CRITERION}-Opt WF",
    color="#1f77b4",
    markersize=8,
)
ax_x_naive.plot(
    theta_deviations,
    naive_bias_x_mis_x,
    "^-",
    label="MIS_x WF",
    color="#2ca02c",
    markersize=8,
)
ax_x_naive.axhline(0, color="k", linestyle="--")
ax_x_naive.grid(True, linestyle=":")
ax_x_naive.set_title("Naive $\\hat{x}$ Bias", fontsize=14)
ax_x_naive.set_xlabel(
    "Nuisance Deviation ($\\theta_{true} - \\theta_0$) [rad]", fontsize=12
)
ax_x_naive.set_ylabel("Systematic Estimation Bias of $\\hat{x}$ (Naive Estimator)", fontsize=12)
ax_x_naive.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax_x_naive.legend(fontsize=10)

# --- Right Plot: Bias of Z-hat (Naive) ---
ax_z_naive.plot(
    theta_deviations,
    naive_bias_z_nuisance_aware,
    "o-",
    label=f"Nuisance-Aware {OPTIMALITY_CRITERION}-Opt WF", # Legend might be redundant if shared Y and clear title
    color="#ff7f0e",
    markersize=8,
)
ax_z_naive.plot(
    theta_deviations,
    naive_bias_z_subdet_xz,
    "s-",
    label=f"Subdet XZ {OPTIMALITY_CRITERION}-Opt WF",
    color="#1f77b4",
    markersize=8,
)
ax_z_naive.plot(
    theta_deviations,
    naive_bias_z_mis_x,
    "^-",
    label="MIS_x WF",
    color="#2ca02c",
    markersize=8,
)
ax_z_naive.axhline(0, color="k", linestyle="--")
ax_z_naive.grid(True, linestyle=":")
ax_z_naive.set_title("Naive $\\hat{z}$ Bias", fontsize=14)
ax_z_naive.set_xlabel(
    "Nuisance Deviation ($\\theta_{true} - \\theta_0$) [rad]", fontsize=12
)
# ax_z_naive.ticklabel_format(style="sci", axis="y", scilimits=(0, 0)) # Already on left due to sharey
# ax_z_naive.yaxis.tick_right() # Not needed with sharey
# ax_z_naive.yaxis.set_label_position("right") # Not needed
# ax_z_naive.set_ylabel("Systematic Estimation Bias of $\\hat{z}$ (Naive Estimator)", fontsize=12) # Redundant with shared Y
ax_z_naive.legend(fontsize=10)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('naive_bias_comparison.png', dpi=300)

plt.show()

# %% Print J for one of the optimal states
# Choose which J to print, e.g., NuisanceAware or Subdet_XZ
# For example, printing the NuisanceAware FIM:
# J_to_print_key = f"NuisanceAware_{OPTIMALITY_CRITERION}opt_Marginal"
# Or the Subdet_XZ FIM
J_to_print_key = f"Subdet_XZ_{OPTIMALITY_CRITERION}opt"

J_to_print = fisher_matrices_for_wavefronts[J_to_print_key]
print("\\n" + "=" * 50)
print(
    f"Fisher Information Matrix J for the {OPTIMALITY_CRITERION}-Optimal Wavefront:"
)
print("=" * 50)
print(f"{'':>6} | {'x':^12} | {'z':^12} | {'theta':^12}")
print("-" * 50)
for i, row in enumerate(J_to_print):
    print(
        f"{param_names[i]:>6} | {row[0]:>12.3e} | {row[1]:>12.3e} | {row[2]:>12.3e}"
    )
print("=" * 50)

# %% produce nuisance wiggle animation

if False:
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
