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
    print("\n--- D-Optimality (Maximize det(J)) ---")
    opt_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=lossfunctions.determinant_of_J,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=True,
        photon_budget=photon_budget,
        base_seed=42,
    )
    optimized_wavefronts["Full_FIM_Optimal"] = opt_results["bestvec"]
    fisher_matrices_for_wavefronts["Full_FIM_Optimal"] = (
        optimizer.J_fisher_info_matrix(opt_results["bestvec"])
    )

    print("\n--- D-Optimality for Marginal FIM (Maximize det(J_marginal)) ---")
    nuisance_param_index_for_marginal = 2
    loss_fn_marginal = lambda J_full: lossfunctions.determinant_of_marginal_J(
        optimizer._compute_marginal_fisher_info_matrix(
            J_full, nuisance_param_index_for_marginal
        )
    )
    marginal_opt_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=loss_fn_marginal,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=True,
        photon_budget=photon_budget,
        base_seed=42,
    )
    optimized_wavefronts["Marginal_FIM_Optimal"] = marginal_opt_results[
        "bestvec"
    ]
    fisher_matrices_for_wavefronts["Marginal_FIM_Optimal"] = (
        optimizer.J_fisher_info_matrix(marginal_opt_results["bestvec"])
    )

elif OPTIMALITY_CRITERION == "A":
    print("\n--- A-Optimality (Minimize Tr(J_inv)) ---")
    opt_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=lossfunctions.minimize_trace_of_J_inv,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=False,  # Note: A-Optimality is a minimization problem
        photon_budget=photon_budget,
        base_seed=42,
    )
    optimized_wavefronts["Full_FIM_Optimal"] = opt_results["bestvec"]
    fisher_matrices_for_wavefronts["Full_FIM_Optimal"] = (
        optimizer.J_fisher_info_matrix(opt_results["bestvec"])
    )

    print(
        "\n--- A-Optimality for Marginal FIM (Minimize Tr((J_marginal)^-1)) ---"
    )
    nuisance_param_index_for_marginal = 2
    loss_fn_marginal = (
        lambda J_full: lossfunctions.minimize_trace_of_marginal_J_inv(
            optimizer._compute_marginal_fisher_info_matrix(
                J_full, nuisance_param_index_for_marginal
            )
        )
    )
    marginal_opt_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=loss_fn_marginal,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=False,  # Note: A-Optimality is a minimization problem
        photon_budget=photon_budget,
        base_seed=42,
    )
    optimized_wavefronts["Marginal_FIM_Optimal"] = marginal_opt_results[
        "bestvec"
    ]
    fisher_matrices_for_wavefronts["Marginal_FIM_Optimal"] = (
        optimizer.J_fisher_info_matrix(marginal_opt_results["bestvec"])
    )

elif OPTIMALITY_CRITERION == "E":
    print("\n--- E-Optimality (Maximize min_eig(J)) ---")
    opt_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=lossfunctions.maximize_min_eig_J,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=True,
        photon_budget=photon_budget,
        base_seed=42,
    )
    optimized_wavefronts["Full_FIM_Optimal"] = opt_results["bestvec"]
    fisher_matrices_for_wavefronts["Full_FIM_Optimal"] = (
        optimizer.J_fisher_info_matrix(opt_results["bestvec"])
    )

    print(
        "\n--- E-Optimality for Marginal FIM (Maximize min_eig(J_marginal)) ---"
    )
    nuisance_param_index_for_marginal = 2
    loss_fn_marginal = (
        lambda J_full: lossfunctions.maximize_min_eig_marginal_J(
            optimizer._compute_marginal_fisher_info_matrix(
                J_full, nuisance_param_index_for_marginal
            )
        )
    )
    marginal_opt_results = optimizer.torch_optimize_loss_multirun(
        loss_function_of_J=loss_fn_marginal,
        runs=multi_run_count,
        steps=single_run_iterations,
        lr=1e-3,
        verbose=True,
        maximize=True,
        photon_budget=photon_budget,
        base_seed=42,
    )
    optimized_wavefronts["Marginal_FIM_Optimal"] = marginal_opt_results[
        "bestvec"
    ]
    fisher_matrices_for_wavefronts["Marginal_FIM_Optimal"] = (
        optimizer.J_fisher_info_matrix(marginal_opt_results["bestvec"])
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

all_estimates_full_opt = []
all_estimates_marginal = []
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

# Components for Full FIM optimal wavefront
wf_full_opt = optimized_wavefronts["Full_FIM_Optimal"]
dE_vectors_full_opt = [
    dS_tensor[i] @ wf_full_opt for i in range(len(param_names))
]
sim.set_input_field_kx(kx_array, wf_full_opt)
E0_full_opt = sim.compute_zL_fields_direct()
J_inv_full_opt = np.linalg.inv(
    fisher_matrices_for_wavefronts["Full_FIM_Optimal"]
)

# Components for Marginal FIM optimal wavefront
wf_marginal = optimized_wavefronts["Marginal_FIM_Optimal"]
dE_vectors_marginal = [
    dS_tensor[i] @ wf_marginal for i in range(len(param_names))
]
sim.set_input_field_kx(kx_array, wf_marginal)
E0_marginal = sim.compute_zL_fields_direct()
J_inv_marginal = np.linalg.inv(
    fisher_matrices_for_wavefronts["Marginal_FIM_Optimal"]
)

# Components for MIS_x wavefront
wf_mis_x = optimized_wavefronts["MIS_x"]
dE_vectors_mis_x = [dS_tensor[i] @ wf_mis_x for i in range(len(param_names))]
sim.set_input_field_kx(kx_array, wf_mis_x)
E0_mis_x = sim.compute_zL_fields_direct()
J_inv_mis_x = np.linalg.inv(fisher_matrices_for_wavefronts["MIS_x"])

print("Starting estimation loop...")
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

    # Full FIM Optimal Estimation
    sim.set_input_field_kx(kx_array, wf_full_opt)
    E_true = sim.compute_zL_fields_direct()
    Z_noisy_samples = measure_heterodyne(
        E_true, total_photons, nsamples=N_estimations
    )
    estimates = np.array(
        [
            best_estimate_multi(
                Z, E0_full_opt, dE_vectors_full_opt, J_inv_full_opt
            )
            for Z in Z_noisy_samples
        ]
    )
    all_estimates_full_opt.append(estimates)

    # Marginal FIM Optimal Estimation
    sim.set_input_field_kx(kx_array, wf_marginal)
    E_true = sim.compute_zL_fields_direct()
    Z_noisy_samples = measure_heterodyne(
        E_true, total_photons, nsamples=N_estimations
    )
    estimates = np.array(
        [
            best_estimate_multi(
                Z, E0_marginal, dE_vectors_marginal, J_inv_marginal
            )
            for Z in Z_noisy_samples
        ]
    )
    all_estimates_marginal.append(estimates)

    # MIS_x Estimation
    sim.set_input_field_kx(kx_array, wf_mis_x)
    E_true = sim.compute_zL_fields_direct()
    Z_noisy_samples = measure_heterodyne(
        E_true, total_photons, nsamples=N_estimations
    )
    estimates = np.array(
        [
            best_estimate_multi(Z, E0_mis_x, dE_vectors_mis_x, J_inv_mis_x)
            for Z in Z_noisy_samples
        ]
    )
    all_estimates_mis_x.append(estimates)

print("\nSimulation loop finished.")

# %% 7) Plotting Bias and Precision for X-parameter (Multi-param estimator)
bias_x_full_opt = [
    np.mean(estimates[:, 0]) - x_true_pos
    for estimates in all_estimates_full_opt
]
bias_x_marginal = [
    np.mean(estimates[:, 0]) - x_true_pos
    for estimates in all_estimates_marginal
]
bias_x_mis_x = [
    np.mean(estimates[:, 0]) - x_true_pos for estimates in all_estimates_mis_x
]  # NEW
std_x_full_opt = [
    np.std(estimates[:, 0]) for estimates in all_estimates_full_opt
]
std_x_marginal = [
    np.std(estimates[:, 0]) for estimates in all_estimates_marginal
]
std_x_mis_x = [
    np.std(estimates[:, 0]) for estimates in all_estimates_mis_x
]  # NEW

crlb_prefactor = 1 / np.sqrt(2)
crlb_std_x_full_opt = crlb_prefactor * np.sqrt(J_inv_full_opt[0, 0])
crlb_std_x_marginal = crlb_prefactor * np.sqrt(J_inv_marginal[0, 0])
crlb_std_x_mis_x = crlb_prefactor * np.sqrt(J_inv_mis_x[0, 0])  # NEW

fig_x, (ax1_x, ax2_x) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig_x.suptitle(
    f"Multi-Parameter Estimator Performance for X-Parameter ({OPTIMALITY_CRITERION}-Optimality)",
    fontsize=16,
)

ax1_x.plot(
    theta_deviations,
    bias_x_full_opt,
    "o-",
    label=f"{OPTIMALITY_CRITERION}-Optimal (Full)",
    color="#1f77b4",
    markersize=8,
    lw=2,
)
ax1_x.plot(
    theta_deviations,
    bias_x_marginal,
    "s-",
    label=f"{OPTIMALITY_CRITERION}-Optimal (Marginal)",
    color="#ff7f0e",
    markersize=8,
    lw=2,
)
ax1_x.plot(
    theta_deviations,
    bias_x_mis_x,
    "^-",
    label="MIS_x",
    color="#2ca02c",
    markersize=8,
    lw=2,
)
ax1_x.axhline(0, color="k", linestyle="--", label="Unbiased")
ax1_x.set_ylabel("Estimation Bias of $\\hat{x}$", fontsize=14)
ax1_x.legend(fontsize=12)
ax1_x.grid(True, which="both", linestyle=":", alpha=0.6)
ax1_x.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax1_x.set_title("Accuracy (Bias)", fontsize=14)

ax2_x.plot(
    theta_deviations,
    std_x_full_opt,
    "o-",
    label=f"{OPTIMALITY_CRITERION}-Optimal (Full)",
    color="#1f77b4",
    markersize=8,
    lw=2,
)
ax2_x.plot(
    theta_deviations,
    std_x_marginal,
    "s-",
    label=f"{OPTIMALITY_CRITERION}-Optimal (Marginal)",
    color="#ff7f0e",
    markersize=8,
    lw=2,
)
ax2_x.plot(
    theta_deviations,
    std_x_mis_x,
    "^-",
    label="MIS_x",
    color="#2ca02c",
    markersize=8,
    lw=2,
)
ax2_x.axhline(
    y=crlb_std_x_full_opt / np.sqrt(total_photons),
    color="#1f77b4",
    linestyle="--",
    label=f"CRLB for {OPTIMALITY_CRITERION}-Optimal State",
)
ax2_x.axhline(
    y=crlb_std_x_marginal / np.sqrt(total_photons),
    color="#ff7f0e",
    linestyle="--",
    label=f"CRLB for Marginal {OPTIMALITY_CRITERION}-Optimal State",
)
ax2_x.axhline(
    y=crlb_std_x_mis_x / np.sqrt(total_photons),
    color="#2ca02c",
    linestyle="--",
    label="CRLB for MIS_x State",
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

# %% 8) Plotting Bias and Precision for Z-parameter (Multi-param estimator)
bias_z_full_opt = [
    np.mean(estimates[:, 1]) - z_true_pos
    for estimates in all_estimates_full_opt
]
bias_z_marginal = [
    np.mean(estimates[:, 1]) - z_true_pos
    for estimates in all_estimates_marginal
]
bias_z_mis_x = [
    np.mean(estimates[:, 1]) - z_true_pos for estimates in all_estimates_mis_x
]
std_z_full_opt = [
    np.std(estimates[:, 1]) for estimates in all_estimates_full_opt
]
std_z_marginal = [
    np.std(estimates[:, 1]) for estimates in all_estimates_marginal
]
std_z_mis_x = [np.std(estimates[:, 1]) for estimates in all_estimates_mis_x]

crlb_std_z_full_opt = crlb_prefactor * np.sqrt(J_inv_full_opt[1, 1])
crlb_std_z_marginal = crlb_prefactor * np.sqrt(J_inv_marginal[1, 1])
crlb_std_z_mis_x = crlb_prefactor * np.sqrt(J_inv_mis_x[1, 1])

fig_z, (ax1_z, ax2_z) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig_z.suptitle(
    f"Multi-Parameter Estimator Performance for Z-Parameter ({OPTIMALITY_CRITERION}-Optimality)",
    fontsize=16,
)

ax1_z.plot(
    theta_deviations,
    bias_z_full_opt,
    "o-",
    label=f"{OPTIMALITY_CRITERION}-Optimal (Full)",
    color="#1f77b4",
    markersize=8,
    lw=2,
)
ax1_z.plot(
    theta_deviations,
    bias_z_marginal,
    "s-",
    label=f"{OPTIMALITY_CRITERION}-Optimal (Marginal)",
    color="#ff7f0e",
    markersize=8,
    lw=2,
)
ax1_z.plot(
    theta_deviations,
    bias_z_mis_x,
    "^-",
    label="MIS_x",
    color="#2ca02c",
    markersize=8,
    lw=2,
)
ax1_z.axhline(0, color="k", linestyle="--", label="Unbiased")
ax1_z.set_ylabel("Estimation Bias of $\\hat{z}$", fontsize=14)
ax1_z.legend(fontsize=12)
ax1_z.grid(True, which="both", linestyle=":", alpha=0.6)
ax1_z.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax1_z.set_title("Accuracy (Bias)", fontsize=14)

ax2_z.plot(
    theta_deviations,
    std_z_full_opt,
    "o-",
    label=f"{OPTIMALITY_CRITERION}-Optimal (Full)",
    color="#1f77b4",
    markersize=8,
    lw=2,
)
ax2_z.plot(
    theta_deviations,
    std_z_marginal,
    "s-",
    label=f"{OPTIMALITY_CRITERION}-Optimal (Marginal)",
    color="#ff7f0e",
    markersize=8,
    lw=2,
)
ax2_z.plot(
    theta_deviations,
    std_z_mis_x,
    "^-",
    label="MIS_x",
    color="#2ca02c",
    markersize=8,
    lw=2,
)
ax2_z.axhline(
    y=crlb_std_z_full_opt / np.sqrt(total_photons),
    color="#1f77b4",
    linestyle="--",
    label=f"CRLB for {OPTIMALITY_CRITERION}-Optimal State",
)
ax2_z.axhline(
    y=crlb_std_z_marginal / np.sqrt(total_photons),
    color="#ff7f0e",
    linestyle="--",
    label=f"CRLB for Marginal {OPTIMALITY_CRITERION}-Optimal State",
)
ax2_z.axhline(
    y=crlb_std_z_mis_x / np.sqrt(total_photons),
    color="#2ca02c",
    linestyle="--",
    label="CRLB for MIS_x State",
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
naive_bias_x_full_opt = []
naive_bias_x_marginal = []
naive_bias_x_mis_x = []
naive_bias_z_full_opt = []
naive_bias_z_marginal = []
naive_bias_z_mis_x = []

dEx_full_opt = dE_vectors_full_opt[0]
dEx_marginal = dE_vectors_marginal[0]
dEx_mis_x = dE_vectors_mis_x[0]
dEz_full_opt = dE_vectors_full_opt[1]
dEz_marginal = dE_vectors_marginal[1]
dEz_mis_x = dE_vectors_mis_x[1]


print(
    "\nApplying naive single-parameter estimator to noise-free data for both X and Z..."
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

    # Full FIM Optimal
    sim.set_input_field_kx(kx_array, wf_full_opt)
    E_true = sim.compute_zL_fields_direct()
    naive_bias_x_full_opt.append(
        best_estimate(E_true, dEx_full_opt, E0_full_opt) - x_true_pos
    )
    naive_bias_z_full_opt.append(
        best_estimate(E_true, dEz_full_opt, E0_full_opt) - z_true_pos
    )

    # Marginal FIM Optimal
    sim.set_input_field_kx(kx_array, wf_marginal)
    E_true = sim.compute_zL_fields_direct()
    naive_bias_x_marginal.append(
        best_estimate(E_true, dEx_marginal, E0_marginal) - x_true_pos
    )
    naive_bias_z_marginal.append(
        best_estimate(E_true, dEz_marginal, E0_marginal) - z_true_pos
    )

    # NEW: MIS_x
    sim.set_input_field_kx(kx_array, wf_mis_x)
    E_true = sim.compute_zL_fields_direct()
    naive_bias_x_mis_x.append(
        best_estimate(E_true, dEx_mis_x, E0_mis_x) - x_true_pos
    )
    naive_bias_z_mis_x.append(
        best_estimate(E_true, dEz_mis_x, E0_mis_x) - z_true_pos
    )

print("\nBias calculation for naive estimators finished.")

fig_naive, (ax_x, ax_z) = plt.subplots(1, 2, figsize=(14, 6))
fig_naive.suptitle(
    f"Bias of Naive Single-Parameter Estimator vs. Nuisance Mismatch ({OPTIMALITY_CRITERION}-Optimality)",
    fontsize=16,
)
plt.subplots_adjust(wspace=0.05)

# --- Left Plot: Bias of X-hat ---
ax_x.plot(
    theta_deviations,
    naive_bias_x_full_opt,
    "o-",
    label=f"{OPTIMALITY_CRITERION}-Optimal (Full)",
    color="#1f77b4",
    markersize=8,
)
ax_x.plot(
    theta_deviations,
    naive_bias_x_marginal,
    "s-",
    label=f"{OPTIMALITY_CRITERION}-Optimal (Marginal)",
    color="#ff7f0e",
    markersize=8,
)
ax_x.plot(
    theta_deviations,
    naive_bias_x_mis_x,
    "^-",
    label="MIS_x",
    color="#2ca02c",
    markersize=8,
)  # NEW
ax_x.axhline(0, color="k", linestyle="--")
ax_x.grid(True, linestyle=":")
ax_x.set_title("Bias of $\\hat{x}$ Estimate", fontsize=14)
ax_x.set_xlabel(
    "Nuisance Deviation ($\\theta_{true} - \\theta_0$) [rad]", fontsize=12
)
ax_x.set_ylabel("Systematic Estimation Bias of $\\hat{x}$", fontsize=12)
ax_x.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax_x.legend(fontsize=10)

# --- Right Plot: Bias of Z-hat ---
ax_z.plot(
    theta_deviations,
    naive_bias_z_full_opt,
    "o-",
    label=f"{OPTIMALITY_CRITERION}-Optimal (Full)",
    color="#1f77b4",
    markersize=8,
)
ax_z.plot(
    theta_deviations,
    naive_bias_z_marginal,
    "s-",
    label=f"{OPTIMALITY_CRITERION}-Optimal (Marginal)",
    color="#ff7f0e",
    markersize=8,
)
ax_z.plot(
    theta_deviations,
    naive_bias_z_mis_x,
    "^-",
    label="MIS_x",
    color="#2ca02c",
    markersize=8,
)  # NEW
ax_z.axhline(0, color="k", linestyle="--")
ax_z.grid(True, linestyle=":")
ax_z.set_title("Bias of $\\hat{z}$ Estimate", fontsize=14)
ax_z.set_xlabel(
    "Nuisance Deviation ($\\theta_{true} - \\theta_0$) [rad]", fontsize=12
)
ax_z.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax_z.yaxis.tick_right()
ax_z.yaxis.set_label_position("right")
ax_z.set_ylabel("Systematic Estimation Bias of $\\hat{z}$", fontsize=12)
ax_z.legend(fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('out.png', dpi=300)

plt.show()

# %% Print J for one of the optimal states
J_to_print = fisher_matrices_for_wavefronts["Full_FIM_Optimal"]
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
