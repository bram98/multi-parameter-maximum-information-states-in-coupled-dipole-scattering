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
movable_dipoles_line = placedipoles.place_line_of_dipoles(
    center=(5.0, 0.0), number=7
)
frozen_dipoles_line = placedipoles.place_line_of_dipoles(
    center=(3.0, 0.0), number=7
)

### Quasi-random double line
movable_dipoles_line = placedipoles.place_quasi_random_line_of_dipoles(
    center=(5.0, 0.0), number=6, small_dist=0.786, start_index=5
)
frozen_dipoles_line = placedipoles.place_quasi_random_line_of_dipoles(
    center=(3.0, 0.0), number=6, small_dist=0.786, start_index=22
)

### Square
movable_dipoles_line = placedipoles.place_square_of_dipoles(
    center=(5.0, 0.0), side=1.0
)

frozen_dipoles_line = placedipoles.place_line_of_dipoles(
    center=(2.5, 0.0), number=7
)
# frozen_dipoles_line = None


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
print("\n--- D-Optimality (Maximize det(J)) ---")
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
wavefront_detJ_opt = det_J_optimization_results["bestvec"]
optimized_wavefronts["DetJ_torch"] = wavefront_detJ_opt
J_for_detJ_opt = optimizer.J_fisher_info_matrix(wavefront_detJ_opt)
fisher_matrices_for_wavefronts["DetJ_torch"] = J_for_detJ_opt

nuisance_param_index_for_marginal = 2
loss_fn_det_marginal = (
    lambda J_full_torch: lossfunctions.determinant_of_marginal_J(
        optimizer._compute_marginal_fisher_info_matrix(
            J_full_torch, nuisance_param_index_for_marginal
        )
    )
)
print("\n--- D-Optimality for Marginal FIM (Maximize det(J_marginal)) ---")
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
opt_key_marginal = f"DetJ_marginal_n{nuisance_param_index_for_marginal}_torch"
optimized_wavefronts[opt_key_marginal] = wavefront_detJ_marginal_opt
J_full_for_detJ_marginal = optimizer.J_fisher_info_matrix(
    wavefront_detJ_marginal_opt
)
fisher_matrices_for_wavefronts[opt_key_marginal] = J_full_for_detJ_marginal

# %% 6) Run simulation with the CORRECT multi-parameter estimator
total_photons = 1e6
N_estimations = 2000
theta0_rad = 0.0
x_true_pos = 0.0
z_true_pos = 0.0
N_deviations = 51
max_deviation = 0.02 * 10
theta_deviations = np.linspace(-max_deviation, max_deviation, N_deviations)

all_estimates_detJ = []
all_estimates_marginal = []

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

wf_detJ = optimized_wavefronts["DetJ_torch"]
dE_vectors_detJ = [dS_tensor[i] @ wf_detJ for i in range(len(param_names))]
sim.set_input_field_kx(kx_array, wf_detJ)
E0_detJ = sim.compute_zL_fields_direct()
J_inv_detJ = np.linalg.inv(fisher_matrices_for_wavefronts["DetJ_torch"])

wf_marginal = optimized_wavefronts[opt_key_marginal]
dE_vectors_marginal = [
    dS_tensor[i] @ wf_marginal for i in range(len(param_names))
]
sim.set_input_field_kx(kx_array, wf_marginal)
E0_marginal = sim.compute_zL_fields_direct()
J_inv_marginal = np.linalg.inv(
    fisher_matrices_for_wavefronts[opt_key_marginal]
)

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

    sim.set_input_field_kx(kx_array, wf_detJ)
    E_true = sim.compute_zL_fields_direct()
    Z_noisy_samples = measure_heterodyne(
        E_true, total_photons, nsamples=N_estimations
    )
    estimates = np.array(
        [
            best_estimate_multi(Z, E0_detJ, dE_vectors_detJ, J_inv_detJ)
            for Z in Z_noisy_samples
        ]
    )
    all_estimates_detJ.append(estimates)

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
print("\nSimulation loop finished.")

# %% 7) Plotting Bias and Precision for X-parameter (Multi-param estimator)
bias_x_detJ = [
    np.mean(estimates[:, 0]) - x_true_pos for estimates in all_estimates_detJ
]
bias_x_marginal = [
    np.mean(estimates[:, 0]) - x_true_pos
    for estimates in all_estimates_marginal
]
std_x_detJ = [np.std(estimates[:, 0]) for estimates in all_estimates_detJ]
std_x_marginal = [
    np.std(estimates[:, 0]) for estimates in all_estimates_marginal
]

crlb_prefactor = 1 / np.sqrt(2)
crlb_std_x_detJ = crlb_prefactor * np.sqrt(J_inv_detJ[0, 0])
crlb_std_x_marginal = crlb_prefactor * np.sqrt(J_inv_marginal[0, 0])

fig_x, (ax1_x, ax2_x) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig_x.suptitle(
    "Multi-Parameter Estimator Performance for X-Parameter", fontsize=16
)

ax1_x.plot(
    theta_deviations,
    bias_x_detJ,
    "o-",
    label="D-Optimal (Max det(J))",
    color="#1f77b4",
    markersize=8,
    lw=2,
)
ax1_x.plot(
    theta_deviations,
    bias_x_marginal,
    "s-",
    label="Nuisance-Aware (Max det(J_marginal))",
    color="#ff7f0e",
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
    std_x_detJ,
    "o-",
    label="D-Optimal (Max det(J))",
    color="#1f77b4",
    markersize=8,
    lw=2,
)
ax2_x.plot(
    theta_deviations,
    std_x_marginal,
    "s-",
    label="Nuisance-Aware (Max det(J_marginal))",
    color="#ff7f0e",
    markersize=8,
    lw=2,
)
ax2_x.axhline(
    y=crlb_std_x_detJ / np.sqrt(total_photons),
    color="#1f77b4",
    linestyle="--",
    label="CRLB for D-Optimal State",
)
ax2_x.axhline(
    y=crlb_std_x_marginal / np.sqrt(total_photons),
    color="#ff7f0e",
    linestyle="--",
    label="CRLB for Nuisance-Aware State",
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
plt.show()

# %% 8) Plotting Bias and Precision for Z-parameter (Multi-param estimator)
bias_z_detJ = [
    np.mean(estimates[:, 1]) - z_true_pos for estimates in all_estimates_detJ
]
bias_z_marginal = [
    np.mean(estimates[:, 1]) - z_true_pos
    for estimates in all_estimates_marginal
]
std_z_detJ = [np.std(estimates[:, 1]) for estimates in all_estimates_detJ]
std_z_marginal = [
    np.std(estimates[:, 1]) for estimates in all_estimates_marginal
]

crlb_std_z_detJ = crlb_prefactor * np.sqrt(J_inv_detJ[1, 1])
crlb_std_z_marginal = crlb_prefactor * np.sqrt(J_inv_marginal[1, 1])

fig_z, (ax1_z, ax2_z) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig_z.suptitle(
    "Multi-Parameter Estimator Performance for Z-Parameter", fontsize=16
)

ax1_z.plot(
    theta_deviations,
    bias_z_detJ,
    "o-",
    label="D-Optimal (Max det(J))",
    color="#1f77b4",
    markersize=8,
    lw=2,
)
ax1_z.plot(
    theta_deviations,
    bias_z_marginal,
    "s-",
    label="Nuisance-Aware (Max det(J_marginal))",
    color="#ff7f0e",
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
    std_z_detJ,
    "o-",
    label="D-Optimal (Max det(J))",
    color="#1f77b4",
    markersize=8,
    lw=2,
)
ax2_z.plot(
    theta_deviations,
    std_z_marginal,
    "s-",
    label="Nuisance-Aware (Max det(J_marginal))",
    color="#ff7f0e",
    markersize=8,
    lw=2,
)
ax2_z.axhline(
    y=crlb_std_z_detJ / np.sqrt(total_photons),
    color="#1f77b4",
    linestyle="--",
    label="CRLB for D-Optimal State",
)
ax2_z.axhline(
    y=crlb_std_z_marginal / np.sqrt(total_photons),
    color="#ff7f0e",
    linestyle="--",
    label="CRLB for Nuisance-Aware State",
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
plt.show()

# %% 9) Revisiting the Naive Estimator to Visualize its Bias for X and Z
naive_bias_x_detJ = []
naive_bias_x_marginal = []
naive_bias_z_detJ = []
naive_bias_z_marginal = []

dEx_detJ = dE_vectors_detJ[0]
dEx_marginal = dE_vectors_marginal[0]
dEz_detJ = dE_vectors_detJ[1]
dEz_marginal = dE_vectors_marginal[1]

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

    sim.set_input_field_kx(kx_array, wf_detJ)
    E_true_detJ_wf = sim.compute_zL_fields_direct()
    naive_est_x = best_estimate(E_true_detJ_wf, dEx_detJ, E0_detJ)
    naive_bias_x_detJ.append(naive_est_x - x_true_pos)
    naive_est_z = best_estimate(E_true_detJ_wf, dEz_detJ, E0_detJ)
    naive_bias_z_detJ.append(naive_est_z - z_true_pos)

    sim.set_input_field_kx(kx_array, wf_marginal)
    E_true_marginal_wf = sim.compute_zL_fields_direct()
    naive_est_x = best_estimate(E_true_marginal_wf, dEx_marginal, E0_marginal)
    naive_bias_x_marginal.append(naive_est_x - x_true_pos)
    naive_est_z = best_estimate(E_true_marginal_wf, dEz_marginal, E0_marginal)
    naive_bias_z_marginal.append(naive_est_z - z_true_pos)
print("\nBias calculation for naive estimators finished.")

fig_naive, (ax_x, ax_z) = plt.subplots(1, 2, figsize=(14, 6))
fig_naive.suptitle(
    "Bias of Naive Single-Parameter Estimator vs. Nuisance Mismatch",
    fontsize=16,
)
plt.subplots_adjust(wspace=0.05)  # Make plots very close

# --- Left Plot: Bias of X-hat ---
ax_x.plot(
    theta_deviations,
    naive_bias_x_detJ,
    "o-",
    label="D-Optimal WF",
    color="#1f77b4",
    markersize=8,
)
ax_x.plot(
    theta_deviations,
    naive_bias_x_marginal,
    "s-",
    label="Nuisance-Aware WF",
    color="#ff7f0e",
    markersize=8,
)
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
    naive_bias_z_detJ,
    "o-",
    label="D-Optimal WF",
    color="#1f77b4",
    markersize=8,
)
ax_z.plot(
    theta_deviations,
    naive_bias_z_marginal,
    "s-",
    label="Nuisance-Aware WF",
    color="#ff7f0e",
    markersize=8,
)
ax_z.axhline(0, color="k", linestyle="--")
ax_z.grid(True, linestyle=":")
ax_z.set_title("Bias of $\\hat{z}$ Estimate", fontsize=14)
ax_z.set_xlabel(
    "Nuisance Deviation ($\\theta_{true} - \\theta_0$) [rad]", fontsize=12
)
ax_z.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

# Move the y-axis labels and ticks to the right side
ax_z.yaxis.tick_right()
ax_z.yaxis.set_label_position("right")
ax_z.set_ylabel("Systematic Estimation Bias of $\\hat{z}$", fontsize=12)
ax_z.legend(fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %% Print J for DetJ

J_to_print = fisher_matrices_for_wavefronts["DetJ_torch"]
param_names = ["x", "z", "theta"]

print("\\n" + "=" * 50)
print("Fisher Information Matrix J for the D-Optimal Wavefront:")
print("=" * 50)
print(f"{'':>6} | {'x':^12} | {'z':^12} | {'theta':^12}")
print("-" * 50)
for i, row in enumerate(J_to_print):
    print(
        f"{param_names[i]:>6} | {row[0]:>12.3e} | {row[1]:>12.3e} | {row[2]:>12.3e}"
    )
print("=" * 50)
