import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

import scatteringlibs.lossfunctions
from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction, LossFunctionsDict
from scatteringlibs.matplotlibmagic import PlotGroup, register_turbo
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

sys.exit()


# %% -1) define functions
def measure_heterodyne(E, photon_budget, nsamples=None, seed=None):
    """
    Models a camera with complex field detection under shot noise, consistent
    with the quantum mechanical model where the variance of a quadrature
    measurement of is 1/2 for a coherent field with one photon on average. If
    E = P + i Q, then [P, Q] = i/2 and the vacuum variance is 1/4. Measuring P
    and Q jointly gives variance 1/2 for each quadrature.
    """
    E = np.ascontiguousarray(E).astype(np.complex128)
    signal_field = E * np.sqrt(photon_budget)
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
    return field_samples.view(np.complex128) / np.sqrt(photon_budget)


def measure_noisy(E, noise_std_dev, nsamples=None, seed=None):
    """
    Models a camera with complex field detection under shot noise, consistent
    with the quantum mechanical model where the variance of a quadrature
    measurement of is 1/2 for a coherent field with one photon on average. If
    E = P + i Q, then [P, Q] = i/2 and the vacuum variance is 1/4. Measuring P
    and Q jointly gives variance 1/2 for each quadrature.
    """
    E = np.ascontiguousarray(E).astype(np.complex128)
    E_real = E.view(np.float64)
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
    return field_samples.view(np.complex128)


def measure_heterodyne(E, photon_budget, nsamples=None, seed=None):
    """
    Models a camera with complex field detection under shot noise, consistent
    with the quantum mechanical model where the variance of a quadrature
    measurement of is 1/2 for a coherent field with one photon on average. If
    E = P + i Q, then [P, Q] = i/2 and the vacuum variance is 1/4. Measuring P
    and Q jointly gives variance 1/2 for each quadrature.
    """
    E = np.ascontiguousarray(E)
    signal_field = E * np.sqrt(photon_budget)
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
    return field_samples.view(np.complex128) / np.sqrt(photon_budget)


def measure_noisy(E, noise_std_dev, nsamples=None, seed=None):
    """
    What happens when the detected wavefronts are noisy?
    """
    E = np.ascontiguousarray(E)
    E_real = E.view(np.float64)
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
    return field_samples.view(np.complex128) / np.sqrt(photon_budget)


def best_estimate_multi(Z_noisy, E0, dE):
    """
    Optimal multi-parameter linear estimator. Assumes the model
        E_true = E0 + theta @ dE
    where Z_noisy ~ normal(mu=E_true, sigma=?)

    Z_noisy: shape (nsamples, nhilbert)
    E0: shape (nhilbert,)
    dE: shape (nparams, nhilhbert)

    returns theta_est: shape (nparams, nsamples)

    """
    H = np.asarray(dE).T
    delta_E = Z_noisy.T - E0.reshape(-1, 1)
    theta_est_vector = np.linalg.inv(np.real(H.conj().T @ H)) @ np.real(
        H.conj().T @ delta_E
    )
    return theta_est_vector


def best_estimate(E, E0, dE_dtheta):
    """
    Single-parameter estimator.
    """
    theta_est = np.vecdot(dE_dtheta, E - E0) / np.vecdot(dE_dtheta, dE_dtheta)
    theta_est = np.real(theta_est)
    return theta_est


if __name__ == "__main__":
    theta0 = np.array([0.1, 0.2, 0.3])
    E0 = np.random.randn(5, 2).view(complex).squeeze()
    dE = np.random.randn(3, 5, 2).view(complex).squeeze()
    noise = np.random.randn(5) * 0.05

    E = E0 + dE[0] * theta0[0]
    s = "Estimation without noise"
    print(s, "\n" + "-" * len(s))

    print(
        "Single parameter estimate: ",
        best_estimate([E, E + noise], E0, dE[0]),
    )

    E = E0 + theta0 @ dE

    print(
        f"Multilinear parameter estimate:\n{best_estimate_multi(np.array([E, E+noise]), E0, dE)}\n"
    )

    s = "Verifying CRLB"
    print(s, "\n" + "-" * len(s))

    photon_budget = 1e3
    J = dE.conj() @ dE.T
    J = J + J.T
    J = J.real * photon_budget

    Z_noisy = measure_heterodyne(
        E,
        photon_budget,
        nsamples=int(1e3),
    )
    theta_est = best_estimate_multi(Z_noisy, E0, dE)
    theta_std = np.std(theta_est, axis=1)

    crlb = np.sqrt(np.diag(np.linalg.inv(J)))

    print(f"{'Standard dev.:':<15}", theta_std)
    print(f"{'CRLB:':<15}", crlb)

    Sigma = theta_est[:, None, :] * theta_est[None, :, :]
    Sigma = np.mean(Sigma, axis=2)
    print("Sigma: \n", Sigma)


# %% 0) settings
OPTIMALITY_CRITERION = "D"  # Can be 'D', 'A', or 'E'

verbose_plotting = True
max_angle = 80
number_of_input_channels = 100
number_of_output_channels = 300
param_names = ["shiftX", "shiftZ", "rotation"]
delta_param = 1e-3
photon_budget = 1e1
multi_run_count = 5
single_run_iterations = 4000

# %% Loading in data
# project_name = "quasirandom_longrun - Copy"
# project_name = "quasirandom_shortrun_dist2-8"
# project_name = "quasirandom_shortrun_dist2-6"
# project_name = "quasirandom_shortrun_dist2-4"
project_name = "quasirandom_longrun_dist2-8"

# project_name = "quasirandom"
# project_name = "square"
data_folder = f"../data/{project_name}"

sim = CoupledDipoleSimulator.load(folder=data_folder)
fisher_calc = FisherMatrix.load(folder=data_folder)
loss_functions_dict = LossFunctionsDict.load(folder=data_folder)
loss_functions_sweep_dict = LossFunctionsDict.load(
    name="LossFunctionsSweepDict", folder=data_folder
)


original_movable_dipoles = sim.movable_scatterers.copy()
original_frozen_dipoles = sim.frozen_scatterers.copy()

print("Lossfunction Names")
for lf_name in loss_functions_dict.keys():
    print(f"\t- {lf_name}")

# %% Noisy measurements

lf = loss_functions_dict["det"]
fisher_calc.use_heterodyne = True
optimizer = Optimizer(fisher_calc.F)

E = lf.wavefront

det_arr = []
det_std_arr = []
n = 50
noise_readout_arr = np.ones(n)
noise_heterodyne = np.sqrt(0.5) ** 2
# I_in = np.linalg.norm(E)**2
I_in_arr = np.geomspace(2e-1, 1e10, num=n)

noise_arr = np.sqrt(noise_readout_arr**2 + noise_heterodyne**2)

SNR_arr = np.sqrt(I_in_arr) / noise_arr

for I_in, noise_readout in zip(I_in_arr, noise_arr):
    E_noisy = measure_noisy(E * np.sqrt(I_in), noise_readout, nsamples=100)

    E_noisy *= 1 / np.linalg.norm(E_noisy, axis=1)[:, np.newaxis]
    # E_noisy *= 1 / np.sqrt(I_in) # Wrong: noisy states get huge intensity

    J_noisy = [optimizer.J_fisher_info_matrix(E_) for E_ in E_noisy]
    # det_noisy = np.linalg.det(J_noisy)**(1/3)
    det_noisy = np.linalg.det(J_noisy)
    det_arr.append(np.mean(det_noisy))
    det_std_arr.append(np.std(det_noisy))

det_arr = np.array(det_arr)
det_std_arr = np.array(det_std_arr)

plt.plot(1 / SNR_arr, det_arr)
plt.fill_between(
    1 / SNR_arr, det_arr - det_std_arr, det_arr + det_std_arr, alpha=0.1
)
plt.xlabel("1/SNR")
plt.ylabel("D-optimality$^{1/3}$")


# %% Noisy measurements V2

dS0 = fisher_calc.dS
dS0_flat = dS0.flatten()

fisher_calc_noisy = FisherMatrix(dS0, use_heterodyne=True)


det_arr = []
det_std_arr = []
n = 20
noise_std_heterodyne = np.sqrt(0.5)
Iin = 100000.0
I_in_arr = np.ones(n) * Iin

one_over_SNR = np.linspace(1 / (np.sqrt(1.9 * Iin)), 1, num=n)

noise_readout_arr = np.sqrt(
    I_in_arr * one_over_SNR**2 - noise_std_heterodyne**2
)

# I_in = np.linalg.norm(E)**2
# I_in_arr = np.geomspace(2e-1, 1e6, num=n)

noise_arr = np.sqrt(noise_readout_arr**2 + noise_heterodyne**2)

# SNR_arr = np.sqrt(I_in_arr) / noise_arr

E_noisy_arr = []
for i, (I_in, noise_readout) in enumerate(zip(I_in_arr, noise_arr)):
    dS_noisy = measure_noisy(
        dS0_flat * np.sqrt(I_in), noise_readout, nsamples=30
    )
    dS_noisy = dS_noisy.reshape((-1,) + dS0.shape) / np.sqrt(I_in)

    J_noisy_arr = []
    for j, dS in enumerate(dS_noisy):
        print(
            f"noise level {i+1}/{len(noise_arr)}  dS sample {j+1}/{len(dS_noisy)}"
        )
        fisher_calc_noisy.dS = dS
        optimizer_noisy = Optimizer(fisher_calc_noisy.F)

        result = optimizer_noisy.torch_optimize_loss(
            lossfunctions.determinant_of_J,
            steps=1000,
            lr=1e-2,
            maximize=True,
            E_init=lf.wavefront,
            E_init_strength=1.0,
        )
        E_noisy = result["bestvec"]
        E_noisy_arr.append(E_noisy)
        J_noisy_arr.append(optimizer.J_fisher_info_matrix(E_noisy))
    det_noisy = np.linalg.det(J_noisy_arr) ** (1 / 3)
    det_arr.append(np.mean(det_noisy))
    det_std_arr.append(np.std(det_noisy))

det_arr = np.array(det_arr)
det_std_arr = np.array(det_std_arr)


# %% plot noisy meas


fig, ax = plt.subplots(figsize=(0.495 * TEXTWIDTH, 0.4 * TEXTWIDTH), dpi=DPI)
plt.plot(one_over_SNR, det_arr)
plt.fill_between(
    one_over_SNR, det_arr - det_std_arr, det_arr + det_std_arr, alpha=0.2
)
# plt.axhline(2714.871457318225 / 2)
# plt.axvline(1)
plt.xlabel("1/SNR")
plt.ylabel("D-optimality$^{1/3}$")
plt.tight_layout()
plt.grid(**GRID_KWARGS)
plt.savefig(
    r"C:\Users\5914787\OneDrive\Bureaublad\PhD\Thesis\MPMIS for optical metrology\figures\noisy_measurement.png"
)
plt.savefig(
    r"C:\Users\5914787\OneDrive\Bureaublad\PhD\Thesis\MPMIS for optical metrology\figures\noisy_measurement.pdf"
)
plt.show()
# %%

# Analytically calculate detection noise degradation (No random samples needed!)
det_arr = []

J_ideal_norm1 = optimizer.J_fisher_info_matrix(E)  # Base J for 1 photon


I_in_arr = np.geomspace(2e4, 1e5)

for I_in, noise_readout in zip(I_in_arr, noise_readout_arr):
    # Total variance = heterodyne variance (0.5) + readout variance
    total_variance = 0.5 + noise_readout**2

    # The Fisher matrix scales directly by Intensity / (2 * total_variance)
    # Note: If noise_readout = 0, this multiplier is exactly I_in
    J_noisy_det = J_ideal_norm1 * (I_in / (2 * total_variance))

    det_arr.append(np.linalg.det(J_noisy_det))

plt.plot(1 / SNR_arr, det_arr)


# %%


# %%

sim.visualize_intensity(title="a")
# %% 6) Run simulation with the multi-parameter estimator
N_estimations = int(1e5)
theta0_rad = 0.0
x_true_pos = 0.0
z_true_pos = 0.0
N_deviations = 25
max_deviation = 0.02
theta_deviations = np.linspace(-max_deviation, max_deviation, N_deviations)

# # Lists to store results for each strategy
all_estimates_nuisance_aware = []
all_estimates_subdet_xz = []
all_estimates_full_fim = []
all_estimates_mis_x = []


kx_array = sim.k_x_array
dS = fisher_calc.dS
# S = fisher_calc.S
param_names = fisher_calc.param_names


sim.set_scatterers(movable_scatterers=original_movable_dipoles)

S, _ = sim.build_scattering_matrix(flux_normalize=False)

dS = placedipoles.calculate_dS_for_3_parameters(
    sim.get_kwargs(),
    sim.theta_inc_deg,
    movable_dipoles=original_movable_dipoles,
    frozen_dipoles=original_frozen_dipoles,
    delta_param=delta_param,
    flux_normalize=False,
)
fisher_calc = FisherMatrix(dS=dS, S=S, use_heterodyne=True)
optimizer = Optimizer(fisher_calc.F)

# %%
photon_budget = 1e2
optimality_criterion = "det"
lf = {}
wf = {}
E0 = {}
dE = {}
J = {}
used_params = {}
all_estimates = {}

strategies = ["nuisance_aware", "full_fim", "subdet_xz", "mis_x", "mis_z"]

# Partial (nuisance aware)
lf["nuisance_aware"] = loss_functions_dict[f"partial_{optimality_criterion}"]
lf["full_fim"] = loss_functions_dict[f"{optimality_criterion}"]
lf["subdet_xz"] = loss_functions_dict[f"dropped_{optimality_criterion}"]
lf["mis_x"] = loss_functions_dict["x"]
lf["mis_z"] = loss_functions_dict["z"]

used_params["nuisance_aware"] = range(3)
used_params["full_fim"] = range(3)
# used_params["subdet_xz"] = [0, 1]
# used_params["mis_x"] = [0]
# used_params["mis_z"] = [1]
used_params["subdet_xz"] = range(3)
used_params["mis_x"] = range(3)
used_params["mis_z"] = range(3)


for strategy in strategies:
    wf[strategy] = lf[strategy].wavefront
    E0[strategy] = S @ wf[strategy]
    dE[strategy] = np.array(
        [dS[i] @ wf[strategy] for i in used_params[strategy]]
    )
    J_indices = np.ix_(used_params[strategy], used_params[strategy])
    # J[strategy] = lf[strategy].J[J_indices]
    J[strategy] = optimizer.J_fisher_info_matrix(wf[strategy])[J_indices]
    all_estimates[strategy] = []

sim.set_scatterers(
    movable_scatterers=original_movable_dipoles,
    frozen_scatterers=original_frozen_dipoles,
)
sim.reset()
sim.set_input_field_kx(kx_array, wf["full_fim"], flux_normalize=False)

# E1 = E0["full_fim"]
# # Compute E0 exactly how E_true will be computed
# E2 = sim.compute_output_fields_direct()
# E2 = np.fft.fft(E2, norm="ortho")
# E2 = np.fft.fftshift(E2)

# sim.reset()
# S2, _ = sim.build_scattering_matrix(flux_normalize=False)
# E3 = S2 @ wf["full_fim"]

# plt.plot(E1, label="E1", ls='-')
# plt.plot(E2+.0, label="E2", ls=':')
# plt.plot(E3+.0, label="E3", ls='-.')
# plt.legend()
# plt.show()
E_true_arr = []
E_lin_arr = []

print("\nStarting estimation loop...")
for i, theta_dev in enumerate(theta_deviations):
    print(
        f"\rSimulating deviation {i+1}/{len(theta_deviations)}: {theta_dev:.4f} rad",
        end="     ",
    )
    theta_true = theta0_rad + theta_dev
    true_movable_dipoles = placedipoles.shift_and_rotate_dipoles(
        original_movable_dipoles, x_true_pos, z_true_pos, theta_true
    )
    # if frozen_dipoles_line is not None and frozen_dipoles_line.size > 0:
    #     sim.scatterers = np.hstack((frozen_dipoles_line, true_movable_dipoles))
    # else:
    #     sim.scatterers = true_movable_dipoles

    sim.set_scatterers(
        movable_scatterers=true_movable_dipoles,
        frozen_scatterers=original_frozen_dipoles,
    )

    # sim.reset()

    # --- Run Estimations for Each Strategy ---
    for j, strategy in enumerate(strategies):
        sim.reset()
        sim.set_input_field_kx(kx_array, wf[strategy], flux_normalize=False)

        E_true = sim.compute_output_fields_direct()
        E_true = np.fft.fft(E_true, norm="ortho")
        E_true = np.fft.fftshift(E_true)

        # E_true_arr.append(E_true - (E0[strategy] ))
        # dtheta = np.array([x_true_pos, z_true_pos, theta_true])
        # dtheta = dtheta[used_params[strategy]]
        # E_true_test = E0[strategy] + dtheta @ dE[strategy]

        Z_noisy = measure_heterodyne(
            E_true, photon_budget, nsamples=N_estimations, seed=i * 10 + j
        )
        estimates = best_estimate_multi(Z_noisy, E0[strategy], dE[strategy])
        all_estimates[strategy].append(estimates)
        if strategy == "full_fim":

            # E_true_arr.append(E_true - (E0[strategy] + theta_true*dE[strategy][2]))
            # E_true_arr.append(E_true - (E0[strategy]))
            E_true_arr.append(E_true)
            E_lin_arr.append(E0[strategy] + theta_true * dE[strategy][2])
            # plt.plot(E_true, label="matrix")
            # plt.plot(E_true2, label="compute")
            # plt.legend()
            # plt.title(theta_true)
            # plt.show()
            # sim.visualize_intensity()
            print(
                np.mean(estimates, axis=1),
                f"--- [{x_true_pos:.3f} {z_true_pos:.3f} {theta_true:.3f}]",
            )
E_true_arr = np.array(E_true_arr)
E_lin_arr = np.array(E_lin_arr)

# plt.plot(E_true_arr[:, 12])
# plt.plot(E_lin_arr[:, 12])
plt.imshow(np.abs(E_true_arr - E_lin_arr))
plt.colorbar()
plt.show()

print("\nSimulation loop finished.")


# %% 7) Plotting Bias and Precision for X and Z Parameters
def plot_performance_on_axes(ax1, strategies, param_name, param_index):
    """
    Plots bias and precision for a given parameter on a set of two axes.
    `axes` should be a tuple or list of two axes: (ax_bias, ax_precision).
    """
    global bias, std
    # ax1, ax2 = axes

    if param_name == "X":
        strategies = [
            strategy for strategy in strategies if strategy != "mis_z"
        ]
        true_pos = 0
    elif param_name == "Z":
        strategies = [
            strategy for strategy in strategies if strategy != "mis_x"
        ]
        true_pos = 0
    print(strategies)
    bias = {}
    std = {}
    for strategy in strategies:
        idx = np.where(np.array(used_params[strategy]) == param_index)[0]
        bias[strategy] = [np.mean(est[idx]) for est in all_estimates[strategy]]
        std[strategy] = [np.std(est[idx]) for est in all_estimates[strategy]]
        bias[strategy] = np.array(bias[strategy]) - true_pos
        std[strategy] = np.array(std[strategy])

    # --- Define Plotting Styles ---
    # Inspired by result_Pareto_frontier.py
    opt_criterion_label = f"{OPTIMALITY_CRITERION}-Opt."
    styles = {
        "full_fim": {
            "label": "Full",
            "color": "green",  # Blue
            "marker": "^",
        },
        "subdet_xz": {
            "label": rf"Subblock",
            "color": "red",  # Orange
            "marker": "x",
        },
        "nuisance_aware": {
            "label": rf"Partial",
            "color": "blue",  # Green
            "marker": "s",
        },
        "mis_x": {
            "label": r"Scalar",
            "color": "purple",  # Red
            "marker": ".",
        },
        "mis_z": {
            "label": r"Scalar",
            "color": "purple",  # Red
            "marker": ".",
        },
    }

    # --- Bias Plot ---
    # Order: FullFIM, Subdet, Nuisance-Aware, MIS_x
    for strategy in strategies:
        # print(strategy, theta_deviations.shape, np.array(bias[strategy]).shape)
        ax1.plot(
            theta_deviations,
            bias[strategy],
            linestyle="-",
            markersize=6,
            lw=2,
            **styles[strategy],
        )
        ax1.fill_between(
            theta_deviations,
            bias[strategy] - std[strategy],
            bias[strategy] + std[strategy],
            linestyle="None",
            lw=0,
            color=styles[strategy]["color"],
            alpha=0.2,
        )
        # ax2.plot(
        #     theta_deviations,
        #     std[strategy],
        #     linestyle="--",
        #     lw=2,
        #     **styles[strategy],
        # )

    ax1.axhline(0, color="k", linestyle="--", label="Unbiased")
    ax1.set_ylabel(f"Estimation Bias of $\hat{{{param_name.lower()}}}$")
    ax1.legend()
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    # ax1.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # for strategy in strategies:
    #     Jcur = J[strategy] * photon_budget
    #     Jinv = np.linalg.inv(Jcur)
    #     idx = np.where(np.array(used_params[strategy]) == param_index)[0]
    #     sigma_cur = np.sqrt(Jinv[idx, idx])
    #     print(sigma_cur, std[strategy][12], strategy)
    #     ax2.axhline(sigma_cur, color=styles[strategy]["color"])


# Main plotting block
fig, axes = plt.subplots(
    nrows=2, figsize=(TEXTWIDTH, TEXTWIDTH), sharex=True, sharey=False, dpi=DPI
)
# fig.suptitle(
#     f"Estimator Performance ({OPTIMALITY_CRITERION}-Optimality, N_est={N_estimations})",
#     fontsize=16,
# )

idx_x = 0
idx_z = 1
# Plot for X parameter on the left column
plot_performance_on_axes(axes[0], strategies, "X", idx_x)
# axes[0, 0].set_title("Bias (X-parameter)", fontsize=14)
# axes[1, 0].set_title("Precision (X-parameter)", fontsize=14)

# Plot for Z parameter on the right column
plot_performance_on_axes(axes[1], strategies, "Z", idx_z)
# axes[0, 1].set_title("Bias (Z-parameter)", fontsize=14)
# axes[1, 1].set_title("Precision (Z-parameter)", fontsize=14)
# axes[0].set_ylim(None, 7.8e-2)
# axes[1].set_ylim(None, 7.8e-2)

annotate_subplots_abc(axes)

axes[1].set_xlabel(r"Nuisance Deviation $ \phi$ (radians)")

# Final adjustments
plt.tight_layout(rect=[0, 0, 1, 0.96])
figname = f"est_with_nuisance_PB-{photon_budget:.0e}_Nest-{N_estimations:.0e}_{optimality_criterion}"
plt.savefig(save_folder / (figname + ".pdf"))
plt.savefig(save_folder / (figname + ".png"))
plt.show()
# %% Plotting parameters
DPI = 300
TEXTWIDTH = 5.90666  # inches
FONTSIZE = 7
TITLE_FONTSIZE = 8
AXES_LABEL_FONTSIZE = 7
LEGEND_FONTSIZE = 7
TICK_FONTSIZE = 5
SUBPLOT_ANNOTATION_FONTSIZE = 14
PHOTON_BUDGET = 1e3

GRID_KWARGS = dict(which="both", linestyle=":")

save_plots = False
save_folder = Path(f"../figures/{project_name}/")
save_folder.mkdir(parents=True, exist_ok=True)

plt.rcdefaults()
plt.rcParams.update(
    {
        "font.size": FONTSIZE,  # Tiny but legible (standard for subplots)
        "axes.labelsize": AXES_LABEL_FONTSIZE,  # Slightly larger for hierarchy
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "axes.labelpad": 2,  # Pull labels closer to axis
        "lines.linewidth": 1.0,
        "xtick.major.pad": 1,  # Pull ticks closer to axis
    }
)

# %% BIAS PLOT


def plot_bias(ax1, strategies, param_name, param_index):
    if param_name == "X":
        strategies = [
            strategy for strategy in strategies if strategy != "mis_z"
        ]
        true_pos = 0
    elif param_name == "Z":
        strategies = [
            strategy for strategy in strategies if strategy != "mis_x"
        ]
        true_pos = 0

    bias = {}
    std = {}
    for strategy in strategies:
        idx = np.where(np.array(used_params[strategy]) == param_index)[0]
        bias[strategy] = [np.mean(est[idx]) for est in all_estimates[strategy]]
        std[strategy] = [np.std(est[idx]) for est in all_estimates[strategy]]
        bias[strategy] = np.array(bias[strategy]) - true_pos
        std[strategy] = np.array(std[strategy])

    styles = {"X": dict(marker="x"), "Z": dict(marker=r".")}

    for strategy in strategies:
        # print(strategy, theta_deviations.shape, np.array(bias[strategy]).shape)
        ax1.plot(
            theta_deviations * 1000,
            bias[strategy] * 1000,
            linestyle="-",
            markersize=5,
            lw=1,
            label=f"bias ({param_name.lower()})",
            zorder=100 + param_index,
            **styles[param_name],
        )

    ax1.set_ylabel(r"Bias ($10^{{-3}}$ wavelengths)")
    # ax1.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    return bias


fig, ax = plt.subplots(figsize=(0.5 * TEXTWIDTH, 0.31 * TEXTWIDTH), dpi=DPI)
# fig, ax = plt.subplots(figsize=(3, 1.9), dpi=DPI, layout='constrained')

biasx = plot_bias(ax, ["nuisance_aware"], "X", 0)
biasz = plot_bias(ax, ["nuisance_aware"], "Z", 1)

px = np.polynomial.polynomial.Polynomial([0, 0, 6.3, -80])
px = np.polynomial.polynomial.Polynomial([0, 0, 3.3, -29])

# plt.plot(theta_deviations * 1000, px(theta_deviations) * 1000)
# plt.plot(theta_deviations * 1000, pz(theta_deviations) * 1000)

ax.axhline(0, color="k", linestyle="--", label="Unbiased", zorder=2)
ax.grid(True, zorder=1, **GRID_KWARGS)
ax.legend()

ax.set_xlabel(r"Nuisance deviation $ \phi$ ($10^{{-3}}$ radians)")
if save_plots:
    fig_path = save_folder / "est_bias_doptim_partial"
    plt.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".png"), bbox_inches="tight")
plt.show()

px = np.polynomial.polynomial.Polynomial.fit(
    theta_deviations,
    biasx["nuisance_aware"],
    deg=3,
    full=False,
    domain=(-1, 1),
)
pz = np.polynomial.polynomial.Polynomial.fit(
    theta_deviations,
    biasz["nuisance_aware"],
    deg=3,
    full=False,
    domain=(-1, 1),
)

print(px)
print(pz)
# %%
fig, axes = plt.subplots(
    nrows=2, ncols=5, figsize=(TEXTWIDTH, 0.3 * TEXTWIDTH), dpi=DPI
)
# Jmax = np.max([np.abs(J0) for J0 in J.values()])
# Jmin = np.min([np.abs(J0) for J0 in J.values()])


Sigma = {}
for key, value in J.items():
    Sigma[key] = np.linalg.inv(value)

Jmax = np.max([np.abs(J0) for J0 in J.values()])
Jmin = np.min([np.abs(J0) for J0 in J.values()])
Jmax = np.log1p(Jmax)
Jmin = np.log1p(Jmin)

Sigmamax = np.max([np.abs(Sigma0) for Sigma0 in Sigma.values()])
Sigmamin = np.min([np.abs(Sigma0) for Sigma0 in Sigma.values()])
Sigmamax = np.log1p(Sigmamax)
Sigmamin = np.log1p(Sigmamin)

strategy_names = {
    "full_fim": "Full",
    "subdet_xz": " Subblock",
    "mis_x": "Scalar (x)",
    "mis_z": "Scalar (z)",
    "nuisance_aware": "Partial",
}
param_labels = ["$x$", "$z$", r"$\phi$"]


def format_number(num):
    num_sc = f"{num:.1E}"
    mantissa, exp = num_sc.split("E")
    exp = int(exp)
    # if 0 <= exp <= 3:
    #     return f'{num:.0f}'
    return rf"${mantissa}\cdot 10^{{{exp}}}$"


def plot_matrices(axes, matrix_dict, toprow=False):

    Jmax = np.max([np.abs(J0) for J0 in matrix_dict.values()])
    Jmin = np.min([np.abs(J0) for J0 in matrix_dict.values()])
    Jmax = np.log1p(Jmax)
    Jmin = np.log1p(Jmin)

    for k, (ax, strategy) in enumerate(zip(axes, strategies)):
        ax.matshow(
            np.log1p(np.abs(matrix_dict[strategy])),
            vmin=Jmin,
            vmax=Jmax,
            cmap="magma",
        )
        for i, j in np.ndindex(matrix_dict[strategy].shape):
            value = matrix_dict[strategy][i, j]
            value_norm = (np.log1p(np.abs(value)) - Jmin) / (Jmax - Jmin)
            color = matplotlib.colormaps["plasma"](value_norm)
            # print(np.array(color))
            color = 1 - np.array(color)
            color = (
                np.arcsinh((color - 0.5) * 4000) / np.arcsinh(4000) / 2 + 0.5
            )
            color[3] = 1
            ax.text(
                i,
                j,
                format_number(value),
                ha="center",
                va="center",
                fontsize=3,
                color=color,
            )
        if toprow:
            ax.set_xticks(range(3), param_labels)
        else:
            ax.set_xticks([])
        ax.tick_params(axis="x", which="both", bottom=False)
        if k == 0:
            ax.set_yticks(range(3), param_labels)
        else:
            ax.set_yticks([])

        if toprow:
            ax.set_title(strategy_names[strategy])


plot_matrices(axes[0], J, toprow=True)
plot_matrices(axes[1], Sigma)

# %%
# for row in range(3):
#     for strategy in strategies:
#         curJ = J[strategy]
#         for value in curJ[row]:
#             # print(format_number(value), end='&')
#             print(format_number(value), end='&')
#     print('\\\\')
strategies = ["mis_x", "mis_z", "full_fim", "subdet_xz", "nuisance_aware"]
for strategy in strategies:
    curJ = J[strategy]
    for row in range(3):
        for value in curJ[row]:
            # print(format_number(value), end='&')
            print(format_number(value), end="&")
        print("\\\\    ")

print("\n" * 3)
for strategy in strategies:
    curJ = Sigma[strategy]
    for row in range(3):
        for value in curJ[row]:
            # print(format_number(value), end='&')
            print(format_number(value), end="&")
        print("\\\\    ")
    print(" " * 90)
# %%

fig, axes = plt.subplots(ncols=5)
for ax, strategy in zip(axes, strategies):
    ax.imshow(
        np.log1p(np.abs(Sigma[strategy])),
        vmin=Sigmamin,
        vmax=Sigmamax,
        cmap="magma",
    )
    print(
        "{:.3E}".format(np.abs(np.linalg.inv(J[strategy])[0, 2])),
        "x",
        strategy,
    )
    # print('{:.3E}'.format(np.abs(np.linalg.inv(J[strategy])[1, 2])), "z", strategy)
    ax.set_title(strategy)

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
print(f"{'':>8} | {'x':^12} | {'z':^12} | {'theta':^12}")
print("-" * 50)
for i, row in enumerate(J_to_print):
    print(
        f"{param_names[i]:>8} | {row[0]:>12.3e} | {row[1]:>12.3e} | {row[2]:>12.3e}"
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


# %% 10) Data Preparation for XZ Deviation Impact
# (Optimal vs. Random with Fair Estimators for X and Z)

# Settings for the XZ deviation analysis
N_xz_deviations = 41
max_xz_deviation = 0.012
x_dev_range = np.linspace(-max_xz_deviation, max_xz_deviation, N_xz_deviations)
z_dev_range = np.linspace(-max_xz_deviation, max_xz_deviation, N_xz_deviations)

# --- 1. Components for Nuisance-Aware Optimized Wavefront ---
wavefront_key_optimal = f"NuisanceAware_{OPTIMALITY_CRITERION}opt_Marginal"
wf_optimal = optimized_wavefronts[wavefront_key_optimal]
E0_estimator_optimal = E0_nuisance_aware
dE_vectors_estimator_optimal = dE_nuisance_aware
J_inv_estimator_optimal = J_inv_nuisance_aware
absolute_bias_grid_optimal_x = np.zeros((N_xz_deviations, N_xz_deviations))
absolute_bias_grid_optimal_z = np.zeros((N_xz_deviations, N_xz_deviations))

# --- 2. Components for Flux-Normalized Random Wavefront ---
np.random.seed(5)  # for reproducibility
wf_random = np.random.randn(number_of_input_channels) + 1j * np.random.randn(
    number_of_input_channels
)
wf_random = optimizer.normalize_wavefront(wf_random, photon_budget)
absolute_bias_grid_random_x = np.zeros((N_xz_deviations, N_xz_deviations))
absolute_bias_grid_random_z = np.zeros((N_xz_deviations, N_xz_deviations))

### plane wave for testing
# wf_random = np.zeros_like(wf_random)
# wf_random[len(wf_random)//2 + 20] = 1
# wf_random = optimizer.normalize_wavefront(wf_random, photon_budget)

# Calibrate an estimator specifically for the random wavefront
print("\nCalibrating a separate, fair estimator for the random wavefront...")
# a) Calculate the nominal field E0 for the random wavefront
# sim.scatterers = (
#     np.hstack((frozen_dipoles_line, movable_dipoles_line))
#     if frozen_dipoles_line is not None
#     else movable_dipoles_line
# )
sim.set_scatterers(
    movable_scatterers=movable_dipoles_line,
    frozen_scatterers=frozen_dipoles_line,
)
sim.reset()
sim.set_input_field_kx(kx_array, wf_random)
E0_estimator_random = sim.compute_output_fields_direct()
# b) Calculate the derivative vectors for the random wavefront
dE_vectors_estimator_random = [
    dS_tensor[i] @ wf_random for i in range(len(param_names))
]
# c) Calculate the Fisher Matrix and its inverse for the random wavefront
J_matrix_random = optimizer.J_fisher_info_matrix(wf_random)
J_inv_estimator_random = np.linalg.inv(J_matrix_random)

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
        # sim.scatterers = (
        #     np.hstack((frozen_dipoles_line, current_movable_dipoles))
        #     if frozen_dipoles_line is not None
        #     else current_movable_dipoles
        # )
        sim.set_scatterers(
            movable_scatterers=current_movable_dipoles,
            frozen_scatterers=frozen_dipoles_line,
        )

        # --- Process Optimal Wavefront ---
        sim.reset()
        sim.set_input_field_kx(kx_array, wf_optimal)
        E_true_optimal = sim.compute_output_fields_direct()
        Z_noisy_optimal = measure_heterodyne(
            E_true_optimal,
            photon_budget,
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
        absolute_bias_grid_optimal_x[i, j] = np.abs(
            np.mean(estimates_optimal[:, param_names.index("shiftX")]) - x_dev
        )
        absolute_bias_grid_optimal_z[i, j] = np.abs(
            np.mean(estimates_optimal[:, param_names.index("shiftZ")]) - z_dev
        )

        # --- Process Random Wavefront ---
        sim.reset()
        sim.set_input_field_kx(kx_array, wf_random)
        E_true_random = sim.compute_output_fields_direct()
        Z_noisy_random = measure_heterodyne(
            E_true_random,
            photon_budget,
            nsamples=N_estimations,
            seed=(i * N_xz_deviations + j) + N_xz_deviations**2,
        )
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
        absolute_bias_grid_random_x[i, j] = np.abs(
            np.mean(estimates_random[:, param_names.index("shiftX")]) - x_dev
        )
        absolute_bias_grid_random_z[i, j] = np.abs(
            np.mean(estimates_random[:, param_names.index("shiftZ")]) - z_dev
        )

print("\nXZ deviation simulation for both wavefronts finished.")


# %% 11) Combined Contour Plot of Absolute Estimation Bias for X and Z

# Prepare meshgrid and data for plotting
X_dev_grid, Z_dev_grid = np.meshgrid(x_dev_range, z_dev_range)

# Data for X-parameter plot
plot_data_optimal_x = absolute_bias_grid_optimal_x.T
plot_data_random_x = absolute_bias_grid_random_x.T
performance_difference_x = plot_data_random_x - plot_data_optimal_x

# Data for Z-parameter plot
plot_data_optimal_z = absolute_bias_grid_optimal_z.T
plot_data_random_z = absolute_bias_grid_random_z.T
performance_difference_z = plot_data_random_z - plot_data_optimal_z


fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# --- Left Subplot: X-parameter ---
ax_x = axes[0]
vmax_diff = 0.0001
pcm_x = ax_x.pcolormesh(
    X_dev_grid,
    Z_dev_grid,
    performance_difference_x,
    cmap="RdYlGn",
    shading="gouraud",
    vmin=-vmax_diff,
    vmax=vmax_diff,
)
fig.colorbar(
    pcm_x,
    ax=ax_x,
    extend="both",
    label=r"Bias Reduction for $\hat{x}$: $\epsilon_{random} - \epsilon_{optimal}$ [$\lambda$]",
)

contour_levels = [0.001, 0.01]
fmt_dict = {level: f"{level:.3f} $\\lambda$" for level in contour_levels}

cs_optimal_x = ax_x.contour(
    X_dev_grid,
    Z_dev_grid,
    plot_data_optimal_x,
    levels=contour_levels,
    colors="grey",
    linewidths=2.5,
    linestyles="solid",
)
ax_x.clabel(
    cs_optimal_x, inline=True, fontsize=12, fmt=fmt_dict, colors="grey"
)
cs_random_x = ax_x.contour(
    X_dev_grid,
    Z_dev_grid,
    plot_data_random_x,
    levels=contour_levels,
    colors="black",
    linewidths=2.5,
    linestyles="dashed",
)
ax_x.clabel(
    cs_random_x, inline=True, fontsize=12, fmt=fmt_dict, colors="black"
)

ax_x.set_xlabel(
    r"True Deviation in x-position ($\Delta x$) [$\lambda$]", fontsize=14
)
ax_x.set_ylabel(
    r"True Deviation in z-position ($\Delta z$) [$\lambda$]", fontsize=14
)
ax_x.set_title(
    r"Absolute Estimation Bias ($\epsilon$) for $\hat{x}$", fontsize=16
)
ax_x.axhline(0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
ax_x.axvline(0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
ax_x.scatter(
    [0], [0], marker="o", color="magenta", s=80, zorder=5, edgecolors="black"
)
ax_x.set_aspect("equal", adjustable="box")
ax_x.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)


# --- Right Subplot: Z-parameter ---
ax_z = axes[1]
pcm_z = ax_z.pcolormesh(
    X_dev_grid,
    Z_dev_grid,
    performance_difference_z,
    cmap="RdYlGn",
    shading="gouraud",
    vmin=-vmax_diff,
    vmax=vmax_diff,
)
fig.colorbar(
    pcm_z,
    ax=ax_z,
    extend="both",
    label=r"Bias Reduction for $\hat{z}$: $\epsilon_{random} - \epsilon_{optimal}$ [$\lambda$]",
)

cs_optimal_z = ax_z.contour(
    X_dev_grid,
    Z_dev_grid,
    plot_data_optimal_z,
    levels=contour_levels,
    colors="grey",
    linewidths=2.5,
    linestyles="solid",
)
ax_z.clabel(
    cs_optimal_z, inline=True, fontsize=12, fmt=fmt_dict, colors="grey"
)
cs_random_z = ax_z.contour(
    X_dev_grid,
    Z_dev_grid,
    plot_data_random_z,
    levels=contour_levels,
    colors="black",
    linewidths=2.5,
    linestyles="dashed",
)
ax_z.clabel(
    cs_random_z, inline=True, fontsize=12, fmt=fmt_dict, colors="black"
)

ax_z.set_xlabel(
    r"True Deviation in x-position ($\Delta x$) [$\lambda$]", fontsize=14
)
ax_z.set_ylabel(
    r"True Deviation in z-position ($\Delta z$) [$\lambda$]", fontsize=14
)
ax_z.set_title(
    r"Absolute Estimation Bias ($\epsilon$) for $\hat{z}$", fontsize=16
)
ax_z.axhline(0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
ax_z.axvline(0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
ax_z.scatter(
    [0], [0], marker="o", color="magenta", s=80, zorder=5, edgecolors="black"
)
ax_z.set_aspect("equal", adjustable="box")
ax_z.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)


# Create a single legend for the figure
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="grey", lw=2.5, linestyle="-", label=f"Optimal WF"),
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
fig.legend(
    handles=legend_elements,
    loc="upper center",
    ncol=3,
    fontsize=12,
    bbox_to_anchor=(0.5, 0.98),
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)  # Adjust rect to make space for the legend
plt.show()


# %% 12) Data Preparation for Statistical Ensemble Analysis (incl. Plane Waves)

# --- Settings for the ensemble analysis ---
N_random_waves_ensemble = 100
N_plane_waves_ensemble = (
    number_of_input_channels  # Use all available plane waves
)
N_points_on_ring = 36
deviation_radius = 0.05

# Generate the points on the circle for averaging the bias
thetas_ring = np.linspace(0, 2 * np.pi, N_points_on_ring, endpoint=False)
x_dev_ring = deviation_radius * np.cos(thetas_ring)
z_dev_ring = deviation_radius * np.sin(thetas_ring)

# --- 1. Calculate Robustness Metric for the Optimal Wavefront ---
wavefront_key_optimal = f"NuisanceAware_{OPTIMALITY_CRITERION}opt_Marginal"
wf_optimal = optimized_wavefronts[wavefront_key_optimal]
E0_estimator_optimal = E0_nuisance_aware
dE_vectors_estimator_optimal = dE_nuisance_aware
J_inv_estimator_optimal = J_inv_nuisance_aware

bias_values_optimal_x, bias_values_optimal_z = [], []
print(
    f"\nCalculating robustness metric for the optimal wavefront: {wavefront_key_optimal}..."
)
for i in range(N_points_on_ring):
    x_dev, z_dev = x_dev_ring[i], z_dev_ring[i]
    current_movable_dipoles = placedipoles.shift_and_rotate_dipoles(
        movable_dipoles_line, x_dev, z_dev, 0.0
    )
    # sim.scatterers = (
    #     np.hstack((frozen_dipoles_line, current_movable_dipoles))
    #     if frozen_dipoles_line is not None
    #     else current_movable_dipoles
    # )
    sim.set_scatterers(
        movable_scatterers=current_movable_dipoles,
        frozen_scatterers=frozen_dipoles_line,
    )
    sim.reset()
    sim.set_input_field_kx(kx_array, wf_optimal)
    E_true = sim.compute_output_fields_direct()
    Z_noisy = measure_heterodyne(
        E_true, photon_budget, nsamples=N_estimations, seed=i
    )
    estimates = np.array(
        [
            best_estimate_multi(
                Z,
                E0_estimator_optimal,
                dE_vectors_estimator_optimal,
                J_inv_estimator_optimal,
            )
            for Z in Z_noisy
        ]
    )
    bias_values_optimal_x.append(
        np.abs(np.mean(estimates[:, param_names.index("shiftX")]) - x_dev)
    )
    bias_values_optimal_z.append(
        np.abs(np.mean(estimates[:, param_names.index("shiftZ")]) - z_dev)
    )

avg_bias_optimal_x = np.mean(bias_values_optimal_x)
avg_bias_optimal_z = np.mean(bias_values_optimal_z)
print(
    f"  -> Average absolute bias for optimal WF: x={avg_bias_optimal_x:.4f}, z={avg_bias_optimal_z:.4f}"
)


# --- 2. Calculate Robustness Metric for the Ensemble of Random Wavefronts ---
avg_biases_random_x, avg_biases_random_z = [], []
print(
    f"\nCalculating robustness metrics for an ensemble of {N_random_waves_ensemble} random wavefronts..."
)
for k in range(N_random_waves_ensemble):
    print(
        f"\rProcessing random wavefront {k+1}/{N_random_waves_ensemble}",
        end="",
    )

    # Create and calibrate the random wavefront
    np.random.seed(k)  # Use loop index for seed
    wf_random = np.random.randn(
        number_of_input_channels
    ) + 1j * np.random.randn(number_of_input_channels)
    wf_random = optimizer.normalize_wavefront(wf_random, photon_budget)
    sim.reset()
    sim.set_input_field_kx(kx_array, wf_random)
    E0_estimator_random = sim.compute_output_fields_direct()
    dE_vectors_estimator_random = [
        dS_tensor[i] @ wf_random for i in range(len(param_names))
    ]
    J_inv_estimator_random = np.linalg.inv(
        optimizer.J_fisher_info_matrix(wf_random)
    )
    bias_values_random_x_run, bias_values_random_z_run = [], []
    for i in range(N_points_on_ring):
        x_dev, z_dev = x_dev_ring[i], z_dev_ring[i]
        current_movable_dipoles = placedipoles.shift_and_rotate_dipoles(
            movable_dipoles_line, x_dev, z_dev, 0.0
        )
        sim.scatterers = (
            np.hstack((frozen_dipoles_line, current_movable_dipoles))
            if frozen_dipoles_line is not None
            else current_movable_dipoles
        )
        sim.reset()
        sim.set_input_field_kx(kx_array, wf_random)
        E_true = sim.compute_output_fields_direct()
        Z_noisy = measure_heterodyne(
            E_true, photon_budget, nsamples=N_estimations, seed=k * 1000 + i
        )
        estimates = np.array(
            [
                best_estimate_multi(
                    Z,
                    E0_estimator_random,
                    dE_vectors_estimator_random,
                    J_inv_estimator_random,
                )
                for Z in Z_noisy
            ]
        )

        bias_values_random_x_run.append(
            np.abs(np.mean(estimates[:, param_names.index("shiftX")]) - x_dev)
        )
        bias_values_random_z_run.append(
            np.abs(np.mean(estimates[:, param_names.index("shiftZ")]) - z_dev)
        )

    avg_biases_random_x.append(np.mean(bias_values_random_x_run))
    avg_biases_random_z.append(np.mean(bias_values_random_z_run))


# --- 3. Calculate Robustness Metric for the Ensemble of Plane Waves ---
avg_biases_planewave_x, avg_biases_planewave_z = [], []
print(
    f"\nCalculating robustness metrics for an ensemble of {N_plane_waves_ensemble} plane waves..."
)
for k in range(N_plane_waves_ensemble):
    print(f"\rProcessing plane wave {k+1}/{N_plane_waves_ensemble}", end="")
    wf_plane = np.zeros(number_of_input_channels, dtype=complex)
    wf_plane[k] = 1.0
    wf_plane = optimizer.normalize_wavefront(wf_plane, photon_budget)
    sim.reset()
    sim.set_input_field_kx(kx_array, wf_plane)
    E0_estimator_plane = sim.compute_zL_fields_direct()
    dE_vectors_estimator_plane = [
        dS_tensor[i] @ wf_plane for i in range(len(param_names))
    ]
    J_inv_estimator_plane = np.linalg.inv(
        optimizer.J_fisher_info_matrix(wf_plane)
    )
    bias_values_plane_x_run, bias_values_plane_z_run = [], []
    for i in range(N_points_on_ring):
        x_dev, z_dev = x_dev_ring[i], z_dev_ring[i]
        current_movable_dipoles = placedipoles.shift_and_rotate_dipoles(
            movable_dipoles_line, x_dev, z_dev, 0.0
        )
        sim.scatterers = (
            np.hstack((frozen_dipoles_line, current_movable_dipoles))
            if frozen_dipoles_line is not None
            else current_movable_dipoles
        )
        sim.reset()
        sim.set_input_field_kx(kx_array, wf_plane)
        E_true = sim.compute_output_fields_direct()
        Z_noisy = measure_heterodyne(
            E_true,
            photon_budget,
            nsamples=N_estimations,
            seed=(k + N_random_waves_ensemble) * 1000 + i,
        )
        estimates = np.array(
            [
                best_estimate_multi(
                    Z,
                    E0_estimator_plane,
                    dE_vectors_estimator_plane,
                    J_inv_estimator_plane,
                )
                for Z in Z_noisy
            ]
        )
        bias_values_plane_x_run.append(
            np.abs(np.mean(estimates[:, param_names.index("shiftX")]) - x_dev)
        )
        bias_values_plane_z_run.append(
            np.abs(np.mean(estimates[:, param_names.index("shiftZ")]) - z_dev)
        )

    avg_biases_planewave_x.append(np.mean(bias_values_plane_x_run))
    avg_biases_planewave_z.append(np.mean(bias_values_plane_z_run))

print("\nAll ensemble analyses finished.")


# %% 13) Final Statistical Comparison: Optimal vs. Random vs. Plane Wave Ensembles

# --- Plotting Control Variables ---
inset_x_position = 0.15  # Adjust from 0.0 (left) to 1.0 (right)
inset_y_position = 1.35  # Adjust from 0.0 (bottom) to 1.0 (top)
x_axis_margin_factor = 0.04  # e.g., 0.04 means a 4% margin on the left
# -----------------------------

fig, axes = plt.subplots(1, 2, figsize=(22, 9), sharey=True)
# fig.suptitle(f'Statistical Comparison of Wavefront Robustness (Radius R={deviation_radius})$\lambda$', fontsize=18)

# --- Left Subplot: Histogram for X-parameter Bias ---
ax_x = axes[0]
ax_x.hist(
    avg_biases_random_x,
    bins=30,
    color="skyblue",
    edgecolor="black",
    alpha=0.7,
    label=f"Random Ensemble (N={N_random_waves_ensemble})",
)
ax_x.hist(
    avg_biases_planewave_x,
    bins=30,
    color="salmon",
    edgecolor="black",
    alpha=0.7,
    label=f"Plane Wave Ensemble (N={N_plane_waves_ensemble})",
)
ax_x.axvline(
    avg_bias_optimal_x,
    color="black",
    linestyle="--",
    linewidth=3,
    label=f"Mean Bias of Nuisance-Aware Optimal WF\n(Bias = {avg_bias_optimal_x:.4f} $\\lambda$)",
)

mean_planewave_bias_x = np.mean(avg_biases_planewave_x)
improvement_factor_plane_x = mean_planewave_bias_x / avg_bias_optimal_x
mean_random_bias_x = np.mean(avg_biases_random_x)
improvement_factor_random_x = mean_random_bias_x / avg_bias_optimal_x

annotation_text_x = (
    f"Optimal WF Improvement:\n"
    f"---------------------------------\n"
    f"vs. Mean Plane Wave: {improvement_factor_plane_x:.2f}x\n"
    f"vs. Mean Random Wave: {improvement_factor_random_x:.2f}x"
)
ax_x.text(
    0.95,
    0.95,
    annotation_text_x,
    transform=ax_x.transAxes,
    fontsize=14,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
)

ax_x.set_xlabel(
    r"Average Absolute Bias for $\hat{x}$ on Deviation Ring [$\lambda$]",
    fontsize=14,
)
ax_x.set_ylabel("Frequency", fontsize=14)
ax_x.set_title(
    r"Robustness Distribution for $\hat{x}$ Estimation", fontsize=16
)
ax_x.legend(fontsize=12, loc="upper right", bbox_to_anchor=(0.95, 0.8))
ax_x.grid(True, which="both", linestyle=":", linewidth=0.5)

# Automatically adjust x-axis for margin
current_xlim_x = ax_x.get_xlim()
x_range_x = current_xlim_x[1] - current_xlim_x[0]
ax_x.set_xlim(left=current_xlim_x[0] - x_range_x * x_axis_margin_factor)


# --- Inset plot explaining the deviation ring ---
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ax_inset = inset_axes(
    ax_x,
    width="30%",
    height="30%",
    loc="center left",
    bbox_to_anchor=(inset_x_position, 0, 1, inset_y_position),
    bbox_transform=ax_x.transAxes,
    borderpad=0,
)

ax_inset.patch.set_facecolor("white")
ax_inset.patch.set_alpha(0.85)
for spine in ax_inset.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.5)

thetas_circle = np.linspace(0, 2 * np.pi, 100)
ax_inset.plot(
    deviation_radius * np.cos(thetas_circle),
    deviation_radius * np.sin(thetas_circle),
    "k-",
)
ax_inset.scatter([0], [0], c="k", marker="+")
ax_inset.set_title(
    f"Deviation Ring\n$R = \\sqrt{{\\Delta x^2 + \\Delta z^2}} = {deviation_radius}\\lambda$",
    fontsize=10,
)
ax_inset.set_xlabel("$\\Delta x$", fontsize=10)
ax_inset.set_ylabel("$\\Delta z$", fontsize=10)
ax_inset.set_aspect("equal", adjustable="box")
ax_inset.set_xticks([-deviation_radius, 0, deviation_radius])
ax_inset.set_yticks([-deviation_radius, 0, deviation_radius])
ax_inset.tick_params(axis="both", which="major", labelsize=8)


# --- Right Subplot: Histogram for Z-parameter Bias ---
ax_z = axes[1]
ax_z.hist(
    avg_biases_random_z,
    bins=40,
    color="lightgreen",
    edgecolor="black",
    alpha=0.7,
    label=f"Random Ensemble (N={N_random_waves_ensemble})",
)
ax_z.hist(
    avg_biases_planewave_z,
    bins=40,
    color="orchid",
    edgecolor="black",
    alpha=0.7,
    label=f"Plane Wave Ensemble (N={N_plane_waves_ensemble})",
)
ax_z.axvline(
    avg_bias_optimal_z,
    color="black",
    linestyle="--",
    linewidth=3,
    label=f"Mean Bias of Nuisance-Aware Optimal WF\n(Bias = {avg_bias_optimal_z:.4f} $\\lambda$)",
)

mean_planewave_bias_z = np.mean(avg_biases_planewave_z)
improvement_factor_plane_z = mean_planewave_bias_z / avg_bias_optimal_z
mean_random_bias_z = np.mean(avg_biases_random_z)
improvement_factor_random_z = mean_random_bias_z / avg_bias_optimal_z

annotation_text_z = (
    f"Optimal WF Improvement:\n"
    f"---------------------------------\n"
    f"vs. Mean Plane Wave: {improvement_factor_plane_z:.2f}x\n"
    f"vs. Mean Random Wave: {improvement_factor_random_z:.2f}x"
)
ax_z.text(
    0.95,
    0.95,
    annotation_text_z,
    transform=ax_z.transAxes,
    fontsize=14,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
)

ax_z.set_xlabel(
    r"Average Absolute Bias for $\hat{z}$ on Deviation Ring [$\lambda$]",
    fontsize=14,
)
ax_z.set_title(
    r"Robustness Distribution for $\hat{z}$ Estimation", fontsize=16
)
ax_z.legend(fontsize=12, loc="upper right", bbox_to_anchor=(0.95, 0.8))
ax_z.grid(True, which="both", linestyle=":", linewidth=0.5)

# Automatically adjust x-axis for margin
current_xlim_z = ax_z.get_xlim()
x_range_z = current_xlim_z[1] - current_xlim_z[0]
ax_z.set_xlim(left=current_xlim_z[0] - x_range_z * x_axis_margin_factor)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% 14) Data Preparation and Plotting for Optimal vs. ENSEMBLE AVERAGE

# --- Settings ---
N_xz_deviations = 11  # Using the 11x11 grid for a feasible computation
N_random_ensemble = 100
max_xz_deviation = 0.12
x_dev_range = np.linspace(-max_xz_deviation, max_xz_deviation, N_xz_deviations)
z_dev_range = np.linspace(-max_xz_deviation, max_xz_deviation, N_xz_deviations)

# --- 1. Pre-calibrate estimators for the entire random ensemble ---
wf_random_ensemble = []
E0_estimators_random = []
dE_estimators_random = []
J_inv_estimators_random = []

print(
    f"\nPre-calibrating estimators for an ensemble of {N_random_ensemble} random wavefronts..."
)
for k in range(N_random_ensemble):
    print(f"\rCalibrating random wavefront {k+1}/{N_random_ensemble}", end="")
    np.random.seed(k)
    wf_rand = np.random.randn(number_of_input_channels) + 1j * np.random.randn(
        number_of_input_channels
    )
    wf_rand = optimizer.normalize_wavefront(wf_rand, photon_budget)
    wf_random_ensemble.append(wf_rand)

    sim.reset()
    sim.set_input_field_kx(kx_array, wf_rand)
    E0_estimators_random.append(sim.compute_zL_fields_direct())
    dE_estimators_random.append(
        [dS_tensor[i] @ wf_rand for i in range(len(param_names))]
    )
    J_inv_estimators_random.append(
        np.linalg.inv(optimizer.J_fisher_info_matrix(wf_rand))
    )
print("\nEnsemble calibration finished.")

# --- 2. Run Grid Simulation for Both Optimal and Ensemble Average ---
absolute_bias_grid_optimal_x = np.zeros((N_xz_deviations, N_xz_deviations))
absolute_bias_grid_optimal_z = np.zeros((N_xz_deviations, N_xz_deviations))
avg_absolute_bias_grid_random_x = np.zeros((N_xz_deviations, N_xz_deviations))
avg_absolute_bias_grid_random_z = np.zeros((N_xz_deviations, N_xz_deviations))

# Get Optimal WF components
wavefront_key_optimal = f"NuisanceAware_{OPTIMALITY_CRITERION}opt_Marginal"
wf_optimal = optimized_wavefronts[wavefront_key_optimal]
E0_estimator_optimal = E0_nuisance_aware
dE_vectors_estimator_optimal = dE_nuisance_aware
J_inv_estimator_optimal = J_inv_nuisance_aware


print(
    f"\nStarting XZ deviation simulation for optimal and ensemble average..."
)
for i, x_dev in enumerate(x_dev_range):
    for j, z_dev in enumerate(z_dev_range):
        print(
            f"\rSimulating grid point ({i+1}, {j+1}) / ({N_xz_deviations}, {N_xz_deviations})",
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

        # --- Process Optimal Wavefront (re-calculating on the new grid) ---
        sim.reset()
        sim.set_input_field_kx(kx_array, wf_optimal)
        E_true_optimal = sim.compute_zL_fields_direct()
        Z_noisy_optimal = measure_heterodyne(
            E_true_optimal,
            photon_budget,
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
        absolute_bias_grid_optimal_x[i, j] = np.abs(
            np.mean(estimates_optimal[:, param_names.index("shiftX")]) - x_dev
        )
        absolute_bias_grid_optimal_z[i, j] = np.abs(
            np.mean(estimates_optimal[:, param_names.index("shiftZ")]) - z_dev
        )

        # --- Process Ensemble Average ---
        biases_at_this_point_x = []
        biases_at_this_point_z = []
        for k in range(N_random_ensemble):
            sim.reset()
            sim.set_input_field_kx(kx_array, wf_random_ensemble[k])
            E_true = sim.compute_zL_fields_direct()
            Z_noisy = measure_heterodyne(
                E_true,
                photon_budget,
                nsamples=N_estimations,
                seed=(i * N_xz_deviations + j) * N_random_ensemble + k,
            )
            estimates = np.array(
                [
                    best_estimate_multi(
                        Z,
                        E0_estimators_random[k],
                        dE_estimators_random[k],
                        J_inv_estimators_random[k],
                    )
                    for Z in Z_noisy
                ]
            )
            biases_at_this_point_x.append(
                np.abs(
                    np.mean(estimates[:, param_names.index("shiftX")]) - x_dev
                )
            )
            biases_at_this_point_z.append(
                np.abs(
                    np.mean(estimates[:, param_names.index("shiftZ")]) - z_dev
                )
            )

        avg_absolute_bias_grid_random_x[i, j] = np.mean(biases_at_this_point_x)
        avg_absolute_bias_grid_random_z[i, j] = np.mean(biases_at_this_point_z)

print("\nEnsemble grid simulation finished.")


# %% 15) plot using the newly computed data...

X_dev_grid, Z_dev_grid = np.meshgrid(x_dev_range, z_dev_range)
plot_data_optimal_x = absolute_bias_grid_optimal_x.T
plot_data_avg_random_x = avg_absolute_bias_grid_random_x.T
performance_difference_x = plot_data_avg_random_x - plot_data_optimal_x

plot_data_optimal_z = absolute_bias_grid_optimal_z.T
plot_data_avg_random_z = avg_absolute_bias_grid_random_z.T
performance_difference_z = plot_data_avg_random_z - plot_data_optimal_z

fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# --- Left Subplot: X-parameter ---
ax_x = axes[0]
vmax_diff = 0.1
pcm_x = ax_x.pcolormesh(
    X_dev_grid,
    Z_dev_grid,
    performance_difference_x,
    cmap="RdYlGn",
    shading="gouraud",
    vmin=-vmax_diff,
    vmax=vmax_diff,
)
fig.colorbar(
    pcm_x,
    ax=ax_x,
    extend="both",
    label=r"Bias Reduction for $\hat{x}$: $\epsilon_{avg\_random} - \epsilon_{optimal}$ [$\lambda$]",
)
contour_levels = [0.001, 0.01]
fmt_dict = {level: f"{level:.3f} $\\lambda$" for level in contour_levels}
cs_optimal_x = ax_x.contour(
    X_dev_grid,
    Z_dev_grid,
    plot_data_optimal_x,
    levels=contour_levels,
    colors="grey",
    linewidths=2.5,
    linestyles="solid",
)
ax_x.clabel(
    cs_optimal_x, inline=True, fontsize=12, fmt=fmt_dict, colors="grey"
)
cs_random_x = ax_x.contour(
    X_dev_grid,
    Z_dev_grid,
    plot_data_avg_random_x,
    levels=contour_levels,
    colors="black",
    linewidths=2.5,
    linestyles="dashed",
)
ax_x.clabel(
    cs_random_x, inline=True, fontsize=12, fmt=fmt_dict, colors="black"
)
ax_x.set_xlabel(
    r"True Deviation in x-position ($\Delta x$) [$\lambda$]", fontsize=14
)
ax_x.set_ylabel(
    r"True Deviation in z-position ($\Delta z$) [$\lambda$]", fontsize=14
)
ax_x.set_title(
    r"Absolute Estimation Bias ($\epsilon$) for $\hat{x}$", fontsize=16
)
ax_x.axhline(0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
ax_x.axvline(0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
ax_x.scatter(
    [0], [0], marker="o", color="magenta", s=80, zorder=5, edgecolors="black"
)
ax_x.set_aspect("equal", adjustable="box")
ax_x.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

# --- Right Subplot: Z-parameter ---
ax_z = axes[1]
pcm_z = ax_z.pcolormesh(
    X_dev_grid,
    Z_dev_grid,
    performance_difference_z,
    cmap="RdYlGn",
    shading="gouraud",
    vmin=-vmax_diff,
    vmax=vmax_diff,
)
fig.colorbar(
    pcm_z,
    ax=ax_z,
    extend="both",
    label=r"Bias Reduction for $\hat{z}$: $\epsilon_{avg\_random} - \epsilon_{optimal}$ [$\lambda$]",
)
cs_optimal_z = ax_z.contour(
    X_dev_grid,
    Z_dev_grid,
    plot_data_optimal_z,
    levels=contour_levels,
    colors="grey",
    linewidths=2.5,
    linestyles="solid",
)
ax_z.clabel(
    cs_optimal_z, inline=True, fontsize=12, fmt=fmt_dict, colors="grey"
)
cs_random_z = ax_z.contour(
    X_dev_grid,
    Z_dev_grid,
    plot_data_avg_random_z,
    levels=contour_levels,
    colors="black",
    linewidths=2.5,
    linestyles="dashed",
)
ax_z.clabel(
    cs_random_z, inline=True, fontsize=12, fmt=fmt_dict, colors="black"
)
ax_z.set_xlabel(
    r"True Deviation in x-position ($\Delta x$) [$\lambda$]", fontsize=14
)
ax_z.set_ylabel(
    r"True Deviation in z-position ($\Delta z$) [$\lambda$]", fontsize=14
)
ax_z.set_title(
    r"Absolute Estimation Bias ($\epsilon$) for $\hat{z}$", fontsize=16
)
ax_z.axhline(0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
ax_z.axvline(0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
ax_z.scatter(
    [0], [0], marker="o", color="magenta", s=80, zorder=5, edgecolors="black"
)
ax_z.set_aspect("equal", adjustable="box")
ax_z.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

# --- Final Figure Legend ---
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="grey", lw=2.5, linestyle="-", label="Optimal WF"),
    Line2D(
        [0],
        [0],
        color="black",
        lw=2.5,
        linestyle="--",
        label="Ensemble Average Random WF",
    ),
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
fig.legend(
    handles=legend_elements,
    loc="upper center",
    ncol=3,
    fontsize=12,
    bbox_to_anchor=(0.5, 0.98),
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
