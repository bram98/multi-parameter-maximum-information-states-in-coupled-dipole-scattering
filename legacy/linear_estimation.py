import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.conventions import param_names
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction, LossFunctionsDict
from scatteringlibs.matplotlibmagic import PlotGroup
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

sys.exit()


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


def best_estimate_multi(E_noisy, E0, dE_dtheta, J_inv):
    """
    Optimal multi-parameter linear estimator. For the model
        x = H theta + w
    with x, H complex and theta real, the best estimator is given by
        theta = (Re(H^\dagger H))^{-1} . Re(H^\dagger x).
    Note that the part of which the inverse taken is the Fisher matrix, which is
    already real.
    """
    H = np.asarray(dE_dtheta).T
    delta_E = E_noisy - E0
    theta_est_vector = np.real(J_inv) @ np.real(H.conj().T @ delta_E)
    return theta_est_vector


def best_estimate(E, dE_dtheta, E0):
    """
    "Naive" single-parameter estimator.
    """
    theta_est = np.vecdot(dE_dtheta, E - E0) / np.vecdot(dE_dtheta, dE_dtheta)
    theta_est = np.real(theta_est)
    return theta_est


# %% 0) settings
OPTIMALITY_CRITERION = "D"  # Can be 'D', 'A', or 'E'

verbose_plotting = False
max_angle = 80
number_of_input_channels = 100
number_of_output_channels = 300
param_names = ["shiftX", "shiftZ", "rotation"]
delta_param = 1e-3
photon_budget = 1.0
multi_run_count = 5
single_run_iterations = 4000


# %%
project_name = "quasirandom_shortrun_dist2-8"
data_folder = f"../data/{project_name}"

sim = CoupledDipoleSimulator.load(folder=data_folder)
fisher_calc = FisherMatrix.load(folder=data_folder)
loss_functions_dict = LossFunctionsDict.load(folder=data_folder)

# %%
### Quasi-random double line
movable_dipoles_line = placedipoles.place_quasi_random_line_of_dipoles(
    center=(5.2, 0.0), number=6, small_dist=0.786, start_index=6
)
frozen_dipoles_line = placedipoles.place_quasi_random_line_of_dipoles(
    center=(3.0, 0.0), number=6, small_dist=0.786, start_index=26
)

# %% 6) Run simulation with the multi-parameter estimator
total_photons = 1e6
N_estimations = int(1e3)
theta0_rad = 0.0
x_true_pos = 0.0
z_true_pos = 0.0
N_deviations = 25
max_nuis_deviation = 0.1
phi_deviations = np.linspace(
    -max_nuis_deviation, max_nuis_deviation, N_deviations
)


# Lists to store results for each strategy
all_estimates_nuisance_aware = []
all_estimates_subdet_xz = []
all_estimates_full_fim = []
all_estimates_mis_x = []

print("\nPre-calculating estimator components...")
# sim = CoupledDipoleSimulator(**sim_kwargs)
nominal_movable_dipoles = placedipoles.shift_and_rotate_dipoles(
    movable_dipoles_line, 0, 0, 0
)
nominal_movable_dipoles = movable_dipoles_line.copy()
sim.set_scatterers(
    movable_scatterers=nominal_movable_dipoles,
    frozen_scatterers=frozen_dipoles_line,
)


if verbose_plotting:
    sim.visualize_intensity()
    sim.reset()

# Define keys based on optimality criterion
nuisance_aware_wf_key = f"NuisanceAware_{OPTIMALITY_CRITERION}opt_Marginal"
subdet_xz_wf_key = f"Subdet_XZ_{OPTIMALITY_CRITERION}opt"
full_fim_wf_key = f"FullFIM_{OPTIMALITY_CRITERION}opt"

# # --- Estimator Components Setup ---
# # Nuisance-Aware (Marginal FIM)
# print(f"\nSetting up components for: {nuisance_aware_wf_key}")
# wf_nuisance_aware = optimized_wavefronts[nuisance_aware_wf_key]
# dE_nuisance_aware = [
#     dS_tensor[i] @ wf_nuisance_aware for i in range(len(param_names))
# ]
# sim.set_input_field_kx(kx_array, wf_nuisance_aware)
# E0_nuisance_aware = sim.compute_zL_fields_direct()
# J_nuisance_aware = fisher_matrices_for_wavefronts[nuisance_aware_wf_key]
# J_inv_nuisance_aware = np.linalg.inv(J_nuisance_aware)

# # Sub-determinant XZ
# print(f"Setting up components for: {subdet_xz_wf_key}")
# wf_subdet_xz = optimized_wavefronts[subdet_xz_wf_key]
# dE_subdet_xz_est = [
#     dS_tensor[param_names.index("shiftX")] @ wf_subdet_xz,
#     dS_tensor[param_names.index("shiftZ")] @ wf_subdet_xz,
# ]
# sim.set_input_field_kx(kx_array, wf_subdet_xz)
# E0_subdet_xz = sim.compute_zL_fields_direct()
# J_subdet_xz_full = fisher_matrices_for_wavefronts[subdet_xz_wf_key]
# idx_x, idx_z = param_names.index("shiftX"), param_names.index("shiftZ")
# J_subdet_xz_2x2 = J_subdet_xz_full[np.ix_([idx_x, idx_z], [idx_x, idx_z])]
# J_inv_subdet_xz_2x2 = np.linalg.inv(J_subdet_xz_2x2)

# # Full FIM
# print(f"Setting up components for: {full_fim_wf_key}")
# wf_full_fim = optimized_wavefronts[full_fim_wf_key]
# dE_full_fim = [dS_tensor[i] @ wf_full_fim for i in range(len(param_names))]
# sim.set_input_field_kx(kx_array, wf_full_fim)
# E0_full_fim = sim.compute_zL_fields_direct()
# J_full_fim = fisher_matrices_for_wavefronts[full_fim_wf_key]
# J_inv_full_fim = np.linalg.inv(J_full_fim)

# # MIS_x
# print("Setting up components for: MIS_x")
# wf_mis_x = optimized_wavefronts["MIS_x"]
# dE_mis_x = dS_tensor[param_names.index("shiftX")] @ wf_mis_x
# sim.set_input_field_kx(kx_array, wf_mis_x)
# E0_mis_x = sim.compute_zL_fields_direct()
# J_mis_x_full = fisher_matrices_for_wavefronts["MIS_x"]


def estimate_params_varying_nuisance(
    sim: CoupledDipoleSimulator,
    loss_functions_dict: LossFunctionsDict,
    dS: np.ndarray,
    initial_movable_dipoles: list,
    phi_deviations: np.ndarray,
    x_true: float = 0.0,
    z_true: float = 0.0,
    total_photons: float = 1e6,
    N_estimations: int = 1000,
    verbose: bool = True,
):
    """
    Vary the nuisance parameter (phi) over the range phi_deviations. Perform
    multi-parameter estimations for each value and return the result in a dict.
    """
    global E0_dict, dE_dtheta_dict
    estimates_dict = {}
    E0_dict = {}
    dE_dtheta_dict = {}

    if verbose:
        print("Initializing multi-parameter estimation")
    sim.set_scatterers(
        movable_scatterers=initial_movable_dipoles,
        frozen_scatterers=frozen_dipoles_line,
    )
    sim.reset()

    for name, loss_function in loss_functions_dict.items():

        # Initialize estimates as numpy array with correct shape
        estimates_dict[name] = np.zeros(
            (len(phi_deviations), N_estimations, 3)
        )

        # Initialize E0: the field at zero nuisance
        sim.set_input_field_amplitudes(loss_function.wavefront)
        E0_dict[name] = sim.compute_zL_fields_direct()

        # Initialize dE_dtheta: the gradient of the field with respect to the
        # parameters at zero nuisance
        dE_dtheta_dict[name] = dS @ loss_function.wavefront

    for i, phi_true in enumerate(phi_deviations):
        if verbose:
            print(
                f"\rSimulating deviation {i+1}/{len(phi_deviations)}: {phi_true:.4f} rad",
                end="     ",
            )

        true_movable_dipoles = placedipoles.shift_and_rotate_dipoles(
            initial_movable_dipoles, x_true, z_true, phi_true
        )

        sim.set_scatterers(
            movable_scatterers=true_movable_dipoles,
            frozen_scatterers=frozen_dipoles_line,
        )

        sim.reset()

        for j, (name, loss_function) in enumerate(loss_functions_dict.items()):
            sim.set_input_field_amplitudes(loss_function.wavefront)
            E_true = sim.compute_zL_fields_direct()
            E_noisy = measure_heterodyne(
                E_true, total_photons, nsamples=N_estimations, seed=i * 10 + j
            )
            cur_estimate = [
                best_estimate_multi(
                    E,
                    E0_dict[name],
                    dE_dtheta_dict[name],
                    np.linalg.inv(loss_function.J),
                )
                for E in E_noisy
            ]
            estimates_dict[name][i] = cur_estimate
    return estimates_dict


def estimate_crlb_varying_nuisance(
    loss_functions_dict: LossFunctionsDict,
    sim: CoupledDipoleSimulator,
    phi_deviations,
    idx1: int = 0,
    idx2: int = 1,
    optimizer_device: str = "cuda",
    verbose: bool = True,
):
    crlb_estimates = {}

    sim_kwargs = sim.get_kwargs()
    angles_deg = sim.theta_inc_deg

    for name in loss_functions_dict.keys():
        crlb_estimates[name] = []

    for i, phi_dev in enumerate(phi_deviations):
        if verbose:
            print(
                f"\rCalculating CRLB for deviation {i+1}/{len(phi_deviations)}: {phi_dev:.4f} rad",
                end="     ",
            )
        # Recalculate dS for the current deviation
        dS_current = placedipoles.calculate_dS_for_3_parameters(
            sim_kwargs,
            angles_deg,
            movable_dipoles=placedipoles.shift_and_rotate_dipoles(
                movable_dipoles_line, 0, 0, phi_dev
            ),
            frozen_dipoles=frozen_dipoles_line,
            delta_param=delta_param,
            flux_normalize=True,
        )
        fisher_calc_dev = FisherMatrix(
            dS=dS_current, param_names=param_names, delta=delta_param
        )
        optimizer_dev = Optimizer(
            F_operator_tensor=fisher_calc_dev.F, device=optimizer_device
        )
        for name, loss_function in loss_functions_dict.items():
            J_dev = optimizer_dev.J_fisher_info_matrix(loss_function.wavefront)
            J_inv_dev = np.linalg.inv(J_dev)
            crlb_estimates[name].append(np.sqrt(np.diag(J_inv_dev)))
    for name in loss_functions_dict.keys():
        crlb_estimates[name] = np.array(crlb_estimates[name])
    return crlb_estimates


# %%


def plot_bias(
    ax,
    plot_groups: list[PlotGroup],
    phi_deviations: np.ndarray,
    param_idx: int,
):
    for plot_group in plot_groups:
        crlb = plot_group.data[:, param_idx]
        axes[0].plot(
            phi_deviations,
            crlb,
            marker=plot_group.marker,
            linestyle="-",
            label=plot_group.legend_label,
            color=plot_group.color,
            markersize=8,
            lw=2,
        )
    axes[0].legend()


def plot_crlb(
    ax,
    crlbs: dict,
    styles: dict,
    phi_deviations: np.ndarray,
    param_idx: int,
):
    print(styles.keys())
    print(crlbs.keys())
    for name, style in styles.items():
        crlb = crlbs[name][:, param_idx]
        # style = styles.get(name) or {}
        ax.plot(
            phi_deviations,
            crlb,
            linestyle="-",
            markersize=8,
            lw=2,
            **style,
        )
    ax.legend()


# %%
crlb_estimates = estimate_crlb_varying_nuisance(
    loss_functions_dict=loss_functions_dict.select_only(
        ["det", "x", "z", "dropped_det", "partial_det"]
    ),
    sim=sim,
    phi_deviations=phi_deviations[::],
    idx1=0,
    idx2=1,
    optimizer_device="cuda",
    verbose=True,
)
# %%
crlb_styles = {
    "dropped_det": dict(label="dropped det", marker="x"),
    "det": dict(label="det", marker="^"),
    "partial_det": dict(label="partial det", marker="s"),
}

crlbs_styles_x = {"x": dict(label="x", marker=".")}
crlbs_styles_z = {"z": dict(label="z", marker=".")}
crlbs_styles_x.update(crlb_styles)
crlbs_styles_z.update(crlb_styles)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
plot_crlb(
    axes[1, 0], crlb_estimates, crlbs_styles_x, phi_deviations[::], param_idx=1
)
plot_crlb(
    axes[1, 1], crlb_estimates, crlbs_styles_z, phi_deviations[::], param_idx=1
)
plt.show()


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
idx_x = 0
idx_z = 1
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
plt.savefig("figures/nuisance_robustness.png", dpi=400)
plt.show()

# %% 8) Data Preparation for Statistical Ensemble Analysis (incl. Plane Waves)

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
    sim.set_scatterers(
        movable_scatterers=current_movable_dipoles,
        frozen_scatterers=frozen_dipoles_line,
    )
    sim.reset()
    sim.set_input_field_kx(kx_array, wf_optimal)
    E_true = sim.compute_zL_fields_direct()
    Z_noisy = measure_heterodyne(
        E_true, total_photons, nsamples=N_estimations, seed=i
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
    E0_estimator_random = sim.compute_zL_fields_direct()
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

        sim.set_scatterers(
            movable_scatterers=current_movable_dipoles,
            frozen_scatterers=frozen_dipoles_line,
        )
        sim.reset()
        sim.set_input_field_kx(kx_array, wf_random)
        E_true = sim.compute_zL_fields_direct()
        Z_noisy = measure_heterodyne(
            E_true, total_photons, nsamples=N_estimations, seed=k * 1000 + i
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
        sim.set_scatterers(
            movable_scatterers=current_movable_dipoles,
            frozen_scatterers=frozen_dipoles_line,
        )
        sim.reset()
        sim.set_input_field_kx(kx_array, wf_plane)
        E_true = sim.compute_zL_fields_direct()
        Z_noisy = measure_heterodyne(
            E_true,
            total_photons,
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


# %% 9) Final Statistical Comparison: Optimal vs. Random vs. Plane Wave Ensembles

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
plt.savefig(
    "figures/nuisance-aware_robustness_for_paramofinterest_estimation.png",
    dpi=400,
)
plt.show()
