# playground/calc_fisher_info_and_MIS_3params_refactored.py

# %% Imports and functions

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import (
    LossFunction,
    LossFunctionsDict,
    LossFunctionSweep,
    dropped_lossfunction,
    partial_lossfunction,
)
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

sys.exit()


# %%

data_folder = "../data/quasirandom"  # Where to save state
save_state = False  # If True, save state to disk
# %% 1) define dipole configurations
movable_dipoles_line = placedipoles.place_quasi_random_line_of_dipoles(
    center=(5.2, 0.0), number=6, small_dist=0.786, start_index=5
)

frozen_dipoles_line = placedipoles.place_quasi_random_line_of_dipoles(
    center=(3, 0.0), number=6, small_dist=0.786, start_index=22
)

# %% 2) simulator geometry and input space
total_Ns_sim = movable_dipoles_line.shape[1]
if frozen_dipoles_line is not None and frozen_dipoles_line.size > 0:
    total_Ns_sim += frozen_dipoles_line.shape[1]

max_angle = 80  # degrees
angles_deg = np.linspace(-max_angle, max_angle, 100)  # Input angle channels

sim_kwargs = dict(
    lambda_0=1.0,
    Lz=8.0,
    Lx=8.0,
    Ns=total_Ns_sim,  # Ns in sim_kwargs should be the total number of dipoles
    seed=123,
    grid_points_x=300,  # Number of points for field calculation (output basis for S-matrix)
    theta_inc_deg=angles_deg,
)


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
base_sim_for_vis.movable_scatterers = movable_dipoles_line
base_sim_for_vis.frozen_scatterers = frozen_dipoles_line

ampl_init = np.zeros_like(angles_deg)
ampl_init[len(angles_deg) // 2] = 1
base_sim_for_vis.set_input_field_amplitudes(ampl_init)
base_sim_for_vis.visualize_intensity(title="Dipole Configuration")

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

if save_state:
    base_sim_for_vis.save(folder=data_folder)
    fisher_calc.save(folder=data_folder)

    print(f"Saved simulator and Fisher matrix to '{data_folder}'")
# %%

loss_functions_dict = LossFunctionsDict()

# Single parameters
for i, param in enumerate(["x", "z", r"\phi"]):
    name = param.replace("\\", "")
    loss_functions_dict[name] = LossFunction(
        lossfunctions.single_param_F(i),
        name=name,
        displayname=f"Singe parameter ${param}$",
        # shortname=rf"$\mathcal{{J}}_{{{param}{param}}}$",
        shortname=rf"${param}$ (scalar)",
        maximize=True,
        method="eig",
    )

# Single parameters using inverse
for i, param in enumerate(["x", "z", r"\phi"]):
    name = param.replace("\\", "")
    name += "_inv"
    loss_functions_dict[name] = LossFunction(
        lossfunctions.single_param_inv_J(i),
        name=name,
        displayname=f"Singe parameter ${param}$ (joint)",
        # shortname=rf"$\mathcal{{J}}^{{-1}}_{{{param}{param}}}$",
        shortname=rf"${param}$ (joint)",
        maximize=False,
        method="torch",
        steps=5000,
        lr=1e-1,
        lr_decay=0.2,
        lr_decay_step_size=500,
        multiruns=3,
    )

# Trace
loss_functions_dict["tr"] = LossFunction(
    lossfunctions.trace_of_F,
    "tr",
    displayname="Trace",
    shortname=r"$\operatorname{tr}J$",
    maximize=True,
    method="eig",
)


loss_functions_dict["tr_norm"] = LossFunction(
    lossfunctions.normalized_trace_of_F,
    "tr_norm",
    displayname="Normalized Trace",
    shortname=r"$\operatorname{tr}_N J$",
    maximize=True,
    method="eig",
)

# loss_functions_dict["tr_norm_nuissance"] = LossFunction(
#     tr_norm_nuissance,
#     "tr_norm_nuissance",
#     displayname="Nuissance Trace",
#     shortname=r"$\operatorname{tr}_{N}^* J$",
#     maximize=True,
#     method="eig",
# )

# Determinant - D-optimality
loss_functions_dict["det"] = LossFunction(
    lambda J: torch.linalg.det(J),
    "det",
    displayname="D-optimality",
    shortname=r"D-Opt. ($\mathcal{J}$)",
    maximize=True,
    method="torch",
    steps=2000,
    lr=1e-2,
    lr_decay=0.2,
    lr_decay_step_size=500,
)


# Spectral radius - E-optimality
loss_functions_dict["spec_rad"] = LossFunction(
    lossfunctions.min_eigenvalue_of_J,
    "spec_rad",
    displayname="E-optimality",
    shortname=r"E-Opt. ($\mathcal{J}$)",
    maximize=True,
    method="torch",
    steps=4000,
    lr=1e-1,
    lr_decay=0.1,
    lr_decay_step_size=500,
    multiruns=None,
)

# Trace of inverse - A-optimality
loss_functions_dict["tr_inv"] = LossFunction(
    lossfunctions.minimize_trace_of_J_inv,
    "tr_inv",
    displayname="A-optimality",
    shortname=r"A-Opt. ($\mathcal{J}$)",
    maximize=False,
    method="torch",
    steps=4000,
    lr=1e-1,
    lr_decay=0.2,
    lr_decay_step_size=1000,
    multiruns=3,
)

# %%
"""
All partial fisher loss functions. Used to optimize under presence of nuisance
parameters
"""

nuisance_index = 2  # rotations are considered nuisance
parameter_of_interest_latex = r"\vartheta"
nuisance_parameter_latex = r"\xi"
partial_fisher_latex = (
    rf"{{{parameter_of_interest_latex}|{nuisance_parameter_latex}}}"
)
dropped_fisher_latex = (
    rf"{{{parameter_of_interest_latex}{parameter_of_interest_latex}}}"
)


# r"A-Opt. ($\mathcal J$)"
# Partial Determinant
loss_functions_dict["partial_det"] = LossFunction(
    partial_lossfunction(lossfunctions.determinant_of_J, nuisance_index),
    "partial_det",
    displayname="D-optimality (partial)",
    shortname=rf"D-Opt. ($ \mathcal{{J}}_{partial_fisher_latex}$)",
    maximize=True,
    method="torch",
    steps=2000,
    lr=1e-2,
    lr_decay=0.2,
    lr_decay_step_size=500,
)

# Partial Spectral Radius
loss_functions_dict["partial_spec_rad"] = LossFunction(
    partial_lossfunction(lossfunctions.min_eigenvalue_of_J, nuisance_index),
    "partial_spec_rad",
    displayname="E-optimality (partial) ",
    shortname=rf"E-Opt. ($ \mathcal{{J}}_{partial_fisher_latex}$)",
    maximize=True,
    method="torch",
    steps=4000,
    lr=1e-1,
    lr_decay=0.1,
    lr_decay_step_size=1000,
    multiruns=None,
)


# Partial Trace of inverse
loss_functions_dict["partial_tr_inv"] = LossFunction(
    partial_lossfunction(lossfunctions.trace_of_J_inv, nuisance_index),
    "partial_tr_inv",
    displayname="A-optimality (partial) ",
    shortname=rf"A-Opt. ($ \mathcal{{J}}_{partial_fisher_latex}$)",
    maximize=False,
    method="torch",
    steps=4000,
    multiruns=3,
    lr=1e-1,
    lr_decay=0.2,
    lr_decay_step_size=1000,
)


# %%
"""
All fisher loss functions where all information on the nuisance parameter is 
dropped. This is to compare with partial fisher information
"""


dropped_fisher_latex = (
    rf"{{{parameter_of_interest_latex}{parameter_of_interest_latex}}}"
)

# Single parameters
for i, param in enumerate(["x", "z"]):
    loss_functions_dict[f"dropped_{param}"] = LossFunction(
        dropped_lossfunction(lossfunctions.single_param_F(i), nuisance_index),
        f"dropped_{param}",
        displayname=f"Single parameter {param} (dropped)",
        shortname=rf"$\mathcal{{J}}_{dropped_fisher_latex}$)",
        maximize=True,
        method="torch",
        steps=2000,
    )

# dropped Determinant
loss_functions_dict["dropped_det"] = LossFunction(
    dropped_lossfunction(lossfunctions.determinant_of_J, nuisance_index),
    "dropped_det",
    displayname="D-optimality (dropped)",
    shortname=rf"D-Opt. ($\mathcal{{J}}_{dropped_fisher_latex}$)",
    maximize=True,
    method="torch",
    steps=2000,
    lr=1e-2,
    lr_decay=0.2,
    lr_decay_step_size=500,
)

# dropped Spectral Radius
loss_functions_dict["dropped_spec_rad"] = LossFunction(
    dropped_lossfunction(lossfunctions.min_eigenvalue_of_J, nuisance_index),
    "dropped_spec_rad",
    displayname="E-optimality (dropped) ",
    shortname=rf"E-Opt. ($\mathcal{{J}}_{dropped_fisher_latex}$)",
    maximize=True,
    method="torch",
    steps=4000,
    lr=1e-1,
    lr_decay=0.1,
    lr_decay_step_size=1000,
    multiruns=None,
)


# dropped Trace of inverse
loss_functions_dict["dropped_tr_inv"] = LossFunction(
    dropped_lossfunction(lossfunctions.trace_of_J_inv, nuisance_index),
    "dropped_tr_inv",
    displayname="A-optimality (dropped) ",
    shortname=rf"A-Opt. ($\mathcal{{J}}_{dropped_fisher_latex}$)",
    maximize=False,
    method="torch",
    steps=4000,
    multiruns=3,
    lr=1e-1,
    lr_decay=0.2,
    lr_decay_step_size=1000,
)


# %%
"""
Optimize
"""

loss_function: LossFunction
for loss_function in loss_functions_dict.values():
    # if ("x_inv" not in loss_function.name) or "tr_inv" in loss_function.name:
    #     continue
    loss_function.optimize(
        optimizer,
        seed=2025,
        verbose=True,
        lr=0.1,
        lr_decay=0.8,
        lr_decay_step_size=500,
        steps=10000,
        multiruns=5,
    )


# %%
"""
Check convergence by showing loss histories
"""

for loss_function in loss_functions_dict.values():
    if loss_function.method == "torch" and loss_function._J is not None:
        optimizer.plot_torch_loss(
            loss_function.loss_history,
            first_step=-500,
            title=f"{loss_function.displayname} Optimization Loss",
        )


# %%
"""
Visualize optimized wavefronts and scatterers
"""


for loss_function in loss_functions_dict.values():
    # if loss_function.name != "det":
    #     continue
    # loss_function.displayname = "Determinant"
    loss_function.visualize(
        base_sim_for_vis,
        base_sim_for_vis.k_x_array,
        title=f"Maximum Information State for {loss_function.displayname}",
        dpi=300,
    )

# %%
"""
Save results to be used in other scripts
"""

if save_state:
    loss_functions_dict.save(folder=data_folder)
    print(f"Saved loss functions to '{data_folder}'")


# %%
"""
Lossfunctions that contain a sweep over a parameter
"""
loss_functions_sweep_dict = LossFunctionsDict()

# sweep_dict["xz_inv"]


# loss_functions_sweep_dict["xz_scalar"] = LossFunctionSweep(
#     lossfunctions.sweep_xz_F,
#     name="xz_scalar",
#     displayname=f"Interpolation x-z (scalar)",
#     # shortname=rf"$\mathcal{{J}}^{{-1}}_{{{param}{param}}}$",
#     shortname=rf"${param}$ (joint)",
#     maximize=True,
#     method="eig",
#     n_sweep=50,
# )

loss_functions_sweep_dict["xz_joint"] = LossFunctionSweep(
    lossfunctions.sweep_xz_inv_J,
    name="xz_joint",
    displayname=f"Interpolation x-z (joint)",
    # shortname=rf"$\mathcal{{J}}^{{-1}}_{{{param}{param}}}$",
    shortname=rf"${param}$ (joint)",
    maximize=False,
    method="torch",
    n_sweep=50,
    steps_init=5000,
    steps=50,
    lr=1e-2,
    multiruns=3,
)
# %%
"""
Optimize sweeps
"""

loss_function: LossFunctionSweep
for loss_function in loss_functions_sweep_dict.values():
    # if ("x_inv" not in loss_function.name) or "tr_inv" in loss_function.name:
    #     continue
    loss_function.optimize(
        optimizer,
        seed=2025,
        verbose=True,
    )

# %%
"""
Check convergence by showing loss histories of sweeps
"""

for loss_function in loss_functions_sweep_dict.values():
    if loss_function.method == "torch" and loss_function._Js is not None:
        optimizer.plot_torch_loss_sweep(
            loss_function.loss_history,
            first_step=-50,
            title=f"{loss_function.displayname} Optimization Loss",
        )


# %%
ss = [
    loss_functions_sweep_dict["xz_scalar"],
    loss_functions_sweep_dict["xz_joint"],
]
for s in ss:
    J_invs = [np.linalg.inv(J) for J in s.Js]
    z = [J_inv[1, 1] for J_inv in J_invs]
    x = [J_inv[0, 0] for J_inv in J_invs]
    plt.plot(z[:], x[:])
plt.xlabel("z")
plt.ylabel("x")
plt.title(
    r"Interpolation of Scalar CRLB: $\langle E|\left((1-\alpha)F_{xx} + \alpha F_{zz}\right)|E\rangle$"
)
plt.show()
