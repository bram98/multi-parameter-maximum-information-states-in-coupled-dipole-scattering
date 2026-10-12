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
    dropped_lossfunction_F,
    partial_lossfunction_given_nuisance,
)
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

# sys.exit()


# %%

# data_folder = "../data/quasirandom_longrun"  # Where to save state
# data_folder = "../data/quasirandom_longrun_dist2-2"  # Where to save state
# data_folder = "../data/quasirandom_shortrun_dist2-8"  # Where to save state
data_folder = "../data/quasirandom_longrun_dist2-8"  # Where to save state
save_state = True  # If True, save state to disk


print(f"save_state={save_state}")
# %% Define dipole configurations
movable_dipoles = placedipoles.place_quasi_random_line_of_dipoles(
    center=(1.4, 0.0), number=6, small_dist=0.786, start_index=5
)

frozen_dipoles = placedipoles.place_quasi_random_line_of_dipoles(
    center=(-1.4, 0.0), number=6, small_dist=0.786, start_index=23
)
# movable_dipoles = placedipoles.place_square_of_dipoles(side=1.57)
# frozen_dipoles = None

# %% Simulator geometry and input space

N_modes = 45
lambda_0 = 1.0
NA_out = 0.95

critical_sampling = CoupledDipoleSimulator.calc_critical_sampling_at_N_modes(
    N_modes=N_modes,
    NA_full=0.95,
    NA_in=0.95,
    lambda_0=lambda_0,
)
Lx = critical_sampling["Lx"]
k_x_array = critical_sampling["k_x_array"]

sim = CoupledDipoleSimulator(
    lambda_0=lambda_0,
    Lx=critical_sampling["Lx"],
    Lz=critical_sampling["Lx"],
    grid_points_x=critical_sampling["grid_points_x"],
    NA_out=NA_out,
    movable_scatterers=movable_dipoles,
    frozen_scatterers=frozen_dipoles,
    k_x_array=k_x_array,
    Lz_vis=8,
    Lx_vis=8,
    grid_points_x_vis=300,
)

sim_for_vis = sim.copy()

sim_for_vis.set_input_field_kx([0], [1])
sim_for_vis.visualize_intensity(
    title="Dipole Configuration",
    movable_kwargs=dict(
        s=100, marker="o", edgecolors="r", c="none", facecolors="none"
    ),
)
print()

# %% Build partial derivatives for each param => dS[mu]
param_names = ["shiftX", "shiftZ", "rotation"]
delta_param = 1e-3

S, _ = sim.build_scattering_matrix(flux_normalize=True)

print("Calculating dS for 3 parameters...")
dS_tensor = placedipoles.calculate_dS_for_3_parameters(
    sim.get_kwargs(),
    sim.theta_inc_deg,
    movable_dipoles=movable_dipoles,
    frozen_dipoles=frozen_dipoles,
    delta_param=delta_param,
    flux_normalize=True,
)
print(f"Built dS tensor with shape: {dS_tensor.shape}")
# %% Build the Fisher-operator array and set global variables
fisher_calc = FisherMatrix(
    dS=dS_tensor, S=S, param_names=param_names, delta=delta_param
)
F_tensor_operator = fisher_calc.F

print(
    "Built F[mu,nu,:,:], shape =",
    F_tensor_operator.shape,
    f" (should be 3,3,{len(sim.theta_inc)},{len(sim.theta_inc)})",
)


# Initialize Optimizer (used for Blocks 5, 6, and 7 onwards)
optimizer_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for Optimizer: {optimizer_device}")
optimizer = Optimizer(
    F_operator_tensor=F_tensor_operator, device=optimizer_device
)

# Photon budget for normalization (can be defined once, used across sections)
photon_budget = 1.0


if save_state:
    sim.save(folder=data_folder)
    fisher_calc.save(folder=data_folder)

    print(f"Saved simulator and Fisher matrix to '{data_folder}'")
# %% Lossfunctions


loss_functions_dict = LossFunctionsDict()

# %%% Full
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
    shortname=r"D-Opt.",
    maximize=True,
    method="torch",
)


# Spectral radius - E-optimality
loss_functions_dict["spec_rad"] = LossFunction(
    lossfunctions.min_eigenvalue_of_J,
    "spec_rad",
    displayname="E-optimality",
    shortname=r"E-Opt.",
    maximize=True,
    method="torch",
)

# Trace of inverse - A-optimality
loss_functions_dict["tr_inv"] = LossFunction(
    lossfunctions.minimize_trace_of_J_inv,
    "tr_inv",
    displayname="A-optimality",
    shortname=r"A-Opt.",
    maximize=False,
    method="torch",
)

# %%% Partial
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
    partial_lossfunction_given_nuisance(
        lossfunctions.determinant_of_J, nuisance_index
    ),
    "partial_det",
    displayname="D-optimality (partial)",
    shortname=rf"D-Opt",
    maximize=True,
    method="torch",
)

# Partial Spectral Radius
loss_functions_dict["partial_spec_rad"] = LossFunction(
    partial_lossfunction_given_nuisance(
        lossfunctions.min_eigenvalue_of_J, nuisance_index
    ),
    "partial_spec_rad",
    displayname="E-optimality (partial) ",
    shortname=rf"E-Opt.",
    maximize=True,
    method="torch",
)


# Partial Trace of inverse
loss_functions_dict["partial_tr_inv"] = LossFunction(
    partial_lossfunction_given_nuisance(
        lossfunctions.trace_of_J_inv, nuisance_index
    ),
    "partial_tr_inv",
    displayname="A-optimality (partial) ",
    shortname=rf"A-Opt.",
    maximize=False,
    method="torch",
)


# %%% Dropped
"""
All fisher loss functions where all information on the nuisance parameter is 
dropped. This is to compare with partial fisher information
"""


dropped_fisher_latex = (
    rf"{{{parameter_of_interest_latex}{parameter_of_interest_latex}}}"
)


# dropped Determinant
loss_functions_dict["dropped_det"] = LossFunction(
    dropped_lossfunction(lossfunctions.determinant_of_J, nuisance_index),
    "dropped_det",
    displayname="D-optimality (dropped)",
    shortname=rf"D-Opt.",
    maximize=True,
    method="torch",
)

# dropped Spectral Radius
loss_functions_dict["dropped_spec_rad"] = LossFunction(
    dropped_lossfunction(lossfunctions.min_eigenvalue_of_J, nuisance_index),
    "dropped_spec_rad",
    displayname="E-optimality (dropped) ",
    shortname=rf"E-Opt.",
    maximize=True,
    method="torch",
)


# dropped Trace of inverse
loss_functions_dict["dropped_tr_inv"] = LossFunction(
    dropped_lossfunction(lossfunctions.trace_of_J_inv, nuisance_index),
    "dropped_tr_inv",
    displayname="A-optimality (dropped) ",
    shortname=rf"A-Opt.",
    maximize=False,
    method="torch",
)
loss_functions_dict["dropped_tr_norm"] = LossFunction(
    dropped_lossfunction_F(
        lossfunctions.normalized_trace_of_F, nuisance_index
    ),
    "dropped_tr_norm",
    displayname="Normalized Trace (dropped)",
    shortname=r"$\operatorname{tr}_N J$",
    maximize=True,
    method="eig",
)


# %% Optimize
"""
Optimize
"""

"""
Loading the 'quick' setting allows for quick optimization, but the results will
not be fully converged. Loading 'thorough' will take quite a bit longer 
(>1 hour) but are guaranteed to be converged. You can never be 100% sure it is 
converged but you know what I mean.
"""

loss_functions_dict.load_optimization_settings(
    "thorough2",
    # "quick",
    folder="../optimization_settings",
)


loss_function: LossFunction
for loss_function in loss_functions_dict.values():
    # if ("x_inv" not in loss_function.name) or "tr_inv" in loss_function.name:
    #     continue
    loss_function.optimize(
        optimizer,
        seed=2025,
        verbose=True,
    )


# %% Check Convergence
"""
Check convergence by showing loss histories
"""

for loss_function in loss_functions_dict.values():
    if loss_function.method == "torch" and loss_function._J is not None and "":
        optimizer.plot_torch_loss(
            loss_function.loss_history,
            first_step=3000,
            final_step=4000,
            title=f"{loss_function.displayname} Optimization Loss",
        )


# %% Visualize
"""
Visualize optimized wavefronts and scatterers
"""


for loss_function in loss_functions_dict.values():
    # if loss_function.name != "det":
    #     continue
    # loss_function.displayname = "Determinant"
    loss_function.visualize(
        sim_for_vis,
        k_x_array,
        title=f"Maximum Information State for {loss_function.displayname}",
        dpi=300,
    )

# %% Save loss functions
"""
Save results to be used in other scripts
"""

if save_state:
    loss_functions_dict.save(folder=data_folder)
    print(f"Saved loss functions to '{data_folder}'")


# %% Sweep loss functions
"""
Lossfunctions that contain a sweep over a parameter
"""

loss_functions_sweep_dict = LossFunctionsDict()


loss_functions_sweep_dict["xz_scalar"] = LossFunctionSweep(
    lossfunctions.sweep_xz_F,
    name="xz_scalar",
    displayname=f"Interpolation x-z (scalar)",
    shortname=rf"$x\leftrightarrow z$ (scalar)",
    maximize=True,
    method="eig",
    n_sweep=50,
)


loss_functions_sweep_dict["xz_joint"] = LossFunctionSweep(
    lossfunctions.sweep_xz_inv_J,
    name="xz_joint",
    displayname=f"Interpolation x-z (joint)",
    shortname=rf"$x\leftrightarrow z$ (joint)",
    maximize=False,
    method="torch",
)


# %% Optimize sweeps
"""
Optimize sweeps
"""

loss_functions_sweep_dict.load_optimization_settings(
    # "quick_sweep", folder="../optimization_settings"
    "thorough_sweep2",
    folder="../optimization_settings",
)

loss_function: LossFunctionSweep
for loss_function in loss_functions_sweep_dict.values():
    # if ("x_inv" not in loss_function.name) or "tr_inv" in loss_function.name:
    #     continue
    loss_function.optimize(
        optimizer,
        seed=2025,
        verbose=True,
    )

# %% Check convergence
"""
Check convergence by showing loss histories of sweeps
"""

for loss_function in loss_functions_sweep_dict.values():
    if loss_function.method == "torch" and loss_function._Js is not None:
        optimizer.plot_torch_loss_sweep(
            loss_function.loss_history,
            first_step=-1000,
            title=f"{loss_function.displayname} Optimization Loss",
        )

# %% Save sweep loss functions
"""
Save results of sweep to be used in other scripts
"""

if save_state:
    loss_functions_sweep_dict.save(
        name="LossFunctionsSweepDict", folder=data_folder
    )
    print(f"Saved sweep loss functions to '{data_folder}'")


# %%
# """
# Change names without having to run optimizier
# """
# data_folder = "../data/quasirandom_longrun"

# loss_functions_dict_load = LossFunctionsDict.load(folder=data_folder)

# for name, loss_function in loss_functions_dict.items():
#     loss_function._J = loss_functions_dict_load[name].J
#     loss_function.loss_history = loss_functions_dict_load[name].loss_history

# if save_state:
#     loss_functions_dict.save(folder=data_folder)

#     print(f"Renamed loss functions in '{data_folder}'")
