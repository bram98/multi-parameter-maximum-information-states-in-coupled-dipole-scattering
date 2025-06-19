# playground/calc_fisher_info_and_MIS_3params_refactored.py

# %% Imports and functions

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction, LossFunctionsDict
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

sys.exit()


# %%

data_folder = "../data/quasirandom"  # Where to save state
save_state = True  # If True, save state to disk
# %% 1) define dipole configurations
movable_dipoles_line = placedipoles.place_quasi_random_line_of_dipoles(
    center=(5.0, 0.0), number=6, small_dist=0.786, start_index=5
)

frozen_dipoles_line = placedipoles.place_quasi_random_line_of_dipoles(
    center=(3.0, 0.0), number=6, small_dist=0.786, start_index=22
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


# k0 = 2 * np.pi / sim_kwargs["lambda_0"]
# kx_array = k0 * np.sin(np.deg2rad(angles_deg))

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
# %%

loss_functions_dict = LossFunctionsDict()

# Single parameters
for i, param in enumerate(["x", "z", r"\theta"]):
    name = param.replace("\\", "")
    loss_functions_dict[name] = LossFunction(
        # lambda F, i_=i: F[i_, i_],
        lossfunctions.single_param_F(i),
        name=name,
        displayname=f"Singe parameter ${param}$",
        shortname=f"$J_{{{param}}}$",
        maximize=True,
        method="eig",
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


# Normalized Trace
def F_norm(F):
    eig_max = [np.linalg.eigvalsh(F[i, i])[-1] for i in range(3)]
    scale_factors = 1 / np.sqrt(eig_max)
    scale_factors = np.outer(scale_factors, scale_factors)  # Create matrix
    scale_factors = scale_factors[:, :, np.newaxis, np.newaxis]
    return F * scale_factors


def tr_norm(F):
    Fnorm = F_norm(F)
    return Fnorm[0, 0] + Fnorm[1, 1] + Fnorm[2, 2]


def tr_norm_nuissance(F):
    Fnorm = F_norm(F)
    return Fnorm[0, 0] + Fnorm[1, 1] - Fnorm[2, 2]


loss_functions_dict["tr_norm"] = LossFunction(
    lossfunctions.normalized_trace_of_F,
    "tr_norm",
    displayname="Normalized Trace",
    shortname=r"$\operatorname{tr}_N J$",
    maximize=True,
    method="eig",
)

loss_functions_dict["tr_norm_nuissance"] = LossFunction(
    tr_norm_nuissance,
    "tr_norm_nuissance",
    displayname="Nuissance Trace",
    shortname=r"$\operatorname{tr}_{N}^* J$",
    maximize=True,
    method="eig",
)

# Determinant
loss_functions_dict["det"] = LossFunction(
    lambda J: torch.linalg.det(J),
    "det",
    displayname="Determinant",
    shortname=r"$\operatorname{det} J$",
    maximize=True,
    method="torch",
    steps=2000,
)


# Spectral Radius
def spectral_radius(J):
    eigvals = torch.linalg.eigvalsh(J)
    return eigvals[0]  # return smallest eigenvalue


loss_functions_dict["spec_rad"] = LossFunction(
    spectral_radius,
    "spec_rad",
    displayname="Spectral Radius",
    shortname=r"$\rho(J)$",
    maximize=True,
    method="torch",
    steps=4000,
    multiruns=None,
)

# %%
"""
Optimize
"""

for loss_function in loss_functions_dict.values():
    loss_function.optimize(
        optimizer, seed=2025, steps=None, verbose=True, multiruns=None
    )

# %%
"""
Check convergence by showing loss histories
"""

for loss_function in loss_functions_dict.values():
    if loss_function.method == "torch":
        optimizer.plot_torch_loss(
            loss_function.loss_history,
            first_step=-200,
            title=f"{loss_function.displayname} Optimization Loss",
        )

# %%
"""
Visualize optimized wavefronts and scatterers
"""


for loss_function in loss_functions_dict.values():
    loss_function.visualize(
        base_sim_for_vis,
        base_sim_for_vis.k_x_array,
        title=f"Maximum Information State for {loss_function.displayname}",
        dpi=200,
    )

# %%
"""
Save results to be used in other scripts
"""

if save_state:
    loss_functions_dict.save(folder=data_folder)
