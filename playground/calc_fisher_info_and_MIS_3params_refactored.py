# playground/calc_fisher_info_and_MIS_3params_refactored.py

# %% Imports and functions
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs import placedipoles  # Import the whole module
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

# %% 1) define dipole configurations
movable_dipoles_line = placedipoles.place_line_of_dipoles(
    center=(5.0, 0.0), number=7
)

frozen_dipoles_line = placedipoles.place_line_of_dipoles(
    center=(3.0, 0.0), number=7
)

# Example: Using the double line setup, movable part is the second line
# If you want to work with the `place_double_line_of_dipoles` function directly:
# all_dipoles = placedipoles.place_double_line_of_dipoles(number=7)
# frozen_part = all_dipoles[:, :7]
# movable_part = all_dipoles[:, 7:]

# Choose which configuration to use:
# current_movable_dipoles = movable_dipoles_line
# current_frozen_dipoles = frozen_dipoles_line
# To use a single line that moves:
# current_movable_dipoles = placedipoles.place_line_of_dipoles(number=7)
# current_frozen_dipoles = None # Or np.array([[], []])


# %% 2) simulator geometry and input space
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
max_angle = 80  # degrees
angles_deg = np.linspace(-max_angle, max_angle, 100)  # Input angle channels

# We'll create a base simulator instance for potential visualizations later,
# but the S-matrix calculation will create its own instances.
# base_sim = CoupledDipoleSimulator(**sim_kwargs)
# if current_frozen_dipoles is not None and current_frozen_dipoles.size > 0:
# base_sim.scatterers = np.hstack((current_frozen_dipoles, current_movable_dipoles))
# else:
# base_sim.scatterers = current_movable_dipoles

# k0 = base_sim.k # Requires base_sim to be initialized
k0 = 2 * np.pi / sim_kwargs["lambda_0"]
kx_array = k0 * np.sin(np.deg2rad(angles_deg))


# %% 3) Build partial derivatives for each param => dS[mu]
param_names = ["shiftX", "shiftZ", "rotation"]
delta_param = 1e-3  # Finite difference step

print("Calculating dS for 3 parameters...")
dS_tensor = placedipoles.calculate_dS_for_3_parameters(
    sim_kwargs,
    angles_deg,
    movable_dipoles=movable_dipoles_line,
    frozen_dipoles=frozen_dipoles_line,
    delta_param=delta_param,
    flux_normalize=True,
)
print(
    f"Built dS tensor with shape: {dS_tensor.shape}"
)  # Expected: (3, N_out_angles, N_in_angles)
# The S-matrix from build_scattering_matrix is (N_out_direct_samples, N_in_angles).
# So dS_tensor from calculate_dS_for_3_parameters will also be (3, N_out_direct_samples, N_in_angles)

# %% 4) Build the full multi-parameter Fisher-operator array F[mu,nu,:,:]
# The FisherMatrix class expects dS of shape (nparams, N_out, N_in)
# F will be (nparams, nparams, N_in, N_in)

fisher_calc = FisherMatrix(
    dS=dS_tensor, param_names=param_names, delta=delta_param
)
F_tensor_operator = fisher_calc.F  # Access the F property

print(
    "Built F[mu,nu,:,:], shape =",
    F_tensor_operator.shape,
    f" (should be 3,3,{len(angles_deg)},{len(angles_deg)})",
)

# At this point, F_tensor_operator is ready for the subsequent optimization steps
# from the original script (sections 5 onwards).

# Save the calculated dS and F (optional)
# fisher_calc.save(folder="data/refactored_run", name="dS_3params_doubleline")
# np.savez(Path("data/refactored_run") / "F_3params_doubleline.npz", F=F_tensor_operator, param_names=param_names)

print("\nInitial refactoring steps complete. F_tensor_operator is ready.")
print("You can now proceed with sections 5 onwards from your original script,")
print("using `F_tensor_operator` and `kx_array` (or `angles_deg`).")
print(
    "Make sure to adjust imports for Optimizer and other utilities if they are also in scatteringlibs."
)
