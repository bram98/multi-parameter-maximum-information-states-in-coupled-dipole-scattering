# playground/calc_fisher_info_and_MIS_3params_refactored.py

# %% Imports and functions
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs import placedipoles  # Import the whole module
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

# %% 1) define dipole configurations
movable_dipoles_line = placedipoles.place_line_of_dipoles(
    center=(5.0, 0.0), number=7
)

frozen_dipoles_line = placedipoles.place_line_of_dipoles(
    center=(3.0, 0.0), number=7
)

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

# placeholder for base_sim for visualization
# sim_kwargs already contains the correct total_Ns_sim calculated earlier.
# No need to modify sim_kwargs["Ns"] here.
_sim_params_vis = sim_kwargs.copy()

# Correctly combine frozen and movable dipoles for visualization.
# The convention from placedipoles.shift_and_rotate_dipoles is frozen_first, then movable.
if frozen_dipoles_line is not None and frozen_dipoles_line.size > 0:
    _scatterers_vis = np.hstack((frozen_dipoles_line, movable_dipoles_line))
else:
    # If there are no frozen dipoles, just use the movable ones.
    _scatterers_vis = movable_dipoles_line

# Initialize the simulator for visualization with the correct total Ns (from sim_kwargs)
# and set its scatterers to the combined array.
base_sim_for_vis = CoupledDipoleSimulator(**_sim_params_vis)
base_sim_for_vis.scatterers = _scatterers_vis

# %% 5) Single-parameter MIS wavefronts and their Fisher Information Matrices

print("\n--- Section 5: Single-Parameter MIS Wavefronts ---")

# Initialize the Optimizer with the F_tensor_operator from Cell 4
optimizer = Optimizer(F_operator_tensor=F_tensor_operator)

# Photon budget for normalization
photon_budget = 1.0

# Get single-parameter MIS wavefronts
single_mis_results = optimizer.get_single_parameter_mis(
    param_names=param_names, photon_budget=photon_budget
)

# Store wavefronts and their J matrices
optimized_wavefronts = {}
fisher_matrices_for_wavefronts = {}

print(
    "\nCalculating and storing Fisher Information Matrix for each single-parameter MIS:"
)
for name, result in single_mis_results.items():
    wavefront = result["wavefront"]
    optimized_wavefronts[f"MIS_{name}"] = wavefront

    # Calculate the 3x3 Fisher Information Matrix for this specific wavefront
    J_for_mis = optimizer.J_fisher_info_matrix(wavefront)
    fisher_matrices_for_wavefronts[f"MIS_{name}"] = J_for_mis

    # Visualize the intensity pattern for each single-parameter MIS
    print(f"  Visualizing intensity for MIS_{name}...")
    base_sim_for_vis.reset()  # Reset simulator state
    base_sim_for_vis.set_input_field_kx(
        kx_array, wavefront
    )  # kx_array should be from previous cells
    base_sim_for_vis.solve_fields()
    base_sim_for_vis.visualize_intensity(
        title=f"MIS for {name} (Max Eigenvalue F_{name}{name} = {result['eigenvalue']:.2e})",
        dpi=100,
    )
    plt.show()

print("\nSingle-parameter MIS analysis complete.")

# %% 6) Combined wavefront => sum of diagonal blocks (Trace optimization via eigendecomposition)

print(
    "\n--- Section 6: Combined Wavefront (Trace J via Eigendecomposition) ---"
)

# F_sum_op is the operator sum(F_mumu) whose largest eigenvector maximizes Tr(J)
# when J is calculated with that eigenvector.
F_sum_op = np.zeros_like(
    F_tensor_operator[0, 0, :, :]
)  # Initialize as (N_in, N_in)
for i in range(optimizer.nparams):
    F_sum_op += F_tensor_operator[i, i, :, :]

# Find the eigenvector of F_sum_op corresponding to its largest eigenvalue
# This eigenvector is the wavefront that maximizes trace(J) among this class of solutions
trace_opt_eigval, trace_opt_wavefront_eig = optimizer.eig_optimize_matrix(
    F_sum_op, maximize=True, num=1
)
trace_opt_wavefront_eig = Optimizer.normalize_wavefront(
    trace_opt_wavefront_eig, photon_budget
)

optimized_wavefronts["TraceJ_eig"] = trace_opt_wavefront_eig
J_for_trace_opt_eig = optimizer.J_fisher_info_matrix(trace_opt_wavefront_eig)
fisher_matrices_for_wavefronts["TraceJ_eig"] = J_for_trace_opt_eig

print(f"Trace-Optimized Wavefront (from eig(sum F_mumu)):")
print(f"  Eigenvalue of sum(F_mumu): {trace_opt_eigval:.3e}")
# print(f"  Wavefront (first 5 coeffs): {trace_opt_wavefront_eig[:5]}")
print(f"  Fisher Info Matrix J (for this wavefront):")
for row in J_for_trace_opt_eig:
    print(f"    [{row[0]:.2e}, {row[1]:.2e}, {row[2]:.2e}]")
print(f"  Trace(J) = {np.trace(J_for_trace_opt_eig):.3e}")


print(f"  Visualizing intensity for TraceJ_eig...")
base_sim_for_vis.reset()
base_sim_for_vis.set_input_field_kx(kx_array, trace_opt_wavefront_eig)
base_sim_for_vis.solve_fields()
base_sim_for_vis.visualize_intensity(
    title=f"Trace J Optimized Wavefront (Eigendecomposition of sum F_mumu)\nMax Eigenvalue = {trace_opt_eigval:.2e}",
    dpi=100,
)
plt.show()

# For compatibility with monolith variable names, if needed later
v_sum_max = trace_opt_wavefront_eig


print("\nSections 5 and 6 refactoring complete.")
