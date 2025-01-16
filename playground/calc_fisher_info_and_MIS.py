# calc_fisher_info_and_MIS.py
# %%
"""
Script that demonstrates how to:
 1) Import our "build_scattering_matrix_direct()" function (from build_scattering_matrix.py)
 2) Build a scattering matrix T(x_scat) for a 2D CoupledDipoleSimulator system
 3) Perturb a chosen parameter — here the x-coordinate of the first scatterer — by +/- δ
 4) Estimate ∂T/∂x_scat via finite differences
 5) Compute the Fisher information operator F_x = (∂T/∂x_scat)^\dagger (∂T/∂x_scat)
    (in analogy to eq. (2) from Bouchet's paper)
 6) Diagonalize F_x to find its largest eigenvalue and eigenvector = the maximum-information state (MIS)
 7) Inject that MIS back into the simulator, and visualize the resulting intensity distribution in the slab

We thus see how one can do "optimal wavefront shaping" for the metrology task: measuring the x-position
of a chosen scatterer. The logic readily extends to other parameters, e.g. polarizabilities, other coordinates, etc.
"""

import matplotlib.pyplot as plt
# %% Imports
import numpy as np
# Likewise, we import the build_scattering_matrix_direct utility we wrote.
from build_scattering_matrix import (build_scattering_matrix_direct,
                                     flux_amplitude_2d)
# We reuse the CoupledDipoleSimulator from our local ScatteringSimulation script
# (or wherever you store it). Adjust import to your local paths as needed.
from ScatteringSimulation import CoupledDipoleSimulator


# %% 1) Helper: build T-matrix given a CoupledDipoleSimulator and a set of angles
def build_T_for_angles(sim: CoupledDipoleSimulator, angles_deg: np.ndarray):
    """
    Returns the T-matrix in the direct camera basis. Shape: (N_out, N_in).
    and also returns the x-array at z=Lz for reference.
    We do a quick wrapper around the build_scattering_matrix_direct(...) function.
    """
    T_matrix, x_cam = build_scattering_matrix_direct(sim, angles_deg)
    return T_matrix, x_cam


# %% 2) Helper: do finite difference on x of the first scatterer
def partial_derivative_T_xscatter(
    angles_deg,
    x_shift=0.005,  # small shift in same units as simulator uses (lambda_0 multiples, etc.)
    base_sim_kwargs=None,
    scatterer_index=0,
):
    """
    Build T(x0 + dx) and T(x0 - dx) for the *x* coordinate of the chosen scatterer_index,
    using a finite-difference approach. Return Tplus, Tminus, plus the reference x_cam array.

    base_sim_kwargs: dict of arguments to pass to CoupledDipoleSimulator
                     (e.g. Lz=..., Lx=..., Ns=..., alpha=..., seed=...).
    x_shift: the ± shift we apply to x for finite difference
    scatterer_index: which scatterer we want to move along x
    """
    if base_sim_kwargs is None:
        base_sim_kwargs = {}

    # 1) Construct a "base" simulator (with the unperturbed x-position)
    sim0 = CoupledDipoleSimulator(**base_sim_kwargs)
    # We'll read out the default positions of scatterers
    scatterers0 = sim0.scatterers.copy()  # shape (2, Ns)

    # 2) We'll do Tminus: set x of that scatterer to x0 - x_shift
    scatterers_minus = scatterers0.copy()
    scatterers_minus[1, scatterer_index] = (
        scatterers_minus[1, scatterer_index] - x_shift
    )

    sim_minus = CoupledDipoleSimulator(**base_sim_kwargs)
    sim_minus.scatterers = scatterers_minus
    T_minus, x_cam = build_T_for_angles(sim_minus, angles_deg)

    # 3) We'll do Tplus: set x of that scatterer to x0 + x_shift
    scatterers_plus = scatterers0.copy()
    scatterers_plus[1, scatterer_index] = scatterers_plus[1, scatterer_index] + x_shift

    sim_plus = CoupledDipoleSimulator(**base_sim_kwargs)
    sim_plus.scatterers = scatterers_plus
    T_plus, _ = build_T_for_angles(sim_plus, angles_deg)

    return T_plus, T_minus, x_cam


# %% 3) Putting it all together in a main demonstration
# if __name__ == "__main__":

# We choose a small system for speed
base_kwargs = dict(
    lambda_0=1.0,
    Lz=8.0,  # slab thickness in wavelengths
    Lx=8.0,  # slab width in wavelengths
    Ns=4,  # minimal number of scatterers for demonstration
    seed=42,  # reproducible scatterer placement
)

# By default, the first scatterer is near the center. We'll treat x-shifts of that scatterer.
max_angle = 45
angles = np.linspace(-max_angle, max_angle, 100)  # a few angles
dx = 1e-6  # finite difference shift in *x* coordinate

# %% Step A) Build Tplus, Tminus
T_plus, T_minus, x_cam = partial_derivative_T_xscatter(
    angles_deg=angles, x_shift=dx, base_sim_kwargs=base_kwargs, scatterer_index=0
)

# %% Step B) Estimate ∂T/∂x ~ [Tplus - Tminus] / (2*dx)
T_deriv = (T_plus - T_minus) / (2.0 * dx)
N_out, N_in = T_deriv.shape
print("Shape of T_deriv:", T_deriv.shape)

# Step C) Build Fisher operator F_x = (∂T/∂x)^\dagger * (∂T/∂x)
# F_x is shape (N_in, N_in), a square matrix that acts on the "input angle" dimension
# We interpret T_deriv as a linear operator: E_in -> direct-basis E_out
# So ∂T/∂x : C^(N_in) -> C^(N_out). Then F_x: C^(N_in) -> C^(N_in).
F_x = T_deriv.conjugate().T @ T_deriv  # shape (N_in, N_in)

# %% Step D) Diagonalize F_x
# eigenvectors v_j (in angle space) are the maximum-information states for that parameter (x-scatterer)
eigvals, eigvecs = np.linalg.eigh(F_x)  # Hermitian => use eigh
# sort descending
idx_sort = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[idx_sort]
eigvecs_sorted = eigvecs[:, idx_sort]

# The top eigenvalue, top eigenvector
lam_max = eigvals_sorted[0]
v_max = eigvecs_sorted[:, 0]

print("Largest eigenvalue of F_x =", lam_max)
print("Corresponding eigenvector shape =", v_max.shape)
print("Angles:", angles)

# %% Step E) The "maximum information state" (MIS) is that top eigenvector in angle space
# but we also might want flux normalization or so. For demonstration, let's keep it simple:
# We'll interpret v_max as amplitude in each angle channel. We can do a small amplitude factor
# such that sum of intensities = 1 or so. Or we can do naive normalization.
# We'll define the input array E_in for that mode.
E_in_MIS = v_max  # shape (N_in,)

# %% Step F) We'll build a simulator with the original unperturbed scatterer position
sim_base = CoupledDipoleSimulator(**base_kwargs)

# Convert angles -> kx, multiply by E_in_MIS
k0 = sim_base.k
kx_array = k0 * np.sin(np.deg2rad(angles))
# Optionally, do flux amplitude factor to get 1 flux per channel. We'll skip for clarity.
# We'll just use the raw complex amplitude from v_max.
# Or you can do E_in_MIS[i] *= flux_amplitude_2d(k0, kx_array[i]) if desired.

# %% Step G) Put that field in the simulator, solve, visualize
sim_base.set_input_field_kx(kx_array, E_in_MIS)
sim_base.solve_fields()

# visualize the cross-section at z=Lz
sim_base.visualize_output_field(plot_intensity_only=False, direct_computation=True)

# visualize intensity in the slab
sim_base.visualize_intensity()

# Extra sanity check: how large is the Fisher info if we feed in that MIS?
# By eq (2) in the Bouchet paper:  J_x = 4 * <E_in| (∂T/∂x)^\dagger (∂T/∂x) |E_in>
# but note that T_deriv is the "matrix" form in direct basis (N_out x N_in).
# So the user can do:
#   J_x = 4 * conj(E_in_MIS).T @ F_x @ E_in_MIS
J_x = 4.0 * (v_max.conjugate().T @ F_x @ v_max).real
print(f"Fisher info (approx) for x-scat param with this MIS = {J_x:.1f}")

# We can compare that to a random or plane-wave input for curiosity:
# define plane wave at 0 degrees
pw_index = np.argmin(np.abs(angles))  # pick the angle closest to 0
E_in_plane = np.zeros_like(v_max)
E_in_plane[pw_index] = 1.0  # amplitude 1.0 at that angle
J_pw = 4.0 * (E_in_plane.conjugate().T @ F_x @ E_in_plane).real
print(f"Fisher info for plane-wave near 0 deg = {J_pw:.1f}")

plt.show()

# %% Step H) The "lowest information state" (LIS) is the eigenvector with the smallest eigenvalue
# Extract the smallest eigenvalue and its corresponding eigenvector
lam_min = eigvals_sorted[-1]
v_min = eigvecs_sorted[:, -1]

print("Smallest eigenvalue of F_x =", lam_min)
print("Corresponding eigenvector shape =", v_min.shape)

# Define the input array E_in for the LIS
E_in_LIS = v_min  # shape (N_in,)

# Step I) Simulate the LIS field interaction
sim_base.set_input_field_kx(kx_array, E_in_LIS)
sim_base.solve_fields()

# Visualize the cross-section at z=Lz for LIS
sim_base.visualize_output_field(plot_intensity_only=False, direct_computation=True)

# Visualize intensity in the slab for LIS
sim_base.visualize_intensity()

# Extra sanity check: how small is the Fisher info if we feed in that LIS?
J_min = 4.0 * (v_min.conjugate().T @ F_x @ v_min).real
print(f"Fisher info (approx) for x-scat param with this LIS = {J_min:.1f}")

plt.show()

# %% Extra sanity check: how large is the Fisher info if we feed in that MIS?
J_x = 4.0 * (v_max.conjugate().T @ F_x @ v_max).real
std_dev_x = (
    (1.0 / J_x) ** 0.5 if J_x > 0 else np.inf
)  # Avoid sqrt of negative or division by zero
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Fisher info (approx) for x-scat param with this MIS = {J_x:.1f}")
print(f"Standard deviation for x-scat param with this MIS = {std_dev_x:.2e}")

# We can compare that to a random or plane-wave input for curiosity:
pw_index = np.argmin(np.abs(angles))  # pick the angle closest to 0
E_in_plane = np.zeros_like(v_max)
E_in_plane[pw_index] = 1.0  # amplitude 1.0 at that angle
J_pw = 4.0 * (E_in_plane.conjugate().T @ F_x @ E_in_plane).real
std_dev_pw = (1.0 / J_pw) ** 0.5 if J_pw > 0 else np.inf
print(f"Fisher info for plane-wave near 0 deg = {J_pw:.1f}")
print(f"Standard deviation for plane-wave near 0 deg = {std_dev_pw:.2e}")

# After Fisher Information Calculation for LIS
J_min = 4.0 * (v_min.conjugate().T @ F_x @ v_min).real
std_dev_min = (
    (1.0 / J_min) ** 0.5 if J_min > 0 else np.inf
)  # Avoid sqrt of negative or division by zero
print(f"Fisher info (approx) for x-scat param with this LIS = {J_min:.1f}")
print(f"Standard deviation for x-scat param with this LIS = {std_dev_min:.2e}")