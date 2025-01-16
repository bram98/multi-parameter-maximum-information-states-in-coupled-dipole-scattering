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
    grid_points_x=400
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


# %% Study the cross-section and immediate environment of the scatterer

def get_1D_cross_section(sim: CoupledDipoleSimulator,
                         field_2d: np.ndarray,
                         zslice_idx: int = None):
    """
    From a 2D field array (shape: (grid_points_x, grid_points_z)),
    extract the 1D cross-section at the specified z-slice index.

    If zslice_idx is None, default to the midpoint in z.
    Returns x_vals, E_slice (both 1D arrays).
    """
    if zslice_idx is None:
        zslice_idx = sim.grid_points_z // 2  # e.g. mid-plane
    E_slice = field_2d[:, zslice_idx]  # shape (grid_points_x,)
    x_vals = sim.x                     # shape (grid_points_x,)
    return x_vals, E_slice

def discrete_derivatives(yvals, spacing):
    """
    Returns first and second discrete derivatives of yvals
    using a simple finite difference.
    yvals: 1D array (the amplitude or any other array)
    spacing: float, the step size in x

    We use np.gradient for convenience. The 1st derivative is
    dy/dx, and the 2nd derivative is d^2y/dx^2.
    """
    dy_dx = np.gradient(yvals, spacing)
    d2y_dx2 = np.gradient(dy_dx, spacing)
    return dy_dx, d2y_dx2

# Suppose you have a single dipole in the center, plus your MIS (v_max) and LIS (v_min)
# along with kx_array. We'll do the cross-sections for amplitude.

# -- MIS:
sim_base.reset()
sim_base.set_input_field_kx(kx_array, v_max)
sim_base.solve_fields()
E_2d_MIS = sim_base.compute_total_field()   # shape: (grid_points_x, grid_points_z)
x_vals_raw, slice_MIS = get_1D_cross_section(sim_base, E_2d_MIS, zslice_idx=None)

# Convert from simulator x-coords to x/lambda
lambda_0 = sim_base.lambda_0
x_vals = x_vals_raw / lambda_0
# We'll look at amplitude, i.e. abs(E)
amplitude_MIS = np.abs(slice_MIS)

# Discrete derivatives
dx_val = x_vals[1] - x_vals[0]  # step in units of lambda
d1_MIS, d2_MIS = discrete_derivatives(amplitude_MIS, dx_val)

# -- LIS:
sim_base.reset()
sim_base.set_input_field_kx(kx_array, v_min)
sim_base.solve_fields()
E_2d_LIS = sim_base.compute_total_field()
x_vals_raw, slice_LIS = get_1D_cross_section(sim_base, E_2d_LIS, zslice_idx=None)
x_vals = x_vals_raw / lambda_0
amplitude_LIS = np.abs(slice_LIS)
d1_LIS, d2_LIS = discrete_derivatives(amplitude_LIS, dx_val)

########################################
# Plot:  3 rows × 2 columns:
#   row0: amplitude for MIS (left) vs LIS (right)
#   row1: 1st derivative
#   row2: 2nd derivative
########################################
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,8), sharex=True)

# Row 0: amplitude
axes[0,0].plot(x_vals, amplitude_MIS, 'b-')
axes[0,0].set_title("MIS: |E| at z=mid")
axes[0,0].set_ylabel("Amplitude")
axes[0,0].set_xlim([-1,1])  # zoom in on -1..+1 λ

axes[0,1].plot(x_vals, amplitude_LIS, 'r-')
axes[0,1].set_title("LIS: |E| at z=mid")
axes[0,1].set_xlim([-1,1])  # same zoom

# Row 1: 1st derivative
axes[1,0].plot(x_vals, d1_MIS, 'b-')
axes[1,0].set_title("MIS: d|E|/dx")
axes[1,0].set_ylabel("1st deriv")

axes[1,1].plot(x_vals, d1_LIS, 'r-')
axes[1,1].set_title("LIS: d|E|/dx")

# Row 2: 2nd derivative
axes[2,0].plot(x_vals, d2_MIS, 'b-')
axes[2,0].set_title("MIS: d²|E|/dx²")
axes[2,0].set_xlabel("x / λ")
axes[2,0].set_ylabel("2nd deriv")

axes[2,1].plot(x_vals, d2_LIS, 'r-')
axes[2,1].set_title("LIS: d²|E|/dx²")
axes[2,1].set_xlabel("x / λ")

# Turn on the grid for all subplots in one go
for ax_row in axes:
    for ax in ax_row:
        ax.grid(True)

plt.tight_layout()
plt.show()


# %% generate time evolution

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def compute_complex_field(sim, kx_array, wavefront):
    """
    Return the 2D complex field (Nx x Nz) after solving with 'wavefront' input.
    """
    sim.reset()
    sim.set_input_field_kx(kx_array, wavefront)
    sim.solve_fields()
    return sim.compute_total_field()  # shape: (grid_points_x, grid_points_z)

def animate_field_time_evolution(E_2d, sim, frames=60, blit=False):
    """
    Create an animation of the real part of the field vs. time for 'frames' steps per period.
    E_2d: complex 2D field array (Nx x Nz)
    sim:  CoupledDipoleSimulator instance (contains .x, .z, .scatterers, etc.)
    frames: number of time steps in one optical period
    blit: if True, only changes in the main image are redrawn. If you have static objects
          (like scatterers) that must remain visible, you often want blit=False,
          or else return the scatter in the update function's artist list.
    """
    fig, ax = plt.subplots(figsize=(6,5))

    Nx, Nz = E_2d.shape
    x_vals = sim.x / sim.lambda_0
    z_vals = sim.z / sim.lambda_0

    # Initial real part (time=0)
    realE_0 = E_2d.real

    # pcolormesh to show Re(E)
    im = ax.pcolormesh(z_vals, x_vals, realE_0,
                       shading='auto', cmap='RdBu_r')
    ax.set_xlabel("z / λ")
    ax.set_ylabel("x / λ")
    ax.set_title("Real part of E (Time Evolution)")
    cbar = fig.colorbar(im, ax=ax, label="Re(E)")

    # Plot the scatterers in red
    scatx = sim.scatterers[1,:] / sim.lambda_0
    scatz = sim.scatterers[0,:] / sim.lambda_0
    sc = ax.scatter(scatz, scatx, c='red', s=40, label="Scatterers")
    ax.legend()

    # We define the angular frequency for a vacuum wave at lambda_0:
    c = 1.0  # normalized speed of light
    omega = 2*np.pi*c / sim.lambda_0  # if sim.lambda_0 is in the same length units

    # Prepare the update function
    def update(frame):
        # frame goes from 0..(frames-1)
        # We let one full cycle = frames steps => t from 0..(2π/ω)
        t = frame / frames * (2*np.pi / omega)
        phase_factor = np.exp(-1j * omega * t)
        E_t = (E_2d * phase_factor).real
        im.set_array(E_t.ravel())  # update the pcolormesh data

        # If blit=True, you should return *all* artists that need redrawing:
        # return (im, sc)
        return (im,)

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=200,
        blit=blit, repeat=True
    )
    return ani


# ---------------------
# Example usage:
# ---------------------
# 1) Assume 'sim' is your CoupledDipoleSimulator with scatterers
# 2) 'kx_array' is your input angle basis
# 3) 'v_max' is the MIS wavefront in angle space
E_2d_MIS = compute_complex_field(sim_base, kx_array, v_max)
anim = animate_field_time_evolution(E_2d_MIS, sim_base, frames=60, blit=False)

# Now, we can save both MP4 and GIF at 60 FPS
# a) Save an MP4 using ffmpeg:
anim.save("MIS_time_evolution.mp4", writer="ffmpeg", fps=60)

# b) Save a GIF using ImageMagick:
anim.save("MIS_time_evolution.gif", writer="imagemagick", fps=60)

# If you're in a Jupyter environment and just want inline display:
# from IPython.display import HTML
# HTML(anim.to_jshtml())

plt.show()