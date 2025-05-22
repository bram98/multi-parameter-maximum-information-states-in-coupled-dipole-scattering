# %% imports
import matplotlib.pyplot as plt
import numpy as np
from libraries.ScatteringSimulation import CoupledDipoleSimulator

# We'll assume "fourierfun" or the usual np.fft is enough to handle the 1D FFT.
# No explicit dependencies beyond numpy, matplotlib, and your local scatteringsimulation.


# %% parameters and setup

lambda_0 = 1.0
NA = 0.5  # Example numerical aperture
num_incoming = 7  # Number of incoming angles/modes
num_outgoing = 7  # Number of outgoing angles/modes
Lz = 16.0  # Slab thickness in wavelengths
Lx = 12.0
Ns = 0  # Number of scatterers (example)
alpha = 4j / (2 * np.pi / lambda_0) ** 2
seed = 0

# We define input angles symmetrically around 0 degrees. For example:
theta_in_deg = np.linspace(
    -20, 20, num_incoming
)  # center around normal incidence
theta_in = np.deg2rad(theta_in_deg)

# We'll also define the angles at which we want to measure the output channels:
theta_out_deg = np.linspace(-25, 25, num_outgoing)
theta_out = np.deg2rad(theta_out_deg)

# We'll create a CoupledDipoleSimulator that has a large enough x-range
# and enough grid points in x to do a decent FFT.
grid_points_x = 600  # A bit larger for better sampling in k_x
sim = CoupledDipoleSimulator(
    lambda_0=lambda_0,
    Lz=Lz,
    Lx=Lx,
    Ns=Ns,
    alpha=alpha,
    exclusion_radius=0.5,  # example
    grid_points_x=grid_points_x,
    seed=seed,
)


# We'll define a convenience function to convert angles to k_x:
def theta2kx(theta_array, k0=2 * np.pi / lambda_0):
    return k0 * np.sin(theta_array)


# And one to mask out k_x beyond the NA cutoff:
def na_filter_kx(kx_array, E_kx, k0=2 * np.pi / lambda_0, NA=0.5):
    """Returns E_kx with values set to zero for |kx| > k0*NA."""
    cutoff = k0 * NA
    E_filtered = np.where(np.abs(kx_array) <= cutoff, E_kx, 0.0)
    return E_filtered


# We'll define the array of k_x for the simulator's x-grid:
# The x array is sim.x, with shape (grid_points_x,).
# We sample E at z = Lz, so we have E(x). Then do an fft to get E(kx).
dx = sim.x[1] - sim.x[0]
kx_full = np.fft.fftfreq(grid_points_x, d=dx) * 2.0 * np.pi

# We'll store the scattering matrix in a 2D array: S[out_index, in_index].
# S[i,j] ~ amplitude in output channel i from input channel j.
S_matrix = np.zeros((num_outgoing, num_incoming), dtype=complex)

# %% main loop

for j, th_in in enumerate(theta_in):
    # For each input angle, we define the simulator's incident angle:
    sim.theta_inc = th_in
    sim.reset()

    # Solve and compute the total field at z=Lz plane:
    E_zL = sim.compute_zL_fields_direct()  # shape: (grid_points_x,)

    # 1D FFT in x to get E(kx):
    # This is the field in the frequency domain (no shift for now).
    E_k = np.fft.fft(E_zL)

    # Apply the NA-limited top-hat filter in k_x:
    E_k_filtered = na_filter_kx(kx_full, E_k, NA=NA)

    # Now, for each output angle, we find the corresponding k_x_out:
    for i, th_out in enumerate(theta_out):
        kx_out = (2 * np.pi / lambda_0) * np.sin(th_out)

        # We'll find the closest index in kx_full to kx_out:
        idx = (np.abs(kx_full - kx_out)).argmin()

        # S_matrix[i, j] is the amplitude at that kx_out after NA filtering:
        S_matrix[i, j] = E_k_filtered[idx]

# %% visualization and checks

# We'll plot the magnitude of S_matrix as an image:
plt.figure(figsize=(6, 5))
plt.title("S-matrix Magnitude (NA-limited in 1D)")
plt.imshow(
    np.abs(S_matrix),
    origin="lower",
    aspect="auto",
    extent=[
        theta_in_deg[0],
        theta_in_deg[-1],
        theta_out_deg[0],
        theta_out_deg[-1],
    ],
)
plt.colorbar(label="|S(out, in)|")
plt.xlabel("Input angle (deg)")
plt.ylabel("Output angle (deg)")
plt.show()

# We can also look at phases if desired:
plt.figure(figsize=(6, 5))
plt.title("S-matrix Phase (radians)")
plt.imshow(
    np.angle(S_matrix),
    origin="lower",
    aspect="auto",
    extent=[
        theta_in_deg[0],
        theta_in_deg[-1],
        theta_out_deg[0],
        theta_out_deg[-1],
    ],
    cmap="twilight",
)
plt.colorbar(label="Arg[S(out, in)]")
plt.xlabel("Input angle (deg)")
plt.ylabel("Output angle (deg)")
plt.show()

# %%
# -------------------------------
# 4) OPTIONAL: EXAMPLE SINGLE INPUT
#    Show field in x-space and kx-space with/without filtering
# -------------------------------

test_input_index = num_incoming // 2  # pick the middle input angle
th_test = theta_in[test_input_index]
sim.theta_inc = th_test
sim.reset()
E_zL = sim.compute_zL_fields_direct()

E_k = np.fft.fft(E_zL)
E_kfilt = na_filter_kx(kx_full, E_k, NA=NA)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title(f"Real-space field at z=Lz\nInc angle = {np.rad2deg(th_test):.1f}°")
plt.plot(sim.x / lambda_0, np.abs(E_zL) ** 2, "b-", label="Intensity")
plt.xlabel("x / lambda")
plt.ylabel("|E|^2")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("Fourier domain (kx) before/after NA filter")
plt.plot(kx_full, np.abs(E_k), "g-", label="|E_k| raw")
plt.plot(kx_full, np.abs(E_kfilt), "r--", label="|E_k| filtered")
cutval = (2 * np.pi / lambda_0) * NA
plt.axvline(x=cutval, color="k", linestyle=":")
plt.axvline(x=-cutval, color="k", linestyle=":")
plt.xlabel("kx")
plt.ylabel("|E_k|")
plt.legend()
plt.tight_layout()
plt.show()
