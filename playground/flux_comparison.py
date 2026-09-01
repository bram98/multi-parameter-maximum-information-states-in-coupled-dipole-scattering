# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 14:37:43 2025

@author: 5914787
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from scatteringlibs import placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

sys.exit()

# %%
movable_dipoles = placedipoles.place_quasi_random_line_of_dipoles(
    center=(5.2, 0.0), number=6, small_dist=0.786, start_index=5
)
movable_dipoles = None
frozen_dipoles = None


total_Ns_sim = 0

if movable_dipoles is not None and movable_dipoles.size > 0:
    total_Ns_sim += movable_dipoles.shape[1]
if frozen_dipoles is not None and frozen_dipoles.size > 0:
    total_Ns_sim += frozen_dipoles.shape[1]

max_angle = 80  # degrees
angles_deg = np.linspace(-max_angle, max_angle, 100)  # Input angle channels

sim = CoupledDipoleSimulator(
    lambda_0=1.0,
    Lz=8.0,
    Lx=8.0,
    Ns=total_Ns_sim,  # Ns in sim_kwargs should be the total number of dipoles
    seed=123,
    NA_out=None,
    grid_points_x=50,  # Number of points for field calculation (output basis for S-matrix)
)
sim.movable_scatterers = movable_dipoles
sim.frozen_scatterers = frozen_dipoles

kx_array = sim.k * np.linspace(-1, 1.00, num=3200, endpoint=False)
kx_array = np.fft.ifftshift(kx_array)  # zero in the corner looks nice
sim.set_input_field_angles(k_x_array=kx_array)
# sim.set_input_field_angles(theta_inc_deg=np.linspace(-70, 70, num=100))
amplitudes = kx_array * 0
amplitudes[-1] = 1
sim.set_input_field_amplitudes(amplitudes, flux_normalize=False)
Sr, _ = sim.build_scattering_matrix(False)

scaling_factor = np.linalg.norm(Sr[:, 0])
Sr /= scaling_factor
# for i in range(Sr.shape[1]):
#     # print(i)
#     Sr[:, i] /= np.linalg.norm(Sr[:, i])
#     # print(np.linalg.norm(Sr[:, i]))

# for i in range(Sr.shape[0]):
#     # print(i)
#     Sr[i] /= np.linalg.norm(Sr[i])
#     # print(np.linalg.norm(Sr[i]))

# dx = sim.x[1] - sim.x[0]
# Sr = np.diag(np.ones_like(sim.x)*np.sqrt(1)) @ Sr

S = np.fft.fft(Sr, axis=0, norm="ortho")
# S = np.fft.fftshift(S, axes=1)
# plt.imshow(np.abs(S), extent=[0, np.max(kx_array),0, np.max(kx_array)])
plt.imshow(np.abs(S))
plt.xlabel("$k_x\ (in)$")
plt.ylabel("$k_x\ (out)$")
# plt.xticks(kx_array)
plt.colorbar()
plt.show()

svd = np.linalg.svd(S)
plt.plot(svd.S)
plt.axvline(16, c="k", ls="--")
plt.xlim(10, 30)
plt.show()

# f = np.linalg.eigvalsh(S.conj().T @ S)
# plt.plot(f)
# print(np.sum(f))
# plt.show()
# sim.theta_inc_deg = angles_deg
# amplitude = np.zeros_like(angles_deg)
# amplitude[np.argmin(np.abs(angles_deg-10))] = 1
# sim.set_input_field_amplitudes(amplitude)


# %%

"""
Here I try to remove redundancy
"""
S_true = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

S = S_true.copy()

for i in range(10):
    v = np.random.rand(3)
    v = np.array([0, 0, 1.0])
    v /= np.linalg.norm(v)
    v_out = S_true @ v
    S = np.append(S, np.array([v_out]).T, axis=1)

# for c in range(3, S.shape[1]):
#     print(c)
#     S[:, c] /= np.linalg.norm(S[:, c])

print()
svd = np.linalg.svd(S)

U, sig, Vh = svd.U, svd.S, svd.Vh

sig2 = sig.copy()
for r in range(len(sig)):
    row = Vh[r]
    redundancy = np.sum(np.abs(row) > 1e-5)  # nonzero elements
    sig2[r] /= np.sqrt(redundancy)
    print(redundancy)

plt.plot(sig2)

# %%
import numpy as np

# 1. Define the true physical system
# Using a non-trivial matrix with some loss/coupling
S_true = np.array([[0.9, 0.1, 0.1], [0.1, 0.8, 0.2], [0.1, 0.2, 0.7]])
S = S_true.copy()

# 2. Make the matrix overcomplete by appending 10 columns
for i in range(10):
    v = np.random.rand(3)
    v /= np.linalg.norm(v)
    v_out = S_true @ v
    S = np.append(S, np.array([v_out]).T, axis=1)

# 3. Perform SVD on the overcomplete 3x13 matrix
U, sig, Vh = np.linalg.svd(S)

# 4. Recover S_true from the SVD components
# The rank 'r' is the number of significant singular values
r = np.sum(sig > 1e-10)
Sigma_r = np.diag(sig[:r])
S_recovered = U @ Sigma_r
# %%

sim.reset()
E_out = sim.compute_zL_fields_direct()

print(np.abs(E_out[:30]))

dx = 0.01
sim.x += dx
sim.reset()
E_out_dx = sim.compute_zL_fields_direct()
sim.x -= dx

dz = dx
sim.Lz += dz
sim.reset()
E_out_dz = sim.compute_zL_fields_direct()
sim.Lz -= dz

dE_dx = (E_out_dx - E_out) / dx
dE_dz = (E_out_dz - E_out) / dz

Sx = -np.imag(E_out * dE_dx.conjugate())
Sz = -np.imag(E_out * dE_dz.conjugate())

fig, ax = plt.subplots()
sim.visualize_real(ax=ax, title="Real part of E and Poynting vector")
# plt.quiver(sim.x, sim.x*0 + 4, Sx, Sz)
sub = 8
ax.quiver(
    sim.x[::sub] * 0 + 7.1,
    sim.x[::sub],
    Sz[::sub],
    Sx[::sub],
)
# plt.xlim(0, 8.5)
# plt.tight_layout()
plt.show()

# plt.plot(np.abs(E_out), label='Eout')
# plt.plot(np.abs(E_out_dx), label='Eoutdx')
# plt.plot(np.abs(E_out_dz), label='Eoutdz')
# plt.legend()
# plt.show()
z_flux_poynting = np.mean(Sz)

# %%
k = sim.k
NA_out = 0.95
kmax_out = k * NA_out
dx_sim = (np.roll(sim.x, -1) - sim.x)[0]
E_k = np.fft.fft(E_out)
kx_out = np.fft.fftfreq(len(E_k), d=dx_sim)

# NA_filter = np.where(np.abs(kx_out) > kmax_out)
# E_k[NA_filter] = 0

kx_out_clip = np.clip(kx_out, a_min=-k, a_max=k)
kz_out = np.sqrt(k**2 - kx_out_clip**2)
flux_factor = kz_out / k
plt.plot(flux_factor)
plt.show()

plt.plot(kx_out, np.abs(E_k))
plt.plot(kx_out, np.abs(E_k * flux_factor))
plt.yscale("log")
plt.show()

E_out_flux = np.fft.ifft(E_k * flux_factor)
E_out_rest = np.fft.ifft(E_k * (1 - flux_factor))

plt.plot(np.abs(E_out), label="Eout")
plt.plot(np.abs(E_out_flux), label="Eout_flux")
plt.legend()
plt.show()

# %%

scales = np.geomspace(8, 100, num=30)
# scales = [8, 64, 128]
fisher_information_list = []
kx_arrays = []

for scale in scales:
    Lz = 8
    Lx = scale
    movable_dipoles = placedipoles.place_quasi_random_line_of_dipoles(
        center=(1, 0.0),
        number=6,
        small_dist=0.786,
        start_index=5,
    )

    frozen_dipoles = placedipoles.place_quasi_random_line_of_dipoles(
        center=(-1, 0.0),
        number=6,
        small_dist=0.786,
        start_index=22,
    )

    kx_array = sim.make_kx_array_at_critical_sampling(
        NA_in=0.95, Lx=Lx, k=2 * np.pi
    )
    print(len(kx_array))
    kx_array = np.linspace(-np.pi, np.pi, endpoint=True, num=45)
    print(len(kx_array))
    print()
    sim = CoupledDipoleSimulator(
        Lz=Lz,
        Lx=Lx,
        Lz_vis=Lz,
        Lx_vis=Lx,
        grid_points_x=len(kx_array) * 0 + 800,
        NA_out=None,
        movable_scatterers=movable_dipoles,
        frozen_scatterers=frozen_dipoles,
    )

    sim.set_input_field_angles(k_x_array=kx_array)

    kx_arrays.append(kx_array)

    """
    visualize S matrix
    """
    # sim.reset()
    # S, _ = sim.build_scattering_matrix(False)

    # S = np.fft.fft(S, axis=0, norm="forward")
    # S = np.fft.fftshift(S, axes=0)
    # dx = sim.x[1] - sim.x[0]
    # kout = np.fft.fftfreq(len(sim.x), d=dx)
    # kout_range = (np.min(kout), np.max(kout))
    # kin_range = (np.min(kx_array) / sim.k, np.max(kx_array) / sim.k)
    # plt.imshow(np.abs(S), extent=kin_range + kout_range)
    # plt.xlabel(r"$k_x^{\mathrm{in}}/k$")
    # plt.ylabel(r"$k_x^{\mathrm{out}}/k$")
    # plt.colorbar()
    # plt.show()

    # dkin = (kx_array[1] - kx_array[0]) / (2 * np.pi)
    # dkout = kout[1] - kout[0]

    """
    Visualize plane wave
    """
    # sim.set_input_field_angles(k_x_array=kx_array)
    # E_array = (kx_array == 0.0).astype(complex)
    # sim.set_input_field_amplitudes(E_array)

    # # z = np.linspace(0, 8, 200, endpoint=True)
    # # # # x = np.linspace(-Lx/2, Lx/2, 200, endpoint=True)
    # # x = sim.x
    # # Z_grid, X_grid = np.meshgrid(z, x)

    # # # sim.Z_grid = Z_grid
    # # # sim.X_grid = X_grid
    # # sim.visualize_real(title="", dpi=300)

    """
    Calculate max. information state
    """
    angles_deg = sim.k_x2theta(kx_array, degree=True)
    dS = placedipoles.calculate_dS_for_3_parameters(
        sim.get_kwargs(), angles_deg, movable_dipoles
    )

    # dS = np.fft.fft(dS, axis=1, norm="forward")
    fishermatrix = FisherMatrix(dS=None)
    F = fishermatrix.F_from_dS(dS)
    optimizer = Optimizer(F_operator_tensor=F)

    lf = LossFunction(lambda F: F[0, 0], "x", method="eig")

    lf.optimize(optimizer=optimizer)
    fisher_information_list.append(lf.bestval)
    """
    visualize max. information state
    """
    # z = np.linspace(0, Lz, 200, endpoint=True)
    # x = np.linspace(-Lx/2, Lx/2, 200, endpoint=True)
    # Z_grid, X_grid = np.meshgrid(z, x)

    # sim.Z_grid = Z_grid
    # sim.X_grid = X_grid

    # lf.visualize(sim, kx_array)


# %%

fig, ax = plt.subplots()
ax_states = ax.twinx()


oversample = plt.axvline(45 / 2.0, ls="--", c="k", label="oversampled")
(FI_plot,) = ax.plot(
    scales, fisher_information_list, label="Fisher Information"
)

no_kx_arrays = [len(kx_array) for kx_array in kx_arrays]
(states_plot,) = ax_states.plot(
    scales, no_kx_arrays, "C1", label="# of states"
)

ax.yaxis.label.set_color(FI_plot.get_color())
ax_states.yaxis.label.set_color(states_plot.get_color())

plt.xlabel("Lz")
ax.legend(handles=[FI_plot, states_plot, oversample])

ax.set_ylabel("Fisher Information")
ax.set_xlabel("$L_x$")
ax_states.set_ylabel("# of states")
plt.show()


# %%
lambda_0 = 632000
critical_sampling = CoupledDipoleSimulator.calc_critical_sampling_at_N_modes(
    lambda_0=lambda_0
)
kx_array = critical_sampling["k_x_array"]
Lx = critical_sampling["Lx"]
print(Lx, lambda_0)

N = 10
sim = CoupledDipoleSimulator(
    Lx=Lx,
    Lz=Lx,
    lambda_0=lambda_0,
    k_x_array=kx_array,
    grid_points_x=len(kx_array),
    movable_scatterers=np.array(
        [np.random.rand(N) * 0.5, np.linspace(-8, 8, num=N)]
    ),
)
S, (k_in, k_out) = sim.build_scattering_matrix(False)
kextent = (
    np.array((np.min(k_in), np.max(k_in), np.min(k_out), np.max(k_out)))
    / sim.k
)
plt.imshow(np.abs(S), extent=kextent)
plt.colorbar()
plt.show()
