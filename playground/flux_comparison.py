# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 14:37:43 2025

@author: 5914787
"""
import matplotlib.pyplot as plt
import numpy as np

from scatteringlibs import placedipoles
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

# %%
movable_dipoles = placedipoles.place_quasi_random_line_of_dipoles(
    center=(5.2, 0.0), number=6, small_dist=0.786, start_index=5
)
# movable_dipoles = None
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
    grid_points_x=300,  # Number of points for field calculation (output basis for S-matrix)
)
sim.movable_scatterers = movable_dipoles
sim.frozen_scatterers = frozen_dipoles

sim.theta_inc_deg = 20

# sim.theta_inc_deg = angles_deg
# amplitude = np.zeros_like(angles_deg)
# amplitude[np.argmin(np.abs(angles_deg-10))] = 1
# sim.set_input_field_amplitudes(amplitude)
# %%

sim.reset()
E_out = sim.compute_zL_fields_direct()

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
