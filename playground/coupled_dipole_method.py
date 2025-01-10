#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:39:44 2024

@author: Jacob Seifert  - j.seifert@uu.nl
"""

import matplotlib.pyplot as plt
# %% Import libraries
import numpy as np
from scipy.special import hankel1

# %% Constants
lambda_0 = 1.0  # Wavelength (normalized)
k = 2 * np.pi / lambda_0  # Wave number
Lz = 10 * lambda_0
Lx = 8 * lambda_0  # Dimensions of the slab (z is horizontal)
Ns = 5  # Number of scatterers
alpha = 4j / k ** 2  # Polarizability of scatterers (resonant)
exclusion_radius = lambda_0 / 10  # Minimum spacing between particles

# %% Scatterer positions
np.random.seed(3)  # For reproducibility
scatterers = [np.array([Lz / 2, Lx / 2])]  # put first one into the center
while len(scatterers) < Ns:
    candidate = np.array([np.random.uniform(0, Lz), np.random.uniform(0, Lx)])
    if all(np.linalg.norm(candidate - s) > exclusion_radius for s in scatterers):
        scatterers.append(candidate)
scatterers = np.array(scatterers)

# Plot scatterer positions
fig_width = 12  # inches
fig_height = fig_width * (Lx / Lz)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
plt.scatter(
    scatterers[:, 0] / lambda_0, scatterers[:, 1] / lambda_0, c="r", label="Scatterers"
)
plt.xlim(0, Lz / lambda_0)
plt.ylim(0, Lx / lambda_0)
plt.title("Scatterer Positions")
plt.xlabel("z (λ)")
plt.ylabel("x (λ)")
# plt.legend()
plt.gca().set_aspect(1)
plt.tight_layout()
plt.show()


# %% Green's function
def G0(r, r_prime):
    delta_r = np.array(r) - np.array(r_prime)
    if delta_r.ndim == 1:
        # Scalar case
        distance = np.linalg.norm(delta_r)
    else:
        # Array case
        distance = np.linalg.norm(delta_r, axis=-1)
    G = (1j / 4) * hankel1(0, k * distance)
    G = np.where(distance == 0, 0.0, G)  # Handle singularity at r = r_prime
    return G


# Verify Green's function for a fixed pair of points
r1 = np.array([0.5 * Lz, 0.5 * Lx])
r2 = np.array([0.6 * Lz, 0.6 * Lx])
g_value = G0(r1, r2)
print(f"Sample Green's function value: {g_value}")

# %% Solve coupled dipole equations
E0_amplitude = 1.0  # Incident field amplitude
E0_phase = np.exp(1j * k * scatterers[:, 0])  # Assuming propagation along z-axis
E0_vec = E0_amplitude * E0_phase  # Incident field at scatterers

# Interaction matrix
A = np.zeros((Ns, Ns), dtype=complex)
for n in range(Ns):
    for m in range(Ns):
        if n == m:
            A[n, m] = 1
        else:
            A[n, m] = -(k ** 2) * alpha * G0(scatterers[n], scatterers[m])

# Solve the linear system
fields = np.linalg.solve(A, E0_vec)

# Verify dipole fields
print(f"Fields at scatterers:\n{fields}")

# %% Calculate total field on a grid with square pixels
grid_points_x = 540

# Calculate grid spacing based on the z-dimension
delta_x = Lx / grid_points_x
grid_points_z = int(np.ceil(Lz / delta_x))  # Ensure enough points to cover Lx

# Generate the grid points
z = np.linspace(0, Lz, grid_points_z)  # Horizontal dimension
x = np.linspace(0, Lx, grid_points_x)  # Vertical dimension
Z, X = np.meshgrid(z, x)

# Initialize the total electric field array
E_total = np.zeros((grid_points_x, grid_points_z), dtype=complex)

# Compute the incident field across the grid
E_incident = E0_amplitude * np.exp(1j * k * Z)  # Incident field at grid points
E_total = np.copy(E_incident)  # Start with the incident field

# Flatten grid for vectorized computation
points = np.vstack(
    (Z.ravel(), X.ravel())
).T  # Shape: (grid_points_x * grid_points_z, 2)

# Add scattered fields from each scatterer
for n in range(Ns):
    # Compute the Green's function from scatterer n to all grid points
    G = (
        k ** 2 * alpha * G0(points, scatterers[n])
    )  # Shape: (grid_points_x * grid_points_z,)

    # Reshape G back to 2D grid and multiply by the induced field at scatterer n
    E_total += G.reshape(grid_points_x, grid_points_z) * fields[n]


# %% Visualize field phase
fig_width = 10  # inches
fig_height = fig_width * (Lx / Lz)
fig, ax = plt.subplots(figsize=(12, fig_height))
p = ax.pcolormesh(
    Z / lambda_0,
    X / lambda_0,
    np.angle(E_total),
    shading="auto",
    cmap="twilight_shifted",
)
fig.colorbar(p, ax=ax, label="Phase (radians)")
ax.set_title("Field Phase")
ax.set_xlabel("z (λ)")
ax.set_ylabel("x (λ)")
ax.set_aspect(1)
plt.tight_layout()
plt.show()

# %% Plot intensity of the total field
fig_width = 10  # inches
fig_height = fig_width * (Lx / Lz)
fig, ax = plt.subplots(figsize=(12, fig_height))
c = ax.pcolormesh(
    Z / lambda_0, X / lambda_0, np.abs(E_total) ** 2, shading="auto", cmap="viridis"
)
fig.colorbar(c, ax=ax, label="|E|**2")
ax.set_title("Field Intensity")
ax.set_xlabel("z (λ)")
ax.set_ylabel("x (λ)")
ax.set_aspect(1)
plt.tight_layout()
plt.show()
