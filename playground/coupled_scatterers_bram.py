# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:11:23 2024

@author: 5914787
"""

import fourierfun as ff
import matplotlib.pyplot as plt
# %%
import numpy as np
import scipy
from fourierfun import cplot

# %%


xmax = 8
ymax = 8
nx = xmax * 50
ny = ymax * 50
wl = 1  # wavelengtgh
k = 2 * np.pi / wl  # wave number
alpha = 4j / k ** 2  # polarizability


fr = ff.FourierRange2d(nx, ny, xmax=xmax, ymax=ymax)


n_scatters = 3
pos_scatters = np.random.uniform(high=(xmax, ymax), size=(n_scatters, 2)).T

G_scatter = np.zeros((n_scatters, n_scatters), dtype=complex)


def G0(r1, r2, k):
    """
    Free space Greens function

    Parameters
    ----------
    r1, r2 : float numpy array of shape (2, ...)
        positions to evaluate this function at
    k : float
        wavenumber

    Returns
    -------
    numpy array
    """
    return 1j / 4 * scipy.special.hankel1(0, k * np.linalg.norm(r1 - r2, axis=0))


for i in range(n_scatters):
    for j in range(n_scatters):
        if i == j:
            continue
        G_scatter[i, j] = k ** 2 * alpha * G0(pos_scatters[:, i], pos_scatters[:, j], k)


def E0(r, k):
    """
    Generates background field.

    Parameters
    ----------
    r : float numpy array of shape (2, ...)
        positions to evaluate this function at
    k : float
        wavenumber

    Returns
    -------
    numpy array
    """

    return np.exp(1j * k * r[0])


E0_scatters = E0(pos_scatters, k)

E_scatters = np.linalg.pinv(np.eye(n_scatters) - G_scatter) @ E0_scatters

X, Y = fr.XY()

# %%
n_grid = nx * ny
grid = np.array([X, Y])
grid_flat = grid.reshape((2, -1))
G_grid = np.zeros((nx * ny, n_scatters), dtype=complex)

for i in range(n_grid):
    for j in range(n_scatters):
        G_grid[i, j] = k ** 2 * alpha * G0(grid_flat[:, i], pos_scatters[:, j], k)

E0_grid = E0(grid_flat, k)

E_grid = E0_grid + G_grid @ E_scatters

E = E_grid.reshape((ny, nx))

# %%

plt.title("Background")
plt.imshow(
    np.angle(E0(grid, k)),
    extent=fr.plot_extent(),
    cmap="twilight",
    interpolation_stage="rgba",
)
plt.colorbar()
plt.xlabel(r"$x / \lambda$")
plt.ylabel(r"$y / \lambda$")

# %%
plt.figure(dpi=300)
plt.imshow(np.real(E), extent=fr.plot_extent(), origin="lower")
plt.scatter(*(pos_scatters), c="r", s=1)

plt.title(r"Intensity")
plt.colorbar()
plt.xlabel(r"$x / \lambda$")
plt.ylabel(r"$y / \lambda$")
# %%
plt.figure(dpi=300)
plt.imshow(np.abs(E) ** 2, extent=fr.plot_extent(), origin="lower")
plt.scatter(*(pos_scatters), c="r", s=1)

plt.title(r"Intensity")
plt.colorbar()
plt.xlabel(r"$x / \lambda$")
plt.ylabel(r"$y / \lambda$")
