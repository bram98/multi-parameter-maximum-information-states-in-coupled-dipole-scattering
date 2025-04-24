# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:43:43 2025

@author: 5914787
"""
# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
import os
import numpy as np
import torch
from libraries.ScatteringSimulation import CoupledDipoleSimulator
from libraries import placedipoles
from libraries.fishermatrix import FisherMatrix
from libraries.optimizer import Optimizer
from libraries.savestate import saveobj, loadobj

# import sys
# sys.exit()

# %%
folder = "../data/double_line7"

sim = loadobj("coupleddipolesimulator", folder)
kx_array = loadobj("kx_array", folder)
base_dipoles, frozen_dipoles = loadobj("dipoles", folder)
fishermatrix = loadobj("fishermatrix", folder)
loss_functions = loadobj("loss_functions", folder)

# %%
delta = 1e-3

dE_dtheta = np.zeros((3, sim.grid_points_x), dtype=complex)

for p in range(3):
    loss_function = [lf for lf in loss_functions if lf.name == "det"][0]
    wavefront = loss_function.wavefront
    sim.set_input_field_kx(kx_array, wavefront)

    dtheta_vec = delta * np.eye(3)[p]
    shifted_dipoles = placedipoles.shift_and_rotate_dipoles(
        base_dipoles, *dtheta_vec, frozen_dipoles=frozen_dipoles
    )
    sim.reset()
    sim.scatterers = shifted_dipoles
    E_plus = sim.compute_zL_fields_direct()

    shifted_dipoles = placedipoles.shift_and_rotate_dipoles(
        base_dipoles, *(-dtheta_vec), frozen_dipoles=frozen_dipoles
    )
    sim.reset()
    sim.scatterers = shifted_dipoles
    E_minus = sim.compute_zL_fields_direct()

    dE_dtheta[p] = (E_plus - E_minus) / (2 * delta)
# %%
n_theta = 200
E_output = np.zeros((3, n_theta, sim.grid_points_x), dtype=complex)
theta_range = 0.1
theta_in = np.linspace(-theta_range, theta_range, num=n_theta)

# dE_dtheta = np.zeros((3, sim.grid_points_x), dtype=complex)
E0 = np.zeros((3, sim.grid_points_x), dtype=complex)

for p in range(3):
    loss_function = [lf for lf in loss_functions if lf.name == "det"][0]
    wavefront = loss_function.wavefront
    sim.set_input_field_kx(kx_array, wavefront)

    sim.reset()
    sim.scatterers = np.hstack([base_dipoles, frozen_dipoles])
    E0[p] = sim.compute_zL_fields_direct()
    # E0[p] = loss_function.normalize_wavefront(E0[p])
    # dE_dtheta[p] = fishermatrix.dS[p] @ wavefront

    # theta_meas = np.zeros_like(theta_in)

    for i, dtheta in enumerate(theta_in):
        dtheta_vec = dtheta * np.eye(3)[p]
        shifted_dipoles = placedipoles.shift_and_rotate_dipoles(
            base_dipoles, *dtheta_vec, frozen_dipoles=frozen_dipoles
        )
        sim.reset()
        sim.scatterers = shifted_dipoles
        E_output[p, i] = sim.compute_zL_fields_direct()
        # E_output[p, i] = loss_function.normalize_wavefront(E_output[p, i])
    # sim.visualize_intensity()
# %%
for p in range(3):
    plt.figure()
    # ims = plt.imshow(np.abs(E_output[p].T), extent=[np.min(sim.z), np.max(sim.z), np.min(theta_in), np.max(theta_in)])
    ims = plt.imshow(
        np.abs(E_output[p].T),
        # extent=[np.min(theta_in), np.max(theta_in), np.min(sim.z), np.max(sim.z)],
        extent=[np.min(theta_in), np.max(theta_in), np.min(sim.z), np.max(sim.z)],
    )

    plt.gca().set_aspect("auto")
    divider = make_axes_locatable(plt.gca())
    plt.xlabel(r"$\theta-\theta_0$")
    plt.ylabel(r"$x / \lambda$")
    plt.title(f"Output field for {fishermatrix.param_names[p]}")
    cax = divider.append_axes("right", size="4%", pad=0.1)
    plt.colorbar(ims, cax=cax)
    plt.show()
# %%
for p in range(3):
    plt.figure()
    ims = plt.imshow(
        np.abs(
            E0[p, :, np.newaxis]
            + dE_dtheta[p, :, np.newaxis] * theta_in[np.newaxis, :]
            - E_output[p].T
        ),
        extent=[np.min(theta_in), np.max(theta_in), np.min(sim.z), np.max(sim.z)],
    )
    # ims = plt.imshow(
    #     # np.abs(theta_in[np.newaxis, :]),
    #     # np.abs(theta_in[np.newaxis, :]),
    #     # np.abs(E0[p, :, np.newaxis]),
    #     # np.abs(dE_dtheta[p, :, np.newaxis]),
    #     extent=[np.min(theta_in), np.max(theta_in), np.min(sim.z), np.max(sim.z)],
    # )

    plt.gca().set_aspect("auto")
    divider = make_axes_locatable(plt.gca())
    plt.xlabel(r"$\theta-\theta_0$")
    plt.ylabel(r"$x / \lambda$")
    plt.title(
        r"$\left|E_{\mathrm{predict}}-E\right|$" + f" for {fishermatrix.param_names[p]}"
    )
    cax = divider.append_axes("right", size="4%", pad=0.1)
    plt.colorbar(ims, cax=cax)
    plt.show()


# %%
"""
E = E0 + d_theta E0 (theta - theta0)
==> theta - theta0 = d_theta E0^* . (E - E0) /|d_theta E0|^2
"""
for p in range(3):
    i = 0
    dtheta = [
        (dE_dtheta[p].conj()) @ (E - E0[p]) / (dE_dtheta[p].conj() @ dE_dtheta[p])
        for E in E_output[p]
    ]
    dtheta = np.real(dtheta)
    dtheta = np.real(dtheta)

    plt.figure()
    plt.plot(theta_in, dtheta)
    theta_exact = theta_in[n_theta // 2 - 30 : n_theta // 2 + 30]
    plt.plot(theta_exact, theta_exact)
    param = fishermatrix.param_names[p]
    plt.xlabel(r"${}$".format(param))
    plt.ylabel(r"predicted ${}$".format(param))
    plt.title(f"{fishermatrix.param_names[p]}")
    # plt.ylim(np.min(dtheta), np.max(dtheta))
    plt.show()
