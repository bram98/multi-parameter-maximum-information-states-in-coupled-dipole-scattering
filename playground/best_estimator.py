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

import sys

sys.exit()

# %%
folder = "../data/double_line7"

sim = loadobj("coupleddipolesimulator", folder)
kx_array = loadobj("kx_array", folder)
base_dipoles, frozen_dipoles = loadobj("dipoles", folder)
fishermatrix = loadobj("fishermatrix", folder)
loss_functions = loadobj("loss_functions", folder)
# %%
n = 200
E_output = np.zeros((3, n, sim.grid_points_x), dtype=complex)

dE_dtheta = np.zeros((3, sim.grid_points_x), dtype=complex)
E0 = np.zeros((3, sim.grid_points_x), dtype=complex)

for p in range(3):
    loss_function = [lf for lf in loss_functions if lf.name == "det"][0]
    wavefront = loss_function.wavefront
    sim.set_input_field_kx(kx_array, wavefront)

    sim.reset()
    sim.scatterers = np.hstack([base_dipoles, frozen_dipoles])
    E0[p] = sim.compute_zL_fields_direct()
    dE_dtheta[p] = fishermatrix.dS[p] @ wavefront

    theta_range = 0.01
    theta_in = np.linspace(-theta_range, theta_range, num=n)
    theta_meas = np.zeros_like(theta_in)

    for i, dtheta in enumerate(theta_in):
        dtheta_vec = dtheta * np.eye(3)[p]
        shifted_dipoles = placedipoles.shift_and_rotate_dipoles(
            base_dipoles, *dtheta_vec, frozen_dipoles=frozen_dipoles
        )
        sim.reset()
        sim.scatterers = shifted_dipoles
        E_output[p, i] = sim.compute_zL_fields_direct()

    plt.figure()
    # ims = plt.imshow(np.abs(E_output[p].T), extent=[np.min(sim.z), np.max(sim.z), np.min(theta_in), np.max(theta_in)])
    ims = plt.imshow(
        np.abs(E_output[p].T),
        extent=[np.min(theta_in), np.max(theta_in), np.min(sim.z), np.max(sim.z)],
    )
    divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    plt.xlabel("z")
    plt.ylabel("displacement")
    plt.colorbar(ims, cax=cax)
    plt.title(f"{fishermatrix.param_names[p]}")
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

    plt.figure()
    plt.plot(theta_in, dtheta)
    plt.plot(theta_in[n // 2 - 30 : n // 2 + 30], theta_in[n // 2 - 30 : n // 2 + 30])
    param = fishermatrix.param_names[p]
    plt.xlabel(r"${}/\lambda$".format(param))
    plt.ylabel(r"predicted ${}/\lambda$".format(param))
    plt.title(f"{fishermatrix.param_names[p]}")
    # plt.ylim(np.min(dtheta), np.max(dtheta))
    plt.show()
