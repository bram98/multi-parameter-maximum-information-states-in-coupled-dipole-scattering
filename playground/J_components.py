# -*- coding: utf-8 -*-
"""
Created on Fri May 16 09:57:51 2025

@author: 5914787
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
from scipy import stats
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

optimizer = Optimizer(fishermatrix.F)
# optimizer.Ft = None
name = "tr"
result = optimizer.torch_optimize_loss(
    loss_functions[name].loss_function,
    record_vec_history=True,
    maximize=True,
    seed=None,
    steps=2000,
    lr=0.001,
)

# %%

J_history = []
J_inv_history = []
vec_history = result["vec_history"]

for vec in vec_history[:]:
    J = optimizer.J_np(vec)
    J_history.append(J)
    J_inv_history.append(np.linalg.inv(J))

J_history = np.array(J_history)
J_inv_history = np.array(J_inv_history)

# %%

plt.figure(dpi=300)
for i, j in ([0, 0], [1, 1], [2, 2]):
    plt.plot(np.abs(J_history[:, i, j]), label=f"$|\,J_{{{i},{j}}}|$")

for i, j in ([0, 1], [0, 2], [1, 2]):
    plt.plot(np.abs(J_history[:, i, j]), label=f"$|\,J_{{{i},{j}}}|$", ls="--")

# plt.yscale('log')
plt.legend(ncols=2)
plt.title(f"Fisher Information During Trace Optimization")
# plt.title(f'Fisher Information During Determinant Optimization')
plt.xlabel("Steps")
# plt.title(f'Fisher Information During {name} Optimization')
plt.show()

# %%

plt.figure()
for i, j in ([0, 0], [1, 1], [2, 2]):
    plt.plot(np.abs(J_inv_history[:, i, j]), label=f"$|\,J_{{{i},{j}}}|$")

for i, j in ([0, 1], [0, 2], [1, 2]):
    plt.plot(np.abs(J_inv_history[:, i, j]), label=f"$|\,J_{{{i},{j}}}|$", ls="--")

plt.yscale("log")
plt.legend(ncols=2)
plt.title("Inverse of Fisher Information During Determinant Optimization")
plt.show()
