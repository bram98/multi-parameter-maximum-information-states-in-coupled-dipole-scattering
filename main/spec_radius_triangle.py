# %% Imports and functions

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from scatteringlibs import crlbanalysis, lossfunctions, placedipoles
from scatteringlibs.fishermatrix import FisherMatrix
from scatteringlibs.lossfunctions import LossFunction, LossFunctionsDict
from scatteringlibs.optimizer import Optimizer
from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator

sys.exit()

# %%
load_folder = "../data/quasirandom"
sim = CoupledDipoleSimulator.load(folder=load_folder)
fisher_calc = FisherMatrix.load(folder=load_folder)


# %%
# Spectral Radius
def spectral_radius(J):
    eigvals = torch.linalg.eigvalsh(J)
    return eigvals[0]  # return smallest eigenvalue


def weighted_spectral_radius(J, w):
    w = np.array(w)
    scale_factors = np.outer(w, w)  # Create matrix
    return spectral_radius(J * scale_factors)


# %%
import matplotlib
import mpltern
from matplotlib.tri import Triangulation

M = 100
N = 100

points = []
colors = []

p1 = np.array([-1, 0])
p2 = np.array([1, 0])
p3 = np.array([0, 1])

w = []

c = []


max_prod = 0.0
for w1 in np.linspace(0, 1, num=M):
    for w2 in np.linspace(0, 1, num=N):
        if w1 + w2 > 1:
            continue
        w3 = 1 - w1 - w2
        w.append([w1, w2, w3])
        max_prod = max(max_prod, w1 * w2 * w3)
        c.append(np.sin(w1 * w2 * w3 * 2.5 * 27 * np.pi))

w = np.array(w)
c = np.array(c)

ax = plt.subplot(projection="ternary")
ax.tripcolor(*w.T, c, shading="gouraud", rasterized=True)
plt.show()

# %%
import mpltern

ax = plt.subplot(projection="ternary")
ax.plot(x, 1 - x, x)
plt.show()
