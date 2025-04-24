# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:43:43 2025

@author: 5914787
"""
# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import numpy as np
import torch
import dill  # saving objects to disk
from pathlib import Path
from libraries.ScatteringSimulation import CoupledDipoleSimulator
from libraries import placedipoles
from libraries.fishermatrix import FisherMatrix
from libraries.optimizer import Optimizer

import sys

sys.exit()

# %%


class LossFunction:
    """
    Class for handling loss functions conveniently. Expects a loss function
    that takes in the Fisher matrix and returns a scalar

    Parameters
    ----------
    loss_function : function or lambda expression
        Expects a Fisher matrix F and should return a scalar in case method=='torch'
        and should return a matrix in case method=='eig'.
    name: str
        Name of method. Used in printing.
    displayname: str, optional
        Used in plots. If not provided, `name` will be used.
    shortname: str, optional
        Used as an abbreviation. Can be used in plots along the axis. Not used
        by this class.
    maximize: bool, default: True
        If True the loss_function will be maximized, otherwise minimized.
    method: {'eig', 'torch'}
        For 'eig', the optimized will calculate the largest eigenvalue of the
        matrix returned by loss_function. Otherwise it will use torch
        optimization.
    """

    _optim_methods = ("eig", "torch")
    photon_budget = 1.0

    def __init__(
        self,
        loss_function,
        name,
        displayname=None,
        shortname=None,
        maximize=True,
        method="eig",
    ):
        self.loss_function = loss_function
        self.name = name
        if shortname is None:
            shortname = name
        self.shortname = shortname
        if displayname is None:
            displayname = name
        self.displayname = displayname
        self.maximize = maximize
        if not method in LossFunction._optim_methods:
            raise ValueError(
                f"Method '{method}' not supported. Please use "
                "one of " + str(LossFunction._optim_methods)
            )
        self.method = method

        self._wavefront = None
        self.bestval = None
        self.loss_history = None

    @staticmethod
    def normalize_wavefront(wavefront):
        """
        Normalizes wavefront. Use the class variable LossFunction.photon_budget
        to determine the final scale.
        """

        norm_sq = np.sum(np.abs(wavefront) ** 2)

        if norm_sq <= 0.0:
            return wavefront

        return wavefront * np.sqrt(LossFunction.photon_budget / norm_sq)

    def optimize(
        self, optimizer: Optimizer, seed=None, steps=2000, num_eig=1, verbose=True
    ):
        """
        Performs the optimization. If method == 'eig', the optimizer find the
        eigenvectors and eigenvalues of the matrix returned by loss_function.
        If method == 'torch', the loss_function is optimized using torch. It
        should return a scalar and torch methods should be used for calculations.

        Parameters
        ----------
        optimizer: Optimizer
            The optimizer from the libraries.optimizer class
        seed: int, default: None
            Provide a seed for torch optimizer to make the run deterministic.
        steps: int, default: 2000
            Number of steps for torch optimizer
        num_eig: int, default: 1
            How many eigenvalues/eigenvectors to return for the eig optimizer
        verbose: bool, default: True
            Enable printing.
        """

        if self.method == "eig":
            bestval, wavefront = optimizer.eig_optimize_F(
                self.loss_function, maximize=self.maximize, num=num_eig
            )
            self.wavefront = LossFunction.normalize_wavefront(wavefront)
            self.bestval = bestval
        elif self.method == "torch":
            result = optimizer.torch_optimize_loss(
                self.loss_function,
                steps=steps,
                maximize=self.maximize,
                verbose=verbose,
                seed=seed,
            )
            self.wavefront = LossFunction.normalize_wavefront(result["bestvec"])
            self.bestval = result["bestval"]
            # print(result)
            self.loss_history = result["loss"]

        if verbose:
            print(f"Optimized {self.name: <30}loss: {self.bestval:.5g}")

    def visualize(self, sim: CoupledDipoleSimulator, kx_array, title=None, dpi=100):
        """
        Plot wavefront and scatterers.

        Parameters
        ----------
        sim: CoupledDipoleSimulator
            The simulator that is used
        kx_array: np.array
            Input kx_vectors. Can be seen as input angles. Must have same
            dimension as wavefront
        title: str, default: None
            Title to be used on plot
        dpi: int, default: 100
            dpi to be used in plot.

        """
        sim.reset()
        sim.set_input_field_kx(kx_array, self.wavefront)
        sim.solve_fields()

        if title is None:
            title = f"{self.displayname} Optimized Wavefront"

        sim.visualize_intensity(title=title, dpi=dpi)

    @property
    def wavefront(self):
        if self._wavefront is None:
            raise ValueError("Wavefront not defined. Please run optimizer first.")

        return self._wavefront

    @wavefront.setter
    def wavefront(self, wavefront):
        self._wavefront = wavefront


def _bar_loss_functions_metrics(
    loss_functions,
    optimizer: Optimizer,
    metric_functions,
    metric_labels,
    ax=None,
    title=None,
    ylabel=None,
    dpi=100,
):
    newplot = ax is None
    if newplot:
        fig, ax = plt.subplots(dpi=dpi)

    data = []
    wave_abbr = []
    for loss_function in loss_functions:
        J = optimizer.J_np(loss_function.wavefront)
        data.append(metric_functions(J))
        wave_abbr.append(loss_function.shortname)

    data = np.array(data)
    n = len(data)

    x = np.arange(n)
    width = 0.25
    for m in range(len(data[0])):
        offset = (m - 1) * width
        ax.bar(
            x + offset,
            data[:, m],
            width=width,
            label=metric_labels[m],
        )

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(wave_abbr, rotation=0)
    ax.legend()
    ax.grid(True, linestyle=":")
    ax.set_ylabel(ylabel)

    if newplot:
        plt.tight_layout()
        plt.show()


def bar_single_param_fisher(loss_functions, optimizer, title=None, ax=None, dpi=100):
    _bar_loss_functions_metrics(
        loss_functions,
        optimizer,
        lambda J: (J[0, 0], J[1, 1], J[2, 2]),
        metric_labels=(r"$J_{xx}$", r"$J_{zz}$", r"$J_{\theta\theta}$"),
        ylabel="STD (Cramér-Rao bound)",
        ax=ax,
        title=title,
        dpi=dpi,
    )


def bar_single_param_STD(loss_functions, optimizer, title=None, ax=None, dpi=100):
    def std(J):
        Sigma = np.linalg.inv(J)
        return (Sigma[0, 0], Sigma[1, 1], Sigma[2, 2])

    labels = (r"$\Sigma_{xx}$", r"$\Sigma_{zz}$", r"$\Sigma_{\theta\theta}$")

    _bar_loss_functions_metrics(
        loss_functions,
        optimizer,
        std,
        metric_labels=labels,
        ylabel="STD (Cramér-Rao bound)",
        ax=ax,
        title=title,
        dpi=dpi,
    )


def bar_single_param_STD_naive(loss_functions, optimizer, title=None, ax=None, dpi=100):
    labels = (r"$1\,/\ J_{xx}$", r"$1\,/\ J_{zz}$", r"$1\,/\ J_{\theta\theta}$")

    _bar_loss_functions_metrics(
        loss_functions,
        optimizer,
        lambda J: (1 / J[0, 0], 1 / J[1, 1], 1 / J[2, 2]),
        metric_labels=labels,
        ylabel="STD (Cramér-Rao bound)",
        ax=ax,
        title=title,
        dpi=dpi,
    )


def saveobj(obj, name, folder):
    with open(Path(folder) / Path(name), "wb") as file:
        dill.dump(obj, file, dill.HIGHEST_PROTOCOL)


def loadobj(name, folder):
    with open(Path(folder) / Path(name), "rb") as file:
        obj = dill.load(file)
    return obj


# %%

folder = "../data/double_line7"
folder = Path(folder)

if not folder.exists():
    folder.mkdir()
    print(f"Created folder {folder.absolute()}")
# %% 1) define some dipoles in a line
# base_dipoles = placedipoles.square_of_dipoles(center=(4.0, 0.0), side=1.0)
base_dipoles = placedipoles.line_of_dipoles(center=(5.0, 0), number=7)
frozen_dipoles = placedipoles.line_of_dipoles(center=(3.0, 0), number=7)
# frozen_dipoles = [[], []]

saveobj((base_dipoles, frozen_dipoles), "dipoles", folder)

Ns = len(base_dipoles[0]) + len(frozen_dipoles[0])
# %% 2) simulator geometry and input space
sim_kwargs = dict(
    lambda_0=1.0,
    Lz=8.0,
    Lx=8.0,
    Ns=Ns,
    seed=123,
    grid_points_x=300,
    theta_inc_deg=0.0,
)
max_angle = 80
angles_deg = np.linspace(-max_angle, max_angle, 100)

# We'll create a base simulator for the wavefront injection steps later:
base_sim = CoupledDipoleSimulator(**sim_kwargs)

base_sim.scatterers = np.hstack([base_dipoles, frozen_dipoles])
# k0 = base_sim.k
# kx_array = k0 * np.sin(np.deg2rad(angles_deg))
kx_array = base_sim.theta2k_x(angles_deg, degree=True)

plot_dpi = 100  # 100 for speed, 300 for quality
sim_name = f"double_line{Ns}"
param_names = ["shiftX", "shiftZ", "rotation"]
delta = 1e-5

saveobj(base_sim, "coupleddipolesimulator", folder)
saveobj(kx_array, "kx_array", folder)

# %%
dS = placedipoles.simulate_dS_3params(
    sim_kwargs,
    angles_deg,
    base_dipoles,
    dx=delta,
    frozen_dipoles=frozen_dipoles,
    delta=delta,
    flux_normalize=False,
)

fishermatrix = FisherMatrix(dS, param_names=["x", "z", "theta"], delta=delta)
filename = fishermatrix.filename(name=sim_name)

if False:
    fishermatrix.save(filename=filename, folder="../data/simulation_scatter_deriv")

F = fishermatrix.F
optimizer = Optimizer(F)

saveobj(fishermatrix, "fishermatrix", folder)


# %%
LossFunction.photon_budget = 1.0

loss_functions = []

# Single parameters
for i, param in enumerate(["x", "z", r"\theta"]):
    loss_functions.append(
        LossFunction(
            lambda F, i_=i: F[i_, i_],
            name=param.replace("\\", ""),
            displayname=f"Singe parameter ${param}$",
            shortname=f"$J_{{{param}}}$",
            maximize=True,
            method="eig",
        )
    )

# Trace
loss_functions.append(
    LossFunction(
        lambda F: F[0, 0] + F[1, 1] + F[2, 2],
        "tr",
        displayname="Trace",
        shortname=r"$\operatorname{tr}J$",
        maximize=True,
        method="eig",
    )
)


# Normalized Trace
def F_norm(F):
    eig_max = [np.linalg.eigvalsh(F[i, i])[-1] for i in range(3)]
    scale_factors = 1 / np.sqrt(eig_max)
    scale_factors = np.outer(scale_factors, scale_factors)  # Create matrix
    scale_factors = scale_factors[:, :, np.newaxis, np.newaxis]
    return F * scale_factors


def tr_norm(F):
    Fnorm = F_norm(F)
    return Fnorm[0, 0] + Fnorm[1, 1] + Fnorm[2, 2]


loss_functions.append(
    LossFunction(
        tr_norm,
        "tr_norm",
        displayname="Normalized Trace",
        shortname=r"$\operatorname{tr}_N J$",
        maximize=True,
        method="eig",
    )
)

# Determinant
loss_functions.append(
    LossFunction(
        lambda J: torch.linalg.det(J),
        "det",
        displayname="Determinant",
        shortname=r"$\operatorname{det} J$",
        maximize=True,
        method="torch",
    )
)


# Spectral Radius
def spectral_radius(J):
    eigvals = torch.linalg.eigvalsh(J)
    return eigvals[0]  # return smallest eigenvalue


loss_functions.append(
    LossFunction(
        spectral_radius,
        "spec_rad",
        displayname="Spectral Radius",
        shortname=r"$\rho(J)$",
        maximize=True,
        method="torch",
    )
)


# %%
"""
Optimize
"""

for loss_function in loss_functions:
    loss_function.optimize(optimizer, seed=0, steps=2000, verbose=True)

# %%
"""
Visualize
"""
for loss_function in loss_functions:
    loss_function.visualize(base_sim, kx_array)

# %%
"""
Save wavefronts
"""
# filename = f"../data/simulation_scatter_deriv/{sim_name}_wavefronts"
# wavefronts_dict = {}
# print("Saving wavefronts")
# for loss_function in loss_functions:
#     wavefronts_dict[loss_function.name] = loss_function.wavefront

# np.savez(filename, **wavefronts_dict)
saveobj(loss_functions, "loss_functions", folder)

# %%
bar_single_param_STD(
    loss_functions, optimizer, dpi=75, title=r"Single Param.Performance"
)
bar_single_param_STD_naive(
    loss_functions,
    optimizer,
    dpi=75,
    title=r"Single Param. Performance (Without Coupling)",
)
