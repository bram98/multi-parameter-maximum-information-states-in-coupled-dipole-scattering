# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:43:43 2025

@author: 5914787
"""
import os
import sys
from pathlib import Path

import dill  # saving objects to disk
import matplotlib
import matplotlib.pyplot as plt

# %%
import numpy as np
import torch
from adjustText import adjust_text
from libraries.fishermatrix import FisherMatrix
from libraries.optimizer import Optimizer
from libraries.ScatteringSimulation import CoupledDipoleSimulator
from matplotlib import animation

from libraries import placedipoles

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
        self,
        optimizer: Optimizer,
        seed=None,
        steps=2000,
        num_eig=1,
        verbose=True,
    ):
        """
        Performs the optimization. If method == 'eig', the optimizer finds the
        eigenvectors and eigenvalues of the matrix returned by loss_function.
        If method == 'torch', the loss_function is optimized using torch. The
        loss_function should return a scalar and during calculations torch
        methods should be used.

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
            self.wavefront = LossFunction.normalize_wavefront(
                result["bestvec"]
            )
            self.bestval = result["bestval"]
            # print(result)
            self.loss_history = result["loss"]

        if verbose:
            print(f"Optimized {self.name: <30}loss: {self.bestval:.5g}")

    def visualize(
        self, sim: CoupledDipoleSimulator, kx_array, title=None, dpi=100
    ):
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
            raise ValueError(
                "Wavefront not defined. Please run optimizer first."
            )

        return self._wavefront

    @wavefront.setter
    def wavefront(self, wavefront):
        self._wavefront = wavefront

    def __str__(self):
        return f"<LossFunction {self.name}>"

    def __repr__(self):
        return f"<LossFunction {self.name}>"


def calc_results(
    loss_functions: list[LossFunction],
    optimizer: Optimizer,
    metric_function: callable,
):
    """
    Calculates the single parameter results using the loss_functions objects.
    The Fisher information matrix J is calculated and passed onto the metric
    function. Returns a list of tuples of the form (loss_function, result)
    where result is returned from metric_function.

    Parameters
    ----------
    loss_functions: list of LossFunction
        List of LossFunction objects.
    optimizer: Optimizer
        The optimizer object.
    metric_function: lambda or function
        Gets passed J and should return the result.

    Returns
    -------
    results: list of tuples (loss_function, result)

    """
    results = []
    for loss_function in loss_functions.values():
        J = optimizer.J_np(loss_function.wavefront)
        results.append((loss_function, metric_function(J)))
    return results


def calc_std_ideal(loss_functions, optimizer):
    return calc_results(
        loss_functions,
        optimizer,
        lambda J: (1 / J[0, 0], 1 / J[1, 1], 1 / J[2, 2]),
    )


def calc_std_joint(loss_functions, optimizer):
    def std(J):
        # print(J)
        Sigma = np.linalg.inv(J)
        return (Sigma[0, 0], Sigma[1, 1], Sigma[2, 2])

    return calc_results(loss_functions, optimizer, std)


def _bar_plot_loss_functions_metrics(
    results,
    metric_labels,
    ax=None,
    title=None,
    ylabel=None,
    dpi=100,
    **plot_kwargs,
):
    newplot = ax is None
    if newplot:
        fig, ax = plt.subplots(dpi=dpi)

    data = []
    wave_abbr = []
    for result in results:
        # print(result)
        loss_function, data_ = result
        data.append(data_)
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
            **plot_kwargs,
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


def bar_plot_J(results, title=None, ax=None, dpi=100, **plot_kwargs):
    _bar_plot_loss_functions_metrics(
        results,
        metric_labels=(r"$J_{xx}$", r"$J_{zz}$", r"$J_{\theta\theta}$"),
        ylabel="STD (Cramér-Rao bound)",
        ax=ax,
        title=title,
        dpi=dpi,
        **plot_kwargs,
    )


def bar_plot_STD_joint(results, title=None, ax=None, dpi=100, **plot_kwargs):
    labels = (r"$\Sigma_{xx}$", r"$\Sigma_{zz}$", r"$\Sigma_{\theta\theta}$")

    _bar_plot_loss_functions_metrics(
        results,
        metric_labels=labels,
        ylabel="STD (Cramér-Rao bound)",
        ax=ax,
        title=title,
        dpi=dpi,
        **plot_kwargs,
    )


def bar_plot_STD_ideal(results, title=None, ax=None, dpi=100, **plot_kwargs):
    labels = (
        r"$1\,/\ J_{xx}$",
        r"$1\,/\ J_{zz}$",
        r"$1\,/\ J_{\theta\theta}$",
    )

    _bar_plot_loss_functions_metrics(
        results,
        metric_labels=labels,
        ylabel="STD (Cramér-Rao bound)",
        ax=ax,
        title=title,
        dpi=dpi,
        **plot_kwargs,
    )


def pareto_optimal(a, b, lowerisbetter=True):
    """
    Returns True if a is better than b. Assumes a, b are a tuple
    (LossFunction, result).
    """
    if lowerisbetter:
        return np.all(np.array(a[1])[:2] < np.array(b[1])[:2])
    else:
        return np.all(np.array(a[1])[:2] > np.array(b[1])[:2])


# def pareto_order(results, lowerisbetter=True):

#     pareto_dict = {
#         result1[0].name: {
#             result2[0].name
#             for result2 in results
#             if pareto_optimal(result1, result2, lowerisbetter)
#         }
#         for result1 in results
#     }
#     for result1 in results:
#         for result2 in results:
#             if pareto_optimal(result1, result2, lowerisbetter):
#                 s = '>'
#             elif pareto_optimal(result2, result1, lowerisbetter):
#                 s = '<'
#             else:
#                 s = '?'
#             print(f'{result1[0].name} {s} {result2[0].name}')
#     return list(toposort.toposort(pareto_dict)), pareto_dict


def calc_pareto_front(results, lowerisbetter=True):
    pareto_front = []

    for candidate in results:
        dominated = False
        for other in results:
            if candidate is other:
                continue
            if pareto_optimal(other, candidate, lowerisbetter=lowerisbetter):
                print(other[0].name, " dominates ", candidate[0].name)
                dominated = True
                break
            else:
                print(candidate[0].name, other[0].name)
        if not dominated:
            pareto_front.append(candidate)
    return pareto_front


def pareto_plot(results, xmax=None, ymax=None, lowerisbetter=True, dpi=100):
    fig, ax = plt.subplots(dpi=dpi)

    points = []
    texts = []
    p_front = calc_pareto_front(results, lowerisbetter=lowerisbetter)
    p_front_names = [result[0].name for result in p_front]

    for i, result in enumerate(results):
        loss_function, data = result
        x = data[0]  # x
        y = data[1]  # z

        marker = "x"
        color = "blue"
        label = "Non-efficient"

        if loss_function.name in p_front_names:
            color = "r"
            label = "Efficient"

        x_out_bounds = not xmax is None and x > xmax
        y_out_bounds = not ymax is None and y > ymax

        if x_out_bounds:
            x = xmax
            marker = matplotlib.markers.CARETRIGHT
            color = "black"
            label = None
        if y_out_bounds:
            y = ymax
            marker = matplotlib.markers.CARETUP
            color = "black"
            label = None
        if x_out_bounds and y_out_bounds:
            marker = "D"

        point = ax.scatter(x, y, s=80, marker=marker, color=color, label=label)
        text = ax.annotate(
            loss_function.shortname,
            (x, y),
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=1.0
            ),
            zorder=10,
        )
        points.append(point)
        texts.append(text)
    # plt.scatter(4e-5, 6e-5, color='red', marker='v')
    ax.set_xlabel("CRLB x")
    ax.set_ylabel("CRLB z")
    plt.xlim(None, xmax)
    plt.ylim(None, ymax)
    ax.set_title("Pareto Front of Cramér-Rao Lower bounds")

    ax.grid(True)

    # delete duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    adjust_text(
        texts,
        ax=ax,
        # only_move={'text':'xy', 'explode':'xy'}, # allow text to move in both directions, points only in y
        arrowprops=dict(arrowstyle="-", lw=0.5, alpha=1.0, zorder=0),
        expand=(2.5, 2.5),  # how much to "inflate" the text boxes,
        avoid_self=True,
        force_text=0.5,  # strength of repulsion
        force_static=0.1,
        lim=10,  # max iterations
    )

    plt.show()


# calc_pareto_front(std_joint)
# pareto_plot(std_joint, xmax=5e-5, ymax=2.0e-5, dpi=300)


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

save_state = False
# %% 1) define some dipoles in a line
# base_dipoles = placedipoles.square_of_dipoles(center=(4.0, 0.0), side=1.0)
base_dipoles = placedipoles.line_of_dipoles(center=(5.0, 0), number=7)
frozen_dipoles = placedipoles.line_of_dipoles(center=(3.0, 0), number=7)
# frozen_dipoles = [[], []]

Ns = len(base_dipoles[0]) + len(frozen_dipoles[0])

if save_state:
    saveobj((base_dipoles, frozen_dipoles), "dipoles", folder)

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

if save_state:
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
    flux_normalize=True,
)

fishermatrix = FisherMatrix(dS, param_names=["x", "z", "theta"], delta=delta)
filename = fishermatrix.filename(name=sim_name)

# if False:
#     fishermatrix.save(filename=filename, folder="../data/simulation_scatter_deriv")

F = fishermatrix.F
optimizer = Optimizer(F)

if save_state:
    saveobj(fishermatrix, "fishermatrix", folder)


# %%
LossFunction.photon_budget = 1.0

loss_functions = {}

# Single parameters
for i, param in enumerate(["x", "z", r"\theta"]):
    name = param.replace("\\", "")
    loss_functions[name] = LossFunction(
        lambda F, i_=i: F[i_, i_],
        name=name,
        displayname=f"Singe parameter ${param}$",
        shortname=f"$J_{{{param}}}$",
        maximize=True,
        method="eig",
    )

# Trace
loss_functions["tr"] = LossFunction(
    lambda F: F[0, 0] + F[1, 1] + F[2, 2],
    "tr",
    displayname="Trace",
    shortname=r"$\operatorname{tr}J$",
    maximize=True,
    method="eig",
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


def tr_norm_nuissance(F):
    Fnorm = F_norm(F)
    return Fnorm[0, 0] + Fnorm[1, 1] - Fnorm[2, 2]


loss_functions["tr_norm"] = LossFunction(
    tr_norm,
    "tr_norm",
    displayname="Normalized Trace",
    shortname=r"$\operatorname{tr}_N J$",
    maximize=True,
    method="eig",
)

loss_functions["tr_norm_nuissance"] = LossFunction(
    tr_norm_nuissance,
    "tr_norm_nuissance",
    displayname="Nuissance Trace",
    shortname=r"$\operatorname{tr}_{N}^* J$",
    maximize=True,
    method="eig",
)

# Determinant
loss_functions["det"] = LossFunction(
    lambda J: torch.linalg.det(J),
    "det",
    displayname="Determinant",
    shortname=r"$\operatorname{det} J$",
    maximize=True,
    method="torch",
)


# Spectral Radius
def spectral_radius(J):
    eigvals = torch.linalg.eigvalsh(J)
    return eigvals[0]  # return smallest eigenvalue


loss_functions["spec_rad"] = LossFunction(
    spectral_radius,
    "spec_rad",
    displayname="Spectral Radius",
    shortname=r"$\rho(J)$",
    maximize=True,
    method="torch",
)


# %%
"""
Optimize
"""

for loss_function in loss_functions.values():
    loss_function.optimize(optimizer, seed=0, steps=2000, verbose=True)

# %%
"""
Visualize
"""
for loss_function in loss_functions.values():
    loss_function.visualize(base_sim, kx_array)

# %%
"""
Save wavefronts
"""

if save_state:
    saveobj(loss_functions, "loss_functions", folder)

# %%

std_joint = calc_std_joint(loss_functions, optimizer)
std_ideal = calc_std_ideal(loss_functions, optimizer)

fig, ax = plt.subplots()

bar_plot_STD_joint(
    std_joint,
    dpi=75,
    title=r"Single Param. Performance (joint)",
    ax=ax,
    alpha=0.4,
)

plt.gca().set_prop_cycle(None)
bar_plot_STD_ideal(
    std_ideal, dpi=75, title=r"Single Param. Performance (ideal)", ax=ax
)
plt.show()
