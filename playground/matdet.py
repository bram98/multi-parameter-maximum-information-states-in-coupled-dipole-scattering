# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:50:19 2025

@author: 5914787
"""
from itertools import permutations

import matplotlib.pyplot as plt

# %%
import numpy as np
import scipy
import torch
from libraries.fishermatrix import FisherMatrix
from libraries.optimizer import Optimizer
from libraries.ScatteringSimulation import CoupledDipoleSimulator

from libraries import placedipoles

# import sys
# sys.exit()

# %%

fm = FisherMatrix()
fm.load("../data/simulation_scatter_deriv/line2_x-z-theta_c.npz")

F = fm.F

optimizer = Optimizer(F)


# %%
def matrixdet(F):
    # return F[0,0]
    return 1 / 2 * (F[0, 0] @ F[2, 2] + F[2, 2] @ F[0, 0]) - F[0, 2] @ F[0, 2]


def symprod3(A, B, C):
    return (
        1
        / 6
        * (
            A @ B @ C
            + B @ C @ A
            + C @ A @ B
            + B @ A @ C
            + C @ B @ A
            + A @ C @ B
        )
    )


def prod3(A, B, C):
    return A @ B @ C


def perm_sign(perm):
    sign = 1
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            # print(i, j, perm[i], perm[j], perm[i]>perm[j])
            if perm[i] > perm[j]:
                sign *= -1
    return sign


def matrixdet3(F):
    det = np.zeros_like(F[0, 0])
    for perm in permutations(range(3)):
        sign = perm_sign(perm)
        i, j, k = perm
        det += sign * prod3(F[i, 0], F[j, 1], F[k, 2])
        # det += sign * symprod3(F[i, 0], F[j, 1], F[k, 2])
    return det


def diagprod(F):
    # return F[0,0]
    return (
        1
        / 6
        * (
            F[0, 0] @ F[1, 1] @ F[2, 2]
            + F[1, 1] @ F[0, 0] @ F[2, 2]
            + F[1, 1] @ F[2, 2] @ F[0, 0]
            + F[2, 2] @ F[1, 1] @ F[0, 0]
            + F[2, 2] @ F[0, 0] @ F[1, 1]
            + F[0, 0] @ F[2, 2] @ F[1, 1]
        )
    )


eig_matdet, vec_matdet = optimizer.eig_optimize_F(
    matrixdet, maximize=True, num=None
)
eig_matdet3, vec_matdet3 = optimizer.eig_optimize_F(
    matrixdet3, maximize=True, num=None
)
eig_diagprod, vec_diagprod = optimizer.eig_optimize_F(
    diagprod, maximize=True, num=None
)

plt.plot(eig_matdet, label="matdet")
plt.plot(eig_diagprod, label="diagprod")
plt.plot(eig_matdet3, label="matdet3")
plt.legend()
plt.show()
# %%
result2 = optimizer.torch_optimize_loss(
    lambda J: torch.linalg.det(J[::2, ::2]), steps=2000, maximize=True
)
result3 = optimizer.torch_optimize_loss(
    lambda J: torch.linalg.det(J), steps=2000, maximize=True
)
eig_det2 = result2["bestval"]
vec_det2 = result2["bestvec"]
eig_det3 = result3["bestval"]
vec_det3 = result3["bestvec"]


# %%

base_dipoles = placedipoles.line_of_dipoles(number=7)
sim_kwargs = dict(
    lambda_0=1.0,
    Lz=8.0,
    Lx=8.0,
    Ns=len(base_dipoles.T),
    seed=123,
    grid_points_x=300,
    theta_inc_deg=0.0,
)
max_angle = 80
angles_deg = np.linspace(-max_angle, max_angle, 100)

# We'll create a base simulator for the wavefront injection steps later:
base_sim = CoupledDipoleSimulator(**sim_kwargs)
base_sim.scatterers = base_dipoles

kx_array = base_sim.theta2k_x(angles_deg, degree=True)
plot_dpi = 100

# %%

base_sim.reset()
base_sim.set_input_field_kx(kx_array, vec_matdet3[:, 0])
base_sim.solve_fields()
base_sim.visualize_intensity(title=f"Trace optimized wavefron)", dpi=plot_dpi)


# %%
def visualize_multi_param_fisher_comparison(
    wavefronts_dict, optimizer, figure_title="Comparison of Wavefronts"
):
    global results
    param_labels = ["x", "z", r"\theta"]
    results = {}
    wave_names = list(wavefronts_dict.keys())

    for wname, wavefront in wavefronts_dict.items():
        J_mat = optimizer.J_np(wavefront)
        diag_vals = [J_mat[0, 0], J_mat[1, 1], J_mat[2, 2]]
        trace_val = np.trace(J_mat)
        det_val = np.linalg.det(J_mat)
        results[wname] = {
            "diag": diag_vals,
            "trace": trace_val,
            "det": det_val,
        }

    # print(results['a'])

    n_wf = len(wave_names)
    diag_data = np.array([results[w]["diag"] for w in wave_names])
    trace_data = np.array([results[w]["trace"] for w in wave_names])
    # trace_data /= np.max(trace_data)
    det_data = np.array([results[w]["det"] for w in wave_names])
    # det_data /= np.max(det_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_idx = np.arange(n_wf)
    width = 0.25
    for i in range(3):
        offset = (i - 1) * width
        axes[0].bar(
            x_idx + offset,
            diag_data[:, i],
            width=width,
            label=f"$J_{{{param_labels[i]}{param_labels[i]}}}$",
        )
    axes[0].set_title("Diagonal elements of J")
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(wave_names, rotation=25)
    axes[0].legend()
    axes[0].grid(True, linestyle=":")
    axes[0].set_ylabel("Fisher diag value")

    offset_2 = 0.35

    # Create two y-axes that share x-axis
    ax_tr = axes[1]
    ax_det = ax_tr.twinx()

    tr_color = "skyblue"
    det_color = "lightcoral"

    ax_tr.bar(
        x_idx - offset_2 / 2,
        trace_data,
        width=offset_2,
        color=tr_color,
        label="Trace(J)",
    )
    ax_det.bar(
        x_idx + offset_2 / 2,
        det_data,
        width=offset_2,
        color="lightcoral",
        label="Det(J)",
    )

    axes[1].set_title("Trace and Determinant")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(wave_names, rotation=25)

    lines_tr, labels_tr = ax_tr.get_legend_handles_labels()
    lines_det, labels_det = ax_det.get_legend_handles_labels()
    ax_det.legend(lines_tr + lines_det, labels_tr + labels_det)

    axes[1].grid(True, linestyle=":")
    ax_tr.set_ylabel("Trace(J)")
    ax_det.set_ylabel("Det(J)")
    ax_tr.tick_params(axis="y", labelcolor=tr_color)
    ax_tr.ticklabel_format(
        axis="y", style="sci", useOffset=False, scilimits=(0, 1)
    )
    ax_det.tick_params(axis="y", labelcolor=det_color)

    plt.suptitle(figure_title)
    plt.tight_layout()
    plt.show()


wavefronts_dict = {
    r"$matdet 2x2$": vec_matdet[:, 0],
    r"$det2x2$": vec_det2,
    r"$diag prod$": vec_diagprod[:, 0],
    r"$matdet 3x3 $": vec_matdet3[:, 0],
    "det3x3": vec_det3,
}
visualize_multi_param_fisher_comparison(
    wavefronts_dict, optimizer, figure_title="Comparison of all wavefronts"
)
