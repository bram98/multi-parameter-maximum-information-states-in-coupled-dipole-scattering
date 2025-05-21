#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_fisher_info_and_MIS_3params.py

Demo:
1) Place 4 dipoles in a "square" arrangement.
2) Define 3 parameters: global shift in x, global shift in z, rotation about the square's center.
3) Build scattering matrix S for each input angle channel.
4) Use finite differences to get partial derivatives wrt x, z, rotation.
   Then build the full multi-parameter Fisher-operator tensor F[mu,nu,:,:].
5) Extract the single-parameter Fisher operators F_x, F_z, F_theta => each is F[mu,mu,:,:] for mu in {0,1,2}.
   Diagonalize each partial operator to find the single-parameter MIS. Visualize + semilogy eigenvalues.
6) Combine them to find a single wavefront that is "optimal for all 3" by summing F_x + F_z + F_theta
   or, equivalently, summing F[0,0] + F[1,1] + F[2,2]. Diagonalize => pick top eigenvector => visualize.
7a) Optimize the input state by maximizing the determinant of J (via torch determinant optimization).
7b) Optimize the input state by minimizing the spectral radius of J⁻¹ (i.e. maximizing the smallest eigenvalue of J).
8) Compare and visualize all optimal wavefronts.
9) Compute and compare CRLB and STD for each optimal wavefront.
"""

# %% Imports and functions

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import numpy as np
import torch
from libraries.ScatteringSimulation import CoupledDipoleSimulator
from libraries import placedipoles
from libraries.fishermatrix import FisherMatrix
from libraries.optimizer import Optimizer

# import sys
# sys.exit() # Hack to reload modules and filepath without running whole script


# %% 1) define some dipoles in a line
# base_dipoles = placedipoles.square_of_dipoles(center=(4.0, 0.0), side=1.0)
base_dipoles = placedipoles.line_of_dipoles(center=(5.0, 0), number=7)
# frozen_dipoles = placedipoles.line_of_dipoles(center=(3., 0), number=7)
frozen_dipoles = [[], []]

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

plot_dpi = 300  # 100 for speed, 300 for quality
param_names = ["shiftX", "shiftZ", "rotation"]
delta = 1e-3

# %% 3) Build partial derivatives for each param => dS[mu]

# Simple placement (No frozen dipoles)

dS = placedipoles.simulate_dS_3params(
    sim_kwargs, angles_deg, base_dipoles, dx=delta, frozen_dipoles=frozen_dipoles
)

fishermatrix = FisherMatrix(dS, param_names=["x", "z", "theta"])
filename = fishermatrix.filename(name=f"double_line{Ns}")

if False:
    fishermatrix.save(filename=filename, folder="../data/simulation_scatter_deriv")

F_raw = fishermatrix.F
optimizer_raw = Optimizer(F_raw)
# %%

dS_norm = np.zeros_like(dS)

for i in range(3):
    f_max, E_max = optimizer_raw.eig_optimize_F(lambda F: F[i, i])
    dS_norm[i] = dS[i] / np.sqrt(f_max)

fishermatrix_norm = FisherMatrix(dS_norm, param_names=["x", "z", "theta"])
F_norm = fishermatrix_norm.F

# %%
normalize_dS = False

if normalize_dS:
    F = F_norm
    optimizer = Optimizer(F)
    norm_text = "normalized"
else:
    F = F_raw
    optimizer = Optimizer(F)
    norm_text = ""

# %% 5) Single-parameter MIS wavefronts: store them explicitly for later usage
wavefront_x = None
wavefront_z = None
wavefront_rot = None

for i in range(3):
    f_max, E_max = optimizer.eig_optimize_F(lambda F: F[i, i])

    print(f"\nSingle-param MIS => {param_names[i]}, largest eigenvalue = {f_max:.3g}")
    base_sim.reset()
    base_sim.set_input_field_kx(kx_array, E_max)
    base_sim.solve_fields()
    base_sim.visualize_intensity(
        title=f"Single-Parameter Optimized Wavefront for {param_names[i]} (f={f_max:.3g})",
        dpi=plot_dpi,
    )

    result = optimizer.torch_optimize_loss(
        lambda J: J[i, i], steps=3000, maximize=True, verbose=True, seed=0
    )
    f_max_test = result["bestval"]
    E_max_test = result["bestvec"]

    optimizer.plot_loss(title="Singe-param. with torch", first=500)

    print(f"\nSingle-param MIS => {param_names[i]}, largest eigenvalue = {f_max:.3g}")
    base_sim.reset()
    base_sim.set_input_field_kx(kx_array, E_max_test)
    base_sim.solve_fields()
    base_sim.visualize_intensity(
        title=f"Single-Parameter Optimized Wavefront for {param_names[i]} torch (f={f_max_test:.3g})",
        dpi=plot_dpi,
    )

    # Store them for final comparison
    if i == 0:
        wavefront_x = E_max
        f_x = f_max
        f_x_test = f_max_test
    elif i == 1:
        wavefront_z = E_max
        f_z = f_max
        f_z_test = f_max_test
    else:
        wavefront_rot = E_max
        f_rot = f_max
        f_rot_test = f_max_test


# %% 6) Combined wavefront => sum of diagonal blocks (trace)

f_tr, wavefront_tr = optimizer.eig_optimize_F(lambda F: F[0, 0] + F[1, 1] + F[2, 2])

print(f"\nCombined param wavefront => largest eigenvalue = {f_tr:.3g}")

base_sim.reset()
base_sim.set_input_field_kx(kx_array, wavefront_tr)
base_sim.solve_fields()
base_sim.visualize_intensity(
    title=f"Trace optimized wavefront (f={f_tr:.3g})", dpi=plot_dpi
)


print("Done with single-param and sum-diag solutions.")
###########################################################

# %% 7a) Determinant optimization with torch

print("-- Optimizing Determinant --")


result = optimizer.torch_optimize_loss(
    lambda J: torch.linalg.det(J), steps=2000, maximize=True, verbose=True, seed=0
)
optimizer.plot_loss(title="Determinant of J loss", first=500)
wavefront_det = result["bestvec"]
f_det = result["bestval"]
print(f"\nDet-optimized wavefront, best determinant = {f_det:.3f}")

# E_opt_complex = E_optim_det[:, 0] + 1j * E_optim_det[:, 1]
# %%
base_sim.reset()
base_sim.set_input_field_kx(kx_array, wavefront_det)
base_sim.solve_fields()

base_sim.visualize_intensity(title="Determinant Optimized Wavefront", dpi=plot_dpi)

# publish worthy (?) plotting
# base_sim._visualize(
#     display_function=lambda field: np.abs(field) ** 2,
#     title="Determinant Optimized Wavefront",
#     dpi=300,
#     figsize=(7, 5)
# )

# %% 7b) Spectral Radius Optimization with torch (Minimize maximum eigenvalue of J⁻¹)
print("-- Optimizing Spectral radius --")


def spectral_radius(J):
    eigvals = torch.linalg.eigvalsh(J)
    return eigvals[0]  # return smallest eigenvalue


def spectral_radius_direct(J):
    eigvals = torch.linalg.eigvalsh(torch.inverse(J))
    return eigvals[-1]  # return largest eigenvalue


result = optimizer.torch_optimize_loss(
    spectral_radius, steps=2000, maximize=True, verbose=True, seed=0
)
optimizer.plot_loss(title="Spectral Radius of J loss", first=500)
wavefront_spec = result["bestvec"]
f_spec = result["bestval"]
print(f"\nSpectral Radius-optimized wavefront, best determinant = {f_det:.3f}")

base_sim.reset()
base_sim.set_input_field_kx(kx_array, wavefront_spec)
base_sim.solve_fields()

base_sim.visualize_intensity(title="Spectral Radius Optimized Wavefront", dpi=plot_dpi)


# %% 8) Compare and visualize all final wavefronts


def visualize_multi_param_fisher_comparison(
    wavefronts_dict, optimizer, figure_title="Comparison of Wavefronts"
):
    param_labels = ["x", "z", r"\theta"]
    results = {}
    wave_names = list(wavefronts_dict.keys())

    for wname, wavefront in wavefronts_dict.items():
        J_mat = optimizer.J_np(wavefront)
        diag_vals = [J_mat[0, 0], J_mat[1, 1], J_mat[2, 2]]
        trace_val = np.trace(J_mat)
        det_val = np.linalg.det(J_mat)
        results[wname] = {"diag": diag_vals, "trace": trace_val, "det": det_val}

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
    ax_tr.ticklabel_format(axis="y", style="sci", useOffset=False, scilimits=(0, 1))
    ax_det.tick_params(axis="y", labelcolor=det_color)

    plt.suptitle(figure_title)
    plt.tight_layout()
    plt.show()


# normalizations:
def normalize_wavefront(wavefront_cplx, photon_budget):
    """
    wavefront_cplx: 1D complex array shape (N_in,)
    photon_budget: e.g. 10 or 1e6
    Return wavefront of same shape, scaled so sum(|wavefront|^2) = photon_budget.
    """
    current_norm_sq = np.sum(np.abs(wavefront_cplx) ** 2)
    if current_norm_sq <= 0:
        return wavefront_cplx
    scale_factor = (photon_budget / current_norm_sq) ** 0.5
    return wavefront_cplx * scale_factor


photon_budget = 1.0  # e.g. 10 total "photons"

wavefront_x_norm = normalize_wavefront(wavefront_x, photon_budget)
wavefront_z_norm = normalize_wavefront(wavefront_z, photon_budget)
wavefront_rot_norm = normalize_wavefront(wavefront_rot, photon_budget)
wavefront_tr_norm = normalize_wavefront(wavefront_tr, photon_budget)
# wavefront_tr_w_norm = normalize_wavefront(wavefront_tr_w, photon_budget)
wavefront_det_norm = normalize_wavefront(wavefront_det, photon_budget)
wavefront_spec_norm = normalize_wavefront(wavefront_spec, photon_budget)


wavefronts_dict = {
    r"$J_{x}$": wavefront_x_norm,
    r"$J_{z}$": wavefront_z_norm,
    r"$J_{\theta}$": wavefront_rot_norm,
    r"$\mathrm{tr}\, J$": wavefront_tr_norm,
    # r"$\mathrm{tr}_w\, J$": wavefront_tr_w_norm,
    r"$\mathrm{det}\, J$": wavefront_det_norm,
    "Spectral\nRadius": wavefront_spec_norm,
}
visualize_multi_param_fisher_comparison(
    wavefronts_dict, optimizer_raw, figure_title="Comparison of all wavefronts"
)


# %% 9a) Compare CRLB (i.e. diagonal of J^-1) for each wavefront


def visualize_crlb_comparison(
    wavefronts_dict, optimizer, figure_title="CRLB Comparison", take_sqrt=False
):
    """
    wavefronts_dict: { "Label": wavefront_complex_array, ... }
    We'll bar-plot the diagonal of J^-1 for each wavefront,
    i.e. CRLB_x = (J^-1)[0,0], CRLB_z = (J^-1)[1,1], CRLB_theta = (J^-1)[2,2].
    Lower is better in that sense.
    """
    param_labels = ["CRLB_x", "CRLB_z", "CRLB_theta"]
    wave_names = list(wavefronts_dict.keys())
    results = {}

    for wname, wavefront in wavefronts_dict.items():
        J_mat = optimizer.J_np(wavefront)
        M_inv = np.linalg.inv(J_mat)
        diag_vals = [M_inv[0, 0], M_inv[1, 1], M_inv[2, 2]]
        if take_sqrt:
            diag_vals = np.sqrt(diag_vals)
        results[wname] = diag_vals

    n_wf = len(wave_names)
    diag_data = np.array([results[w] for w in wave_names])

    fig, ax = plt.subplots(figsize=(8, 5))

    x_idx = np.arange(n_wf)
    width = 0.25
    for i in range(3):
        offset = (i - 1) * width
        ax.bar(x_idx + offset, diag_data[:, i], width=width, label=param_labels[i])
    ax.set_title(figure_title)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(wave_names, rotation=25)
    ax.set_ylim(0, 0.030)
    ax.legend()
    ax.grid(True, linestyle=":")
    if take_sqrt:
        ax.set_ylabel("Standard Deviation (units of param)")
    else:
        ax.set_ylabel("CRLB (units of param^2)")
    plt.tight_layout()
    plt.show()


# visualize_crlb_comparison(
#     wavefronts_dict,
#     optimizer_raw,
#     figure_title="CRLB (J^-1 diagonal) for each wavefront",
# )

visualize_crlb_comparison(
    wavefronts_dict,
    optimizer_raw,
    figure_title="STD (sqrt(CRLB)) for each wavefront " + norm_text,
    take_sqrt=True,
)
