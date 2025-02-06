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
"""

# %% Imports and functions

import matplotlib.pyplot as plt
import numpy as np
import torch
from ScatteringSimulation import CoupledDipoleSimulator


def place_square_of_dipoles(center=(4.0, 0.0), side=1.0):
    """
    Return an array shape(2,4) with the x,z coords of 4 dipoles forming a square,
    centered near 'center' (z=?, x=?), side length 'side'.
    We'll interpret: scatterers[i] = (z_i, x_i).
    """
    half = side / 2
    corners = np.array([(-half, -half), (-half, +half), (+half, -half), (+half, +half)])
    corners[:, 0] += center[0]
    corners[:, 1] += center[1]
    return corners.T  # shape(2,4)


def place_line_of_dipoles(center=(4.0, 0.0), dist=1.0, number=5):
    """
    Return an array shape(2, number) with the x,z coords of a line of dipoles.
    We'll interpret: scatterers[i] = (z_i, x_i).
    """
    pos = np.zeros((2, number))
    for i in range(number):
        pos[1, i] = i * dist - (number // 2 * dist) + center[1]
        pos[0, i] = center[0]
    return pos


def shift_and_rotate_square(original_positions, dx, dz, dtheta):
    """
    original_positions: shape(2,N), each col is one dipole
    dx, dz: translations in x,z
    dtheta: rotation in radians about the shape's centroid
    Return the new positions shape(2,N).
    """
    meanz = np.mean(original_positions[0, :])
    meanx = np.mean(original_positions[1, :])
    temp_z = original_positions[0, :] - meanz
    temp_x = original_positions[1, :] - meanx
    cosT = np.cos(dtheta)
    sinT = np.sin(dtheta)
    z_rot = cosT * temp_z - sinT * temp_x
    x_rot = sinT * temp_z + cosT * temp_x
    z_rot += meanz
    x_rot += meanx
    z_rot += dz
    x_rot += dx
    return np.vstack((z_rot, x_rot))


def partial_derivative_3params(
    sim_kwargs, angles_deg, base_positions, param_index=0, delta=1e-3
):
    """
    param_index in {0,1,2} => which param do we shift?
      0 => x translation
      1 => z translation
      2 => rotation
    Return (T_plus, T_minus, x_cam) for that param.

    We'll do a CoupledDipoleSimulator with Ns=... and custom scatterers for each +/- shift.
    """
    if param_index == 0:
        dx_minus, dz_minus, dth_minus = (-delta, 0.0, 0.0)
        dx_plus, dz_plus, dth_plus = (+delta, 0.0, 0.0)
    elif param_index == 1:
        dx_minus, dz_minus, dth_minus = (0.0, -delta, 0.0)
        dx_plus, dz_plus, dth_plus = (0.0, +delta, 0.0)
    else:
        dx_minus, dz_minus, dth_minus = (0.0, 0.0, -delta)
        dx_plus, dz_plus, dth_plus = (0.0, 0.0, +delta)

    # T_minus
    scatterers_minus = shift_and_rotate_square(
        base_positions, dx_minus, dz_minus, dth_minus
    )
    sim_minus = CoupledDipoleSimulator(**sim_kwargs)
    sim_minus.scatterers = scatterers_minus
    S_minus, x_cam = sim_minus.build_scattering_matrix(angles_deg, flux_normalize=True)

    # T_plus
    scatterers_plus = shift_and_rotate_square(
        base_positions, dx_plus, dz_plus, dth_plus
    )
    sim_plus = CoupledDipoleSimulator(**sim_kwargs)
    sim_plus.scatterers = scatterers_plus
    S_plus, _ = sim_plus.build_scattering_matrix(angles_deg, flux_normalize=True)

    return S_plus, S_minus, x_cam


# %% 1) define some dipoles in a line
base_dipoles = place_square_of_dipoles(center=(4.0, 0.0), side=1.0)
# base_dipoles = place_line_of_dipoles(number=7)

# %% 2) simulator geometry and input space
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

# %% 3) Build partial derivatives for each param => dS[mu]
param_names = ["shiftX", "shiftZ", "rotation"]
delta_param = 1e-3
dS = []
for i in range(3):
    S_plus, S_minus, x_cam = partial_derivative_3params(
        sim_kwargs, angles_deg, base_dipoles, param_index=i, delta=delta_param
    )
    S_deriv = (S_plus - S_minus) / (2 * delta_param)
    dS.append(S_deriv)
    print(f"Built derivative wrt {param_names[i]}, shape T_deriv = {S_deriv.shape}")

# Now we want a 3x3 "tensor" F[mu,nu,:,:].
# F[mu,nu] = 1/2( dS[mu]^dagger dS[nu] + dS[nu]^dagger dS[mu] ) => shape(N_in,N_in).
N_in = dS[0].shape[1]
F = np.zeros((3, 3, N_in, N_in), dtype=complex)

# %% 4) Build the full multi-parameter Fisher-operator array F[mu,nu,:,:]
for mu in range(3):
    for nu in range(3):
        # F_{mu,nu} = 1/2( dS[mu]^dagger dS[nu] + dS[nu]^dagger dS[mu] )
        # dS[mu] is shape (N_out, N_in). So dS[mu]^\dagger dS[nu] is (N_in,N_in).
        tmp1 = dS[mu].conjugate().T @ dS[nu]
        tmp2 = dS[nu].conjugate().T @ dS[mu]
        F[mu, nu, :, :] = 0.5 * (tmp1 + tmp2)

print("Built F[mu,nu,:,:], shape =", F.shape, "(3,3,N_in,N_in)")

# %% 5) Single-parameter operators => F[0,0,:,:], F[1,1,:,:], F[2,2,:,:]
#    Diagonalize each, plot eigenvalues, pick top eigenvector => MIS
base_sim = CoupledDipoleSimulator(**sim_kwargs)
base_sim.scatterers = base_dipoles
k0 = base_sim.k
kx_array = k0 * np.sin(np.deg2rad(angles_deg))

plt.figure()
for i in range(3):
    F_single = F[i, i, :, :]
    vals, vecs = np.linalg.eigh(F_single)
    eig_sorted = np.sort(vals)[::-1]
    plt.semilogy(eig_sorted, label=f"{param_names[i]}")

plt.title("Eigenvalues (descending) for each single-parameter F_op")
plt.xlabel("Eigenvalue index")
plt.ylabel("Value (log scale)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# MIS for each param
for i in range(3):
    F_single = F[i, i, :, :]
    vals, vecs = np.linalg.eigh(F_single)
    idx_sort = np.argsort(vals)[::-1]
    lam_max = vals[idx_sort[0]]
    v_max = vecs[:, idx_sort[0]]
    print(f"\nSingle-param MIS => {param_names[i]}, largest eigenvalue = {lam_max:.3g}")

    base_sim.reset()
    base_sim.set_input_field_kx(kx_array, v_max)
    base_sim.solve_fields()

    print(f"Visualizing MIS for param {param_names[i]} ...")
    base_sim.visualize_intensity()

# %% 6) Combined wavefront => sum diagonal blocks (trace)
F_sum = F[0, 0, :, :] + F[1, 1, :, :] + F[2, 2, :, :]
vals_sum, vecs_sum = np.linalg.eigh(F_sum)
idx_sum = np.argsort(vals_sum)[::-1]
lam_sum_max = vals_sum[idx_sum[0]]
v_sum_max = vecs_sum[:, idx_sum[0]]
print(f"\nCombined param wavefront => largest eigenvalue = {lam_sum_max:.3g}")

# 7) Visualize that combined wavefront
base_sim.reset()
base_sim.set_input_field_kx(kx_array, v_sum_max)
base_sim.solve_fields()
base_sim.visualize_intensity()

print("Done.")

# %% 7) Optimize Determinant using torch

losses = []


def fisher_3x3(E_vec, F_tensor):
    """
    E_vec: shape (N_in,2) => real, imag parts
    F_tensor: shape(3,3,N_in,N_in), complex (torch.complex128)
    Returns 3x3 real Fisher matrix J_{mu,nu} = E^dagger F_{mu,nu} E
    """
    N_in = E_vec.shape[0]
    E_c = E_vec[:, 0] + 1j * E_vec[:, 1]  # shape (N_in,)

    J_mat = torch.zeros((3, 3), dtype=torch.float64, device=E_vec.device)
    for mu in range(3):
        for nu in range(3):
            Fmn = F_tensor[mu, nu, :, :]  # shape(N_in,N_in)
            # x^\dagger Fmn x
            val = (torch.conj(E_c).unsqueeze(0) @ (Fmn @ E_c.unsqueeze(1)))[0, 0]
            J_mat[mu, nu] = val.real  # discard tiny imaginary part (numerical artifact)
    return J_mat


def example_optimize_det(
    F_tensor, N_in, steps=500, lr=0.02, photon_budget=1e6,
):
    """
    Gradient-based optimization of det( J(E) ),
    with a renormalization step each iteration so total power = photon_budget.

    F_tensor: shape(3,3,N_in,N_in), torch.complex128
    N_in: dimension of the input wavefront
    steps: number of Adam steps
    lr: learning rate
    photon_budget: total power (or # of photons). We'll interpret as sum of intensities = photon_budget.
    """
    # Create random initial E in real+imag
    E_param = torch.randn((N_in, 2), dtype=torch.float64, requires_grad=True)
    # We'll interpret sum of squares of E_param as the total amplitude^2
    # which we want to be 'photon_budget'. So we do an initial normalization:
    with torch.no_grad():
        norm_init = torch.linalg.norm(E_param)
        E_param *= photon_budget ** 0.5 / norm_init

    opt = torch.optim.Adam([E_param], lr=lr)

    best_det_val = None
    best_vec = None

    for step in range(steps):
        opt.zero_grad()
        # compute J
        J = fisher_3x3(E_param, F_tensor)  # shape(3,3)
        det_val = torch.linalg.det(J)
        loss = -det_val  # maximize determinant => minimize negative
        loss.backward()
        opt.step()

        # -- Renormalize to fix total power:
        with torch.no_grad():
            cur_norm = torch.linalg.norm(E_param)
            E_param *= photon_budget ** 0.5 / cur_norm

        # track best solution so far
        if best_det_val is None or det_val.item() > best_det_val:
            best_det_val = det_val.item()
            best_vec = E_param.detach().clone()
            losses.append(-best_det_val)

        if step % 10 == 0:
            print(f"Step {step:4d} => det={det_val.item():.0f}")

    # final best solution
    return best_vec.cpu().numpy(), best_det_val


# Convert F into torch
F_tensor_torch = torch.from_numpy(F).to(torch.complex128)  # shape(3,3,N_in,N_in)
E_optim, best_det = example_optimize_det(
    F_tensor_torch, N_in=N_in, photon_budget=10, steps=100, lr=1e-3
)
print(f"\nOptimization done. Best determinant found = {best_det:.5f}")

# parse E_optim -> complex wavefront
E_opt_complex = E_optim[:, 0] + 1j * E_optim[:, 1]

# If you want to visualize it, feed it to a simulator
final_sim = CoupledDipoleSimulator(**sim_kwargs)
final_sim.scatterers = base_dipoles

# We'll define angles -> k_x array
k0 = final_sim.k
kx_array = k0 * np.sin(np.deg2rad(angles_deg))

# plot loss history
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Determinant (Fisher) or -loss")
plt.show()

# Set the wavefront
final_sim.set_input_field_kx(kx_array, E_opt_complex)
final_sim.solve_fields()

print("\nVisualizing the determinant-optimized wavefront's intensity:")
final_sim.visualize_intensity()

# %% 8) Compare and visualize final results for each wavefront

import matplotlib.pyplot as plt
import numpy as np
import torch

########################################
# 1) Functions to compute J-matrix and to produce bar-chart comparisons
########################################


def compute_fisher_matrix_for_wavefront(wavefront, F_tensor):
    """
    wavefront: 1D complex array of shape (N_in,)
    F_tensor:  shape(3,3,N_in,N_in) that defines F_{mu,nu}
               stored as a torch.complex128 tensor
    Returns a 3x3 real numpy array, J_{mu,nu} = wavefront^\dagger F_{mu,nu} wavefront.
    """
    # 1) convert wavefront -> (N_in,2) real+imag
    real_part = np.real(wavefront)
    imag_part = np.imag(wavefront)
    E_np = np.column_stack([real_part, imag_part])  # shape(N_in,2)
    E_torch = torch.tensor(E_np, dtype=torch.float64)  # for fisher_3x3 usage

    # 2) "fisher_3x3" logic, adapted inline
    N_in = E_torch.shape[0]
    E_c = E_torch[:, 0] + 1j * E_torch[:, 1]  # shape(N_in,)
    J_mat_torch = torch.zeros((3, 3), dtype=torch.float64)
    for mu in range(3):
        for nu in range(3):
            Fmn = F_tensor[mu, nu, :, :]  # shape(N_in,N_in) complex
            val = (torch.conj(E_c).unsqueeze(0) @ (Fmn @ E_c.unsqueeze(1)))[0, 0]
            J_mat_torch[mu, nu] = val.real  # discard small imaginary part
    return J_mat_torch.detach().cpu().numpy()


def visualize_multi_param_fisher_comparison(
    wavefronts_dict, F_tensor_torch, figure_title="Comparison of Wavefronts"
):
    """
    wavefronts_dict: { "Label": wavefront_complex_array, ... }
    F_tensor_torch:  shape(3,3,N_in,N_in) complex (Torch)
    We'll compute each wavefront's 3x3 J-matrix, and plot:
      - bar chart of (J_xx, J_zz, J_{theta,theta}) diagonals
      - bar chart of trace(J) and det(J)
    """
    param_labels = ["x", "z", "theta"]  # for referencing J_{xx}, J_{zz}, J_{tt}
    # gather data
    results = {}
    wave_names = list(wavefronts_dict.keys())

    for wname in wave_names:
        wf_cplx = wavefronts_dict[wname]  # shape (N_in,) complex
        J_mat = compute_fisher_matrix_for_wavefront(wf_cplx, F_tensor_torch)
        diag_vals = [J_mat[0, 0], J_mat[1, 1], J_mat[2, 2]]
        trace_val = np.trace(J_mat)
        det_val = np.linalg.det(J_mat)
        results[wname] = {
            "diag": diag_vals,
            "trace": trace_val,
            "det": det_val,
            "J_mat": J_mat,
        }

    # Build arrays for bar plots
    n_wf = len(wave_names)
    diag_data = np.array([results[w]["diag"] for w in wave_names])  # shape(n_wf,3)
    trace_data = [results[w]["trace"] for w in wave_names]
    det_data = [results[w]["det"] for w in wave_names]

    # 1) Bar chart for diagonal
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: diagonal elements
    x_idx = np.arange(n_wf)
    width = 0.25
    for i in range(3):
        offset = (i - 1) * width
        axes[0].bar(
            x_idx + offset,
            diag_data[:, i],
            width=width,
            label=f"J_{{{param_labels[i]}{param_labels[i]}}}",
        )
    axes[0].set_title("Diagonal elements of J")
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(wave_names, rotation=25)
    axes[0].legend()
    axes[0].grid(True, linestyle=":")
    axes[0].set_ylabel("Fisher diag value")

    # Right: trace & determinant
    offset_2 = 0.35
    axes[1].bar(
        x_idx - offset_2 / 2,
        trace_data,
        width=offset_2,
        color="skyblue",
        label="Trace(J)",
    )
    axes[1].bar(
        x_idx + offset_2 / 2,
        det_data,
        width=offset_2,
        color="lightcoral",
        label="Det(J)",
    )
    axes[1].set_title("Trace and Determinant")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(wave_names, rotation=25)
    axes[1].legend()
    axes[1].grid(True, linestyle=":")
    axes[1].set_ylabel("Value")

    plt.suptitle(figure_title)
    plt.tight_layout()
    plt.show()


########################################
# 2) Usage Example
########################################
def compare_final_wavefronts_example(
    F_tensor_torch, v_max_x, v_max_z, v_max_rot, v_sum_max, E_opt_complex
):
    """
    We'll place these wavefronts into a dictionary, then call the comparison function.
    """
    wavefronts_dict = {
        "MIS_x": v_max_x,
        "MIS_z": v_max_z,
        "MIS_rot": v_max_rot,
        "SumDiag": v_sum_max,
        "DetMax": E_opt_complex,
    }
    visualize_multi_param_fisher_comparison(
        wavefronts_dict,
        F_tensor_torch,
        figure_title="Comparison: single-param, sum-of-diag, det-optimized",
    )


# End of block.
# In your script, you'd do:
# compare_final_wavefronts_example(F_tensor_torch, wavefront_x, wavefront_z, wavefront_rot, wavefront_sum, wavefront_det)
