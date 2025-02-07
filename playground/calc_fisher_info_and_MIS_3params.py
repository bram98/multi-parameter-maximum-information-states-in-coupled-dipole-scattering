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


def shift_and_rotate_dipoles(original_positions, dx, dz, dtheta):
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
    scatterers_minus = shift_and_rotate_dipoles(
        base_positions, dx_minus, dz_minus, dth_minus
    )
    sim_minus = CoupledDipoleSimulator(**sim_kwargs)
    sim_minus.scatterers = scatterers_minus
    S_minus, x_cam = sim_minus.build_scattering_matrix(angles_deg, flux_normalize=True)

    # T_plus
    scatterers_plus = shift_and_rotate_dipoles(
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

# We'll create a base simulator for the wavefront injection steps later:
base_sim = CoupledDipoleSimulator(**sim_kwargs)
base_sim.scatterers = base_dipoles
k0 = base_sim.k
kx_array = k0 * np.sin(np.deg2rad(angles_deg))

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

# %% 4) Build the full multi-parameter Fisher-operator array F[mu,nu,:,:]

N_in = dS[0].shape[1]
F = np.zeros((3, 3, N_in, N_in), dtype=complex)

for mu in range(3):
    for nu in range(3):
        # F_{mu,nu} = 1/2( dS[mu]^dagger dS[nu] + dS[nu]^dagger dS[mu] )
        # dS[mu] is shape (N_out, N_in). So dS[mu]^\dagger dS[nu] is (N_in,N_in).
        tmp1 = dS[mu].conjugate().T @ dS[nu]
        tmp2 = dS[nu].conjugate().T @ dS[mu]
        F[mu, nu, :, :] = 0.5 * (tmp1 + tmp2)

print("Built F[mu,nu,:,:], shape =", F.shape, "(3,3,N_in,N_in)")

# %% 5) Single-parameter MIS wavefronts: store them explicitly for later usage
wavefront_x = None
wavefront_z = None
wavefront_rot = None

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
    base_sim.visualize_intensity()

    # Store them for final comparison
    if i == 0:
        wavefront_x = v_max
    elif i == 1:
        wavefront_z = v_max
    else:
        wavefront_rot = v_max

# %% 6) Combined wavefront => sum of diagonal blocks
F_sum = F[0, 0, :, :] + F[1, 1, :, :] + F[2, 2, :, :]
vals_sum, vecs_sum = np.linalg.eigh(F_sum)
idx_sum = np.argsort(vals_sum)[::-1]
lam_sum_max = vals_sum[idx_sum[0]]
v_sum_max = vecs_sum[:, idx_sum[0]]
print(f"\nCombined param wavefront => largest eigenvalue = {lam_sum_max:.3g}")

base_sim.reset()
base_sim.set_input_field_kx(kx_array, v_sum_max)
base_sim.solve_fields()
base_sim.visualize_intensity()

print("Done with single-param and sum-diag solutions.")

# %% 7a) Determinant optimization with torch
losses_det = []

def fisher_3x3(E_vec, F_tensor):
    E_c = E_vec[:, 0] + 1j * E_vec[:, 1]
    J_mat = torch.zeros((3, 3), dtype=torch.float64, device=E_vec.device)
    for mu in range(3):
        for nu in range(3):
            Fmn = F_tensor[mu, nu, :, :]
            val = (torch.conj(E_c).unsqueeze(0) @ (Fmn @ E_c.unsqueeze(1)))[0, 0]
            J_mat[mu, nu] = val.real
    return J_mat


def torch_optimize_det(F_tensor, N_in, steps=300, lr=1e-3, photon_budget=10):
    E_param = torch.randn((N_in, 2), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        norm_init = torch.linalg.norm(E_param)
        E_param *= photon_budget ** 0.5 / norm_init

    opt = torch.optim.Adam([E_param], lr=lr)
    best_det_val = None
    best_vec = None

    for step in range(steps):
        opt.zero_grad()
        J = fisher_3x3(E_param, F_tensor)
        det_val = torch.linalg.det(J)
        loss = -det_val
        loss.backward()
        opt.step()

        with torch.no_grad():
            cur_norm = torch.linalg.norm(E_param)
            E_param *= photon_budget ** 0.5 / cur_norm

        if best_det_val is None or det_val.item() > best_det_val:
            best_det_val = det_val.item()
            best_vec = E_param.detach().clone()
        losses_det.append(-det_val.item())

        if step % 10 == 0:
            print(f"Det opt Step {step:4d} => det={det_val.item():.3f}")

    return best_vec.cpu().numpy(), best_det_val

F_tensor_torch = torch.from_numpy(F).to(torch.complex128)
E_optim_det, best_det = torch_optimize_det(
    F_tensor_torch, N_in=N_in, steps=5000, lr=1e-3, photon_budget=10
)
print(f"\nDet-optimized wavefront, best determinant = {best_det:.3f}")

E_opt_complex = E_optim_det[:, 0] + 1j * E_optim_det[:, 1]

base_sim.reset()
base_sim.set_input_field_kx(kx_array, E_opt_complex)
base_sim.solve_fields()

plt.figure()
plt.plot(losses_det)
plt.xlabel("Iteration")
plt.ylabel("-(best determinant) so far")
plt.title("Determinant Optimization Loss")
plt.show()

base_sim.visualize_intensity()

# %% 7b) Spectral Radius Optimization with torch (Minimize maximum eigenvalue of J⁻¹)
losses_spec = []

def torch_optimize_spec(F_tensor, N_in, steps=5000, lr=1e-3, photon_budget=10):
    """
    Optimizes the input wavefront to maximize the smallest eigenvalue of J.
    This is equivalent to minimizing the spectral radius (largest eigenvalue)
    of the covariance matrix J⁻¹.
    """
    E_param = torch.randn((N_in, 2), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        norm_init = torch.linalg.norm(E_param)
        E_param *= (photon_budget ** 0.5) / norm_init

    opt = torch.optim.Adam([E_param], lr=lr)
    best_min_eig = None
    best_vec = None

    for step in range(steps):
        opt.zero_grad()
        J = fisher_3x3(E_param, F_tensor)  # shape: (3,3)
        eigvals = torch.linalg.eigvalsh(J)
        min_eig = eigvals[0]  # smallest eigenvalue of J
        loss = -min_eig
        loss.backward()
        opt.step()
        with torch.no_grad():
            cur_norm = torch.linalg.norm(E_param)
            E_param *= (photon_budget ** 0.5) / cur_norm

        if best_min_eig is None or min_eig.item() > best_min_eig:
            best_min_eig = min_eig.item()
            best_vec = E_param.detach().clone()

        losses_spec.append(loss.item())
        if step % 10 == 0:
            print(f"Spectral opt Step {step:4d} => min eigenvalue(J) = {min_eig.item():.3f}")

    return best_vec.cpu().numpy(), best_min_eig, losses_spec

E_optim_spec, best_min_eig, losses_spec = torch_optimize_spec(
    F_tensor_torch, N_in=N_in, steps=5000, lr=1e-3, photon_budget=10
)
print(f"\nSpectral-optimized wavefront, best minimum eigenvalue(J) = {best_min_eig:.3f}")

E_opt_spec_complex = E_optim_spec[:, 0] + 1j * E_optim_spec[:, 1]

base_sim.reset()
base_sim.set_input_field_kx(kx_array, E_opt_spec_complex)
base_sim.solve_fields()

plt.figure()
plt.plot(losses_spec)
plt.xlabel("Iteration")
plt.ylabel("Loss: -(min eigenvalue of J)")
plt.title("Spectral Radius Optimization Loss")
plt.show()

base_sim.visualize_intensity()

# %% 8) Compare and visualize all final wavefronts

def compute_fisher_matrix_for_wavefront(wavefront, F_tensor):
    import torch

    real_part = np.real(wavefront)
    imag_part = np.imag(wavefront)
    E_np = np.column_stack([real_part, imag_part])
    E_torch = torch.tensor(E_np, dtype=torch.float64)
    E_c = E_torch[:, 0] + 1j * E_torch[:, 1]
    J_mat_torch = torch.zeros((3, 3), dtype=torch.float64)
    for mu in range(3):
        for nu in range(3):
            Fmn = F_tensor[mu, nu, :, :]
            val = (torch.conj(E_c).unsqueeze(0) @ (Fmn @ E_c.unsqueeze(1)))[0, 0]
            J_mat_torch[mu, nu] = val.real
    return J_mat_torch.detach().cpu().numpy()


def visualize_multi_param_fisher_comparison(
    wavefronts_dict, F_tensor_torch, figure_title="Comparison of Wavefronts"
):
    param_labels = ["x", "z", "theta"]
    results = {}
    wave_names = list(wavefronts_dict.keys())

    for wname in wave_names:
        wf_cplx = wavefronts_dict[wname]
        J_mat = compute_fisher_matrix_for_wavefront(wf_cplx, F_tensor_torch)
        diag_vals = [J_mat[0, 0], J_mat[1, 1], J_mat[2, 2]]
        trace_val = np.trace(J_mat)
        det_val = np.linalg.det(J_mat)
        results[wname] = {"diag": diag_vals, "trace": trace_val, "det": det_val}

    n_wf = len(wave_names)
    diag_data = np.array([results[w]["diag"] for w in wave_names])
    trace_data = np.array([results[w]["trace"] for w in wave_names])
    trace_data /= np.max(trace_data)
    det_data = np.array([results[w]["det"] for w in wave_names])
    det_data /= np.max(det_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
    axes[1].set_ylabel("Value [normalized]")

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


photon_budget = 1  # e.g. 10 total "photons"

wavefront_x_norm = normalize_wavefront(wavefront_x, photon_budget)
wavefront_z_norm = normalize_wavefront(wavefront_z, photon_budget)
wavefront_rot_norm = normalize_wavefront(wavefront_rot, photon_budget)
v_sum_max_norm = normalize_wavefront(v_sum_max, photon_budget)
E_opt_complex_norm = normalize_wavefront(E_opt_complex, photon_budget)
E_opt_spec_complex_norm = normalize_wavefront(E_opt_spec_complex, photon_budget)

wavefronts_dict = {
    "MIS_x": wavefront_x_norm,
    "MIS_z": wavefront_z_norm,
    "MIS_rot": wavefront_rot_norm,
    "SumDiag": v_sum_max_norm,
    "DetMax": E_opt_complex_norm,
    "SpecOpt": E_opt_spec_complex_norm,
}
visualize_multi_param_fisher_comparison(
    wavefronts_dict, F_tensor_torch, figure_title="Comparison of all wavefronts"
)


# %% 9a) Compare CRLB (i.e. diagonal of J^-1) for each wavefront

def compute_crlb_for_wavefront(wavefront, F_tensor):
    """
    wavefront: 1D complex array of shape (N_in,)
    F_tensor: shape(3,3,N_in,N_in) in torch.complex128
    Returns a 3x3 real numpy array for J^{-1}, i.e. the covariance bound,
    and we can look at diag(M) = CRLB_x, CRLB_z, CRLB_theta.
    """
    J_fish = compute_fisher_matrix_for_wavefront(wavefront, F_tensor)
    M_inv = np.linalg.inv(J_fish)
    return M_inv


def visualize_crlb_comparison(
    wavefronts_dict, F_tensor_torch, figure_title="CRLB Comparison"
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

    for wname in wave_names:
        wf_cplx = wavefronts_dict[wname]
        M_inv = compute_crlb_for_wavefront(wf_cplx, F_tensor_torch)
        diag_vals = [M_inv[0, 0], M_inv[1, 1], M_inv[2, 2]]
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
    ax.legend()
    ax.grid(True, linestyle=":")
    ax.set_ylabel("CRLB (units of param^2)")
    plt.tight_layout()
    plt.show()


visualize_crlb_comparison(
    wavefronts_dict, F_tensor_torch, figure_title="CRLB (J^-1 diagonal) for each wavefront"
)

# %% 9b) Plot STD (square root of CRLB) for each wavefront

def compute_std_for_wavefront(wavefront, F_tensor):
    """
    wavefront: 1D complex array, shape (N_in,)
    F_tensor: shape(3,3,N_in,N_in) in torch.complex128
    Returns a 3-element array of sqrt((J^{-1})_{xx}), sqrt((J^{-1})_{zz}), sqrt((J^{-1})_{\theta\theta}).
    """
    M_inv = compute_crlb_for_wavefront(wavefront, F_tensor)
    diag_std = np.sqrt(np.diag(M_inv))
    return diag_std

def visualize_std_comparison(wavefronts_dict, F_tensor_torch, figure_title="STD (sqrt(CRLB)) for each wavefront"):
    param_labels = ["STD_x", "STD_z", "STD_theta"]
    wave_names = list(wavefronts_dict.keys())
    data = []

    for wname in wave_names:
        wf_cplx = wavefronts_dict[wname]
        std_vals = compute_std_for_wavefront(wf_cplx, F_tensor_torch)
        data.append(std_vals)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(8,5))

    x_idx = np.arange(len(wave_names))
    width = 0.25
    for i in range(3):
        offset = (i-1)*width
        ax.bar(x_idx+offset, data[:,i], width=width, label=param_labels[i])

    ax.set_title(figure_title)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(wave_names, rotation=25)
    ax.legend()
    ax.grid(True, linestyle=":")
    ax.set_ylabel("Standard Deviation (units of param)")
    plt.tight_layout()
    plt.show()


visualize_std_comparison(
    wavefronts_dict, F_tensor_torch, figure_title="STD (sqrt(CRLB)) for each wavefront"
)