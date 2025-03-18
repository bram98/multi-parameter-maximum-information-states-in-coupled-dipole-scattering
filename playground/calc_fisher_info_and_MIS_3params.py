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
7c) Optimize the input state by minimizing trace(J⁻¹) (multi-run).
8) Compare and visualize all optimal wavefronts.
9) Compute and compare CRLB and STD for each optimal wavefront.
"""

# %% Imports and functions

import matplotlib.pyplot as plt
import numpy as np
import torch
from libraries.ScatteringSimulation import CoupledDipoleSimulator


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


def place_double_line_of_dipoles(number=7):
    """
    Returns an array shape(2, 2*number) with the x,z coords of two lines of dipoles:
    - First line at z=3 (fixed).
    - Second line at z=5 (this is the line that gets shifted/rotated in the partial-derivative steps).
    """
    line1 = place_line_of_dipoles(center=(3.0, 0.0), dist=1.0, number=number)
    line2 = place_line_of_dipoles(center=(5.0, 0.0), dist=1.0, number=number)
    return np.hstack((line1, line2))


def shift_and_rotate_dipoles(original_positions, dx, dz, dtheta):
    """
    original_positions: shape(2,N) with each column a dipole coordinate (z, x).
    dx, dz: translations in x,z
    dtheta: rotation in radians about the second line's centroid if we have 2*N dipoles.
            Otherwise, apply shift/rotate to ALL dipoles (single line, square, etc.).

    Return the new positions shape(2,N).
    """
    out = original_positions.copy()
    N = out.shape[1]

    # If we detect 2 lines (e.g., 14 dipoles for 2 lines of 7 each),
    # only the second half gets shifted/rotated; the first half remains fixed.
    if N == 2 * 7:  # or any 2*N if you prefer to generalize further
        halfN = N // 2
        sub = out[:, halfN:]  # second line only
        meanz = np.mean(sub[0, :])
        meanx = np.mean(sub[1, :])
        temp_z = sub[0, :] - meanz
        temp_x = sub[1, :] - meanx
        cosT = np.cos(dtheta)
        sinT = np.sin(dtheta)
        z_rot = cosT * temp_z - sinT * temp_x
        x_rot = sinT * temp_z + cosT * temp_x
        z_rot += meanz + dz
        x_rot += meanx + dx

        out[0, halfN:] = z_rot
        out[1, halfN:] = x_rot

    else:
        # Single line or square => shift/rotate all dipoles
        meanz = np.mean(out[0, :])
        meanx = np.mean(out[1, :])
        temp_z = out[0, :] - meanz
        temp_x = out[1, :] - meanx
        cosT = np.cos(dtheta)
        sinT = np.sin(dtheta)
        z_rot = cosT * temp_z - sinT * temp_x
        x_rot = sinT * temp_z + cosT * temp_x
        z_rot += meanz + dz
        x_rot += meanx + dx

        out[0, :] = z_rot
        out[1, :] = x_rot

    return out


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


multi_runs = 5  # number of cold restarts for some optimizations

# %% 1) define some dipoles
# base_dipoles = place_square_of_dipoles(center=(4.0, 0.0), side=1.0)
# base_dipoles = place_line_of_dipoles(number=7)
base_dipoles = place_double_line_of_dipoles(number=7)

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

dS = np.array(dS)

# %% 4) Build the full multi-parameter Fisher-operator array F[mu,nu,:,:]
N_in = dS[0].shape[1]
F = np.zeros((3, 3, N_in, N_in), dtype=complex)

for mu in range(3):
    for nu in range(3):
        # F_{mu,nu} = 1/2( dS[mu]^dagger dS[nu] + dS[nu]^dagger dS[mu] )
        tmp1 = dS[mu].conjugate().T @ dS[nu]
        tmp2 = dS[nu].conjugate().T @ dS[mu]
        F[mu, nu, :, :] = 0.5 * (tmp1 + tmp2)

print("Built F[mu,nu,:,:], shape =", F.shape, "(3,3,N_in,N_in)")

# %% 5) Single-parameter MIS wavefronts
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
torch.manual_seed(0)


def fisher_3x3(E_vec, F_tensor):
    E_c = E_vec[:, 0] + 1j * E_vec[:, 1]
    J_mat = torch.zeros((3, 3), dtype=torch.float64, device=E_vec.device)
    for mu in range(3):
        for nu in range(3):
            Fmn = F_tensor[mu, nu, :, :]
            val = (torch.conj(E_c).unsqueeze(0) @ (Fmn @ E_c.unsqueeze(1)))[0, 0]
            J_mat[mu, nu] = val.real
    return J_mat


def torch_optimize_det(F_tensor, N_in, steps=300, lr=1e-3, photon_budget=1):
    E_param = torch.randn((N_in, 2), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        norm_init = torch.linalg.norm(E_param)
        E_param *= photon_budget ** 0.5 / norm_init

    opt = torch.optim.Adam([E_param], lr=lr)
    best_det_val = None
    best_vec = None

    for step in range(steps):
        opt.zero_grad()
        E_norm = photon_budget ** 0.5 * E_param / torch.linalg.norm(E_param)
        J = fisher_3x3(E_norm, F_tensor)
        det_val = torch.linalg.det(J)
        loss = -det_val
        loss.backward()
        opt.step()

        with torch.no_grad():
            cur_norm = torch.linalg.norm(E_norm)
            E_norm *= photon_budget ** 0.5 / cur_norm

        if best_det_val is None or det_val.item() > best_det_val:
            best_det_val = det_val.item()
            best_vec = E_norm.detach().clone()
        losses_det.append(-det_val.item())

        if step % 100 == 0:
            print(f"Det opt Step {step:4d} => det={det_val.item():.3f}")

    return best_vec.cpu().numpy(), best_det_val


F_tensor_torch = torch.from_numpy(F).to(torch.complex128)
E_optim_det, best_det = torch_optimize_det(
    F_tensor_torch, N_in=N_in, steps=1000, lr=1e-3, photon_budget=1
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
    of the covariance matrix J^-1.
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
        E_norm = photon_budget ** 0.5 * E_param / torch.linalg.norm(E_param)
        J = fisher_3x3(E_norm, F_tensor)
        eigvals = torch.linalg.eigvalsh(J)
        min_eig = eigvals[0]  # smallest eigenvalue of J
        loss = -min_eig
        loss.backward()
        opt.step()

        with torch.no_grad():
            cur_norm = torch.linalg.norm(E_norm)
            E_norm *= (photon_budget ** 0.5) / cur_norm

        if best_min_eig is None or min_eig.item() > best_min_eig:
            best_min_eig = min_eig.item()
            best_vec = E_norm.detach().clone()

        losses_spec.append(loss.item())
        if step % 100 == 0:
            print(
                f"Spectral opt Step {step:4d} => min eigenvalue(J) = {min_eig.item():.3f}"
            )

    return best_vec.cpu().numpy(), best_min_eig, losses_spec


E_optim_spec, best_min_eig, losses_spec = torch_optimize_spec(
    F_tensor_torch, N_in=N_in, steps=1000, lr=1e-3, photon_budget=1
)
print(
    f"\nSpectral-optimized wavefront, best minimum eigenvalue(J) = {best_min_eig:.3f}"
)

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

# %% 7c) Minimizing Trace(J^-1) with multi-run
def torch_optimize_traceinv(
    F_tensor, N_in, steps=3000, lr=1e-3, photon_budget=1, runs=10
):
    """
    Minimizes trace(J^-1). We'll do multiple restarts, pick the best.
    Returns the best wavefront, best trace, etc.
    """
    best_loss_overall = None
    best_vec = None
    best_loss_curve = None

    all_loss_curves = []

    for run_i in range(runs):
        E_param = torch.randn((N_in, 2), dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            norm_init = torch.linalg.norm(E_param)
            E_param *= photon_budget ** 0.5 / norm_init

        opt = torch.optim.Adam([E_param], lr=lr)
        run_loss_curve = []
        best_run_loss = None
        best_run_vec = None

        for step in range(steps):
            opt.zero_grad()
            E_norm = photon_budget ** 0.5 * E_param / torch.linalg.norm(E_param)
            J = fisher_3x3(E_norm, F_tensor)
            J_inv = torch.linalg.inv(J)
            trace_val = torch.trace(J_inv)
            loss = trace_val  # we want to minimize trace(J^-1)

            loss.backward()
            opt.step()

            with torch.no_grad():
                cur_norm = torch.linalg.norm(E_norm)
                E_norm *= photon_budget ** 0.5 / cur_norm

            cur_loss = loss.item()
            run_loss_curve.append(cur_loss)

            if (best_run_loss is None) or (cur_loss < best_run_loss):
                best_run_loss = cur_loss
                best_run_vec = E_norm.detach().clone()

        all_loss_curves.append(run_loss_curve)
        print(f"Restart {run_i+1}/{runs}: loss = {cur_loss:.5f}")

        # Compare across runs
        if (best_loss_overall is None) or (best_run_loss < best_loss_overall):
            best_loss_overall = best_run_loss
            best_vec = best_run_vec
            best_loss_curve = run_loss_curve

    # Now build final wavefront from best run
    E_opt_cplx = best_vec[:, 0].cpu().numpy() + 1j * best_vec[:, 1].cpu().numpy()
    J_np = fisher_3x3(torch.tensor(best_vec), F_tensor).detach().cpu().numpy()
    J_inv_np = np.linalg.inv(J_np)
    final_trace = np.trace(J_inv_np)

    return E_opt_cplx, final_trace, best_loss_curve, all_loss_curves


print("\n--- Minimizing Trace(J^-1) with multi-run ---")
(
    E_traceinv,
    best_traceinv_val,
    best_traceinv_curve,
    all_traceinv_loss_curves,
) = torch_optimize_traceinv(
    F_tensor_torch, N_in=N_in, steps=4000, lr=1e-3, photon_budget=1, runs=multi_runs
)
print(f"Best final trace(J^-1) after multi-run: {best_traceinv_val:.5f}")

# Visualize the multi-run loss curves in symlog
plt.figure(figsize=(7, 5))
colors = plt.cm.plasma(np.linspace(0, 1, len(all_traceinv_loss_curves)))
for i, lc in enumerate(all_traceinv_loss_curves):
    plt.plot(lc, color=colors[i], alpha=0.5)
plt.yscale("symlog", linthresh=1e-3, linscale=1)
plt.xlabel("Iteration")
plt.ylabel("Trace(J^-1) [unnormalized]")
plt.title("All Runs for Minimizing Trace(J^-1)")
plt.tight_layout()
plt.show()

# Visualize best wavefront
base_sim.reset()
base_sim.set_input_field_kx(kx_array, E_traceinv)
base_sim.solve_fields()
base_sim.visualize_intensity()

# %% 8) Compare and visualize wavefronts
def compute_fisher_matrix_for_wavefront(wavefront, F_tensor):
    E_c = torch.tensor(wavefront, dtype=torch.complex128)
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
    param_labels = ["x", "z", r"\theta"]
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
    det_data = np.array([results[w]["det"] for w in wave_names])

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
        x_idx + offset_2 / 2, det_data, width=offset_2, color=det_color, label="Det(J)",
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


def normalize_wavefront(wavefront_cplx, photon_budget):
    current_norm_sq = np.sum(np.abs(wavefront_cplx) ** 2)
    if current_norm_sq <= 0:
        return wavefront_cplx
    scale_factor = (photon_budget / current_norm_sq) ** 0.5
    return wavefront_cplx * scale_factor


photon_budget = 1
wavefront_x_norm = normalize_wavefront(wavefront_x, photon_budget)
wavefront_z_norm = normalize_wavefront(wavefront_z, photon_budget)
wavefront_rot_norm = normalize_wavefront(wavefront_rot, photon_budget)
v_sum_max_norm = normalize_wavefront(v_sum_max, photon_budget)
E_opt_complex_norm = normalize_wavefront(E_opt_complex, photon_budget)
E_opt_spec_complex_norm = normalize_wavefront(E_opt_spec_complex, photon_budget)
E_traceinv_norm = normalize_wavefront(E_traceinv, photon_budget)

wavefronts_dict = {
    r"Max $\mathcal{J}_x$": wavefront_x_norm,
    r"Max $\mathcal{J}_z$": wavefront_z_norm,
    r"Max $\mathcal{J}_\theta$": wavefront_rot_norm,
    r"Max Trace($\mathcal{J}$)": v_sum_max_norm,
    r"Max Det($\mathcal{J}$)": E_opt_complex_norm,
    "Min Spectral Radius ($\mathcal{J}^{-1}$)": E_opt_spec_complex_norm,
    r"Min Trace($\mathcal{J}^{-1}$)": E_traceinv_norm,
}


visualize_multi_param_fisher_comparison(
    wavefronts_dict, F_tensor_torch, figure_title="Comparison of all wavefronts"
)

# %% 9) Compare CRLB, with optional plane wave or random wavefront comparison


def compute_crlb_matrix_for_wavefront(wavefront, F_tensor):
    J_fish = compute_fisher_matrix_for_wavefront(wavefront, F_tensor)
    M_inv = np.linalg.inv(J_fish)
    return M_inv


def compute_crlb_for_wavefront(wavefront, F_tensor):
    M_inv = compute_crlb_matrix_for_wavefront(wavefront, F_tensor)
    return np.sqrt(np.diag(M_inv))


def compute_crlb_for_planewaves(
    sim_kwargs, base_dipoles, angles_deg, F_tensor, photon_budget, kx_array
):
    from libraries.ScatteringSimulation import _flux_amplitude_2d

    all_crlbs = []
    k0 = 2 * np.pi / sim_kwargs["lambda_0"]
    N_in = len(kx_array)
    for angle in angles_deg:
        theta_rad = np.deg2rad(angle)
        kx = k0 * np.sin(theta_rad)
        amplitude = _flux_amplitude_2d(k0, kx) * np.sqrt(photon_budget)
        idx = np.argmin(np.abs(kx_array - kx))
        E_plane_wave = np.zeros(N_in, dtype=complex)
        E_plane_wave[idx] = amplitude
        crlb_vals = compute_crlb_for_wavefront(E_plane_wave, F_tensor)
        all_crlbs.append(crlb_vals)
    return np.array(all_crlbs)


def compute_crlb_for_random_wavefronts(F_tensor, photon_budget, num_samples, N_in):
    all_crlbs = []
    for _ in range(num_samples):
        E_random = np.random.randn(N_in) + 1j * np.random.randn(N_in)
        E_random_normalized = normalize_wavefront(E_random, photon_budget)
        crlb_vals = compute_crlb_for_wavefront(E_random_normalized, F_tensor)
        all_crlbs.append(crlb_vals)
    return np.array(all_crlbs)


def visualize_crlb_comparison(
    wavefronts_dict,
    F_tensor_torch,
    sim_kwargs,
    base_dipoles,
    angles_deg,
    photon_budget,
    kx_array,
    figure_title="CRLB Comparison",
    show_plane_waves=True,
    show_random_wavefronts=True,
    num_random_samples=100,
):
    param_labels = ["CRLB x", "CRLB z", "CRLB θ"]
    wave_names = list(wavefronts_dict.keys())
    data = []
    for wname in wave_names:
        wf_cplx = wavefronts_dict[wname]
        wf_cplx_normalized = normalize_wavefront(wf_cplx, photon_budget)
        crlb_vals = compute_crlb_for_wavefront(wf_cplx_normalized, F_tensor_torch)
        data.append(crlb_vals)

    if show_plane_waves:
        plane_wave_crlbs = compute_crlb_for_planewaves(
            sim_kwargs,
            base_dipoles,
            angles_deg,
            F_tensor_torch,
            photon_budget,
            kx_array,
        )
        plane_wave_data = [plane_wave_crlbs[:, i] for i in range(3)]

    if show_random_wavefronts:
        random_wavefront_crlbs = compute_crlb_for_random_wavefronts(
            F_tensor_torch, photon_budget, num_random_samples, len(kx_array)
        )
        random_wavefront_data = [random_wavefront_crlbs[:, i] for i in range(3)]

    data = np.array(data)
    n_custom_wavefronts = len(wave_names)
    num_boxplot_groups = 0
    if show_plane_waves:
        num_boxplot_groups += 1
    if show_random_wavefronts:
        num_boxplot_groups += 1

    fig, ax = plt.subplots(figsize=(7 + 2 * num_boxplot_groups, 6))
    x_idx_custom = np.arange(n_custom_wavefronts)
    width = 0.2
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bar_sep_small = width * 0.7
    bar_sep_large = width * 1.5

    for i in range(3):
        offset = (i - 1) * width
        ax.bar(
            x_idx_custom + offset,
            data[:, i],
            width=width,
            label=param_labels[i],
            color=colors[i],
        )

    x_idx_special = n_custom_wavefronts + bar_sep_small

    if show_plane_waves:
        box_positions_plane = [
            x_idx_special - width,
            x_idx_special,
            x_idx_special + width,
        ]
        bp_plane = ax.boxplot(
            plane_wave_data,
            positions=box_positions_plane,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black"},
        )
        for patch, color in zip(bp_plane["boxes"], colors):
            patch.set_facecolor(color)
        if show_random_wavefronts:
            x_idx_special += 3 * width + bar_sep_large

    if show_random_wavefronts:
        box_positions_random = [
            x_idx_special - width,
            x_idx_special,
            x_idx_special + width,
        ]
        bp_random = ax.boxplot(
            random_wavefront_data,
            positions=box_positions_random,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black"},
        )
        for patch, color in zip(bp_random["boxes"], colors):
            patch.set_facecolor(color)

    all_labels = wave_names[:]
    if show_plane_waves:
        all_labels += ["Plane Waves"]
    if show_random_wavefronts:
        all_labels += ["Random WF"]

    all_x_ticks = x_idx_custom.tolist()
    if show_plane_waves:
        x_tick_pw = n_custom_wavefronts + bar_sep_small
        all_x_ticks.append(x_tick_pw)
    if show_random_wavefronts:
        x_tick_rw = x_idx_special
        all_x_ticks.append(x_tick_rw)

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(14)

    ax.set_xticks(all_x_ticks)
    ax.set_xticklabels(all_labels, rotation=20, ha="center")
    ax.set_title(figure_title)
    ax.grid(True, linestyle=":")
    ax.set_ylabel("Estimation Error (units of param)")

    if show_plane_waves or show_random_wavefronts:
        ax.set_yscale("log")

    ax.legend()
    plt.tight_layout()
    plt.show()


visualize_crlb_comparison(
    wavefronts_dict,
    F_tensor_torch,
    sim_kwargs,
    base_dipoles,
    angles_deg,
    photon_budget,
    kx_array,
    show_plane_waves=True,
    show_random_wavefronts=True,
)

# %% 10) Fisher and Covariance Matrix Visualization


def visualize_J_and_inverse(
    wavefronts_dict, F_tensor, wavefronts_to_show=None, title="Comparison of J and J^-1"
):
    """
    Plots J and J^-1 for each wavefront, plus a bar chart of condition numbers.
    This version fixes axis labeling, sets each 3x3 plot to square aspect,
    and clarifies color scales.
    """
    if wavefronts_to_show is None:
        wavefronts_to_show = list(wavefronts_dict.keys())

    # We'll compute J, J^-1, and condition number for each wavefront
    Js = {}
    J_invs = {}
    cond_nums = {}

    for wf_name in wavefronts_to_show:
        wf_cplx = wavefronts_dict[wf_name]
        J_mat = compute_fisher_matrix_for_wavefront(wf_cplx, F_tensor)
        inv_mat = np.linalg.inv(J_mat)
        Js[wf_name] = J_mat
        J_invs[wf_name] = inv_mat
        cond_nums[wf_name] = np.linalg.cond(J_mat)

    n_wave = len(wavefronts_to_show)
    fig = plt.figure(figsize=(12, 3 * n_wave))
    gs = fig.add_gridspec(n_wave, 2, wspace=0.2)

    label_list = ["x", "z", "θ"]  # Axis labels for the J matrix

    for i, wf_name in enumerate(wavefronts_to_show):
        # We'll subdivide each row into two subplots for J and J^-1
        row_gs = gs[i].subgridspec(1, 2, wspace=0.4)
        axJ = fig.add_subplot(row_gs[0, 0])
        axInv = fig.add_subplot(row_gs[0, 1])

        J_mat = Js[wf_name]
        inv_mat = J_invs[wf_name]

        # Plot J
        imJ = axJ.imshow(J_mat, cmap="viridis", aspect="equal")
        axJ.set_xticks(range(3))
        axJ.set_xticklabels(label_list, fontsize=10)
        axJ.set_yticks(range(3))
        axJ.set_yticklabels(label_list, fontsize=10)
        axJ.set_title(f"{wf_name}\n J-matrix", fontsize=11)
        cbJ = plt.colorbar(imJ, ax=axJ, fraction=0.046, pad=0.04)
        cbJ.formatter.set_powerlimits((-2, 2))

        # Plot J^-1
        imInv = axInv.imshow(inv_mat, cmap="magma", aspect="equal")
        axInv.set_xticks(range(3))
        axInv.set_xticklabels(label_list, fontsize=10)
        axInv.set_yticks(range(3))
        axInv.set_yticklabels(label_list, fontsize=10)
        axInv.set_title(f"{wf_name}\n $J^{{-1}}$-matrix", fontsize=11)
        cbInv = plt.colorbar(imInv, ax=axInv, fraction=0.046, pad=0.04)
        cbInv.formatter.set_powerlimits((-2, 2))

    # Condition-number bar chart in a separate figure
    fig_cond = plt.figure(figsize=(6, 4))
    ax_cond = fig_cond.add_subplot(111)
    wf_list = list(wavefronts_to_show)
    cond_vals = [cond_nums[w] for w in wf_list]
    x_indices = np.arange(len(wf_list))
    bars = ax_cond.bar(x_indices, cond_vals, color="skyblue", alpha=0.9)
    ax_cond.set_xticks(x_indices)
    ax_cond.set_xticklabels(wf_list, rotation=25, ha="right", fontsize=9)
    ax_cond.set_ylabel("Condition Number of $\mathcal{J}$")
    ax_cond.set_title("Condition Number Comparison")
    for bar, cval in zip(bars, cond_vals):
        ax_cond.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{cval:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(title, y=0.92)
    # fig.tight_layout()
    # fig_cond.tight_layout()
    plt.show()


# wavefronts_subset = [
#      r"Max $\mathcal{J}_x$",
#      r"Max Det($\mathcal{J}$)",
#      "Min Spectral Radius ($\mathcal{J}^{-1}$)"
# ]
visualize_J_and_inverse(
    wavefronts_dict,
    F_tensor_torch,
    title="Comparison of J and J^-1 for Different Wavefronts",
)

# %% 11) Multi-objective optimization and Pareto Frontier Plot
# For the "OffDiag" approach, we run multiple restarts and then plot all the loss curves.


def nuisance_opt_run(
    F_tensor, N_in, steps, lr, photon_budget, approach_name, alpha=0.1, beta=1.0
):
    """
    Runs the nuisance optimization with a given approach_name: "LogLin", "OffDiag", or "SubDet".
    Returns:
      - E_opt_cplx: final wavefront (complex),
      - best_loss: best scalar loss found,
      - CRLB_diag: sqrt of diag of J^-1 at the end,
      - loss_curve: list with the entire loss history.
    """
    E_param = torch.randn((N_in, 2), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        norm_init = torch.linalg.norm(E_param)
        E_param *= (photon_budget ** 0.5) / norm_init

    opt = torch.optim.Adam([E_param], lr=lr)
    best_loss = None
    best_vec = None
    loss_curve = []

    for _ in range(steps):
        opt.zero_grad()
        # Re-normalize wavefront each iteration
        E_norm = photon_budget ** 0.5 * E_param / torch.linalg.norm(E_param)
        J = fisher_3x3(E_norm, F_tensor)
        J_inv = torch.linalg.inv(J)

        if approach_name == "LogLin":
            loss = (
                torch.log(J_inv[0, 0])
                + torch.log(J_inv[1, 1])
                - alpha * torch.log(J_inv[2, 2])
            )
        elif approach_name == "OffDiag":
            off_diag = J_inv[0, 2] ** 2 + J_inv[1, 2] ** 2 + J_inv[0, 1] ** 2
            off_diag += J_inv[2, 0] ** 2 + J_inv[2, 1] ** 2 + J_inv[1, 0] ** 2
            loss = (J_inv[0, 0] + J_inv[1, 1]) + beta * off_diag
        elif approach_name == "SubDet":
            J_sub = J[0:2, 0:2]
            loss = -torch.log(torch.det(J_sub))
        else:
            raise ValueError(f"Unknown approach_name {approach_name}")

        loss.backward()
        opt.step()

        if _ % 1000 == 0:
            print(f"Iteration {_}: loss = {loss.item():.5f}")

        with torch.no_grad():
            cur_norm = torch.linalg.norm(E_param)
            E_param *= (photon_budget ** 0.5) / cur_norm

        cur_loss = loss.item()
        loss_curve.append(cur_loss)
        if (best_loss is None) or (cur_loss < best_loss):
            best_loss = cur_loss
            best_vec = E_norm.detach().clone()

    E_opt_cplx = (best_vec[:, 0] + 1j * best_vec[:, 1]).cpu().numpy()
    J_np = compute_fisher_matrix_for_wavefront(E_opt_cplx, F_tensor)
    CRLB_diag = np.sqrt(np.diag(np.linalg.inv(J_np)))
    return E_opt_cplx, best_loss, CRLB_diag, loss_curve


# Approaches to compare
approaches = ["LogLin", "OffDiag", "SubDet"]
results = {}
loss_histories = {}

print("=== Multi-Approach Nuisance Optimization + Extended Analysis ===")

for ap in approaches:
    print(f"\n--- Approach: {ap} ---")
    # if True:
    if ap == "OffDiag":
        # Run with multi-restarts
        num_restarts = multi_runs
        all_loss_curves = []
        best_loss_overall = None
        best_wf_overall = None
        best_crlb = None
        best_loss_curve = None
        for i in range(num_restarts):
            wf_temp, loss_temp, crlb_temp, loss_curve_temp = nuisance_opt_run(
                F_tensor_torch,
                N_in=N_in,
                steps=4000,
                lr=1e-3,
                photon_budget=1,
                approach_name=ap,
                alpha=0.1,
                beta=1.0,
            )
            all_loss_curves.append(loss_curve_temp)
            print(f"Restart {i+1}/{num_restarts}: loss = {loss_temp:.5f}")
            if best_loss_overall is None or loss_temp < best_loss_overall:
                best_loss_overall = loss_temp
                best_wf_overall = wf_temp
                best_crlb = crlb_temp
                best_loss_curve = loss_curve_temp
        # Plot all OffDiag loss curves with a color gradient
        plt.figure(figsize=(7, 5))
        colors = plt.cm.winter(np.linspace(0, 1, num_restarts))
        for i in range(num_restarts):
            plt.plot(all_loss_curves[i], color=colors[i], alpha=0.5)
            plt.yscale("symlog", linthresh=1e-3, linscale=1)
        plt.xlabel("Iteration")
        plt.ylabel(f"Loss ({ap}, unnormalized)")
        plt.title(f"{ap} Loss Curves (All Restarts)")
        plt.tight_layout()
        plt.show()
        wf_cplx, best_loss, crlb_vec, loss_curve = (
            best_wf_overall,
            best_loss_overall,
            best_crlb,
            best_loss_curve,
        )
    else:
        wf_cplx, best_loss, crlb_vec, loss_curve = nuisance_opt_run(
            F_tensor_torch,
            N_in=N_in,
            steps=2000,
            lr=1e-3,
            photon_budget=1,
            approach_name=ap,
            alpha=0.1,
            beta=1.0,
        )
    name_key = f"{ap}"
    wavefronts_dict[name_key] = normalize_wavefront(wf_cplx, 1.0)
    results[ap] = {"wavefront": wf_cplx, "loss": best_loss, "CRLB": crlb_vec}
    loss_histories[ap] = loss_curve
    print(f"  -> best loss = {best_loss:.5f}")
    print(
        f"  -> final CRLB = [x={crlb_vec[0]:.2e}, z={crlb_vec[1]:.2e}, θ={crlb_vec[2]:.2e}]"
    )


# 2) Visualize each approach's wavefront using base_sim.visualize_intensity()
for ap in approaches:
    wf_name = f"{ap}"
    print(f"\nVisualizing wavefront: {wf_name}")
    base_sim.reset()
    base_sim.set_input_field_kx(kx_array, wavefronts_dict[wf_name])
    base_sim.solve_fields()
    base_sim.visualize_intensity()


# 3) Condition Number bar plot (include MaxDet if available)
wf_maxdet = wavefronts_dict.get("Max Det($\\mathcal{J}$)")
cond_dict = {}
if wf_maxdet is not None:
    J_mdet = compute_fisher_matrix_for_wavefront(wf_maxdet, F_tensor_torch)
    cond_dict["MaxDetJ"] = np.linalg.cond(J_mdet)
for ap in approaches:
    wf_name = f"{ap}"
    J_ = compute_fisher_matrix_for_wavefront(wavefronts_dict[wf_name], F_tensor_torch)
    cond_dict[ap] = np.linalg.cond(J_)
plt.figure(figsize=(6, 4))
names_c = list(cond_dict.keys())
vals_c = [cond_dict[k] for k in names_c]
plt.bar(range(len(names_c)), vals_c, color="skyblue")
plt.xticks(range(len(names_c)), names_c, rotation=25)
plt.ylabel("Condition Number of J")
plt.title("Fisher Matrix Condition Number")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.show()


# 4) CRLB comparisons with all wavefronts
print("\nVisualizing multi-param fisher, CRLB, and J-and-inverse for all wavefronts.")
visualize_multi_param_fisher_comparison(
    wavefronts_dict, F_tensor_torch, figure_title="All Wavefronts: Multi-Param Fisher"
)
visualize_crlb_comparison(
    wavefronts_dict,
    F_tensor_torch,
    sim_kwargs,
    base_dipoles,
    angles_deg,
    photon_budget=1,
    kx_array=kx_array,
    show_plane_waves=True,
    show_random_wavefronts=True,
)
visualize_J_and_inverse(
    wavefronts_dict,
    F_tensor_torch,
    title="Comparison of J and J^-1 for All Wavefronts",
)

# 5) Bar plot of CRLB just for the new approaches (plus MaxDet if available)
labels = ["x", "z", "θ"]
all_keys = list(approaches)
if E_traceinv_norm is not None:
    all_keys.insert(0, r"MinTrace(J^-1)")
if wf_maxdet is not None:
    all_keys.insert(0, "MaxDetJ")
crlb_array = []
for k in all_keys:
    if k == "MaxDetJ":
        J_mdet = compute_fisher_matrix_for_wavefront(wf_maxdet, F_tensor_torch)
        crlb_array.append(np.sqrt(np.diag(np.linalg.inv(J_mdet))))
    elif k == r"MinTrace(J^-1)":
        J_mdet = compute_fisher_matrix_for_wavefront(E_traceinv_norm, F_tensor_torch)
        crlb_array.append(np.sqrt(np.diag(np.linalg.inv(J_mdet))))
    else:
        crlb_array.append(results[k]["CRLB"])
crlb_array = np.array(crlb_array)
plt.figure(figsize=(6, 4))
x_inds = np.arange(len(all_keys))
width = 0.25
for i in range(3):
    plt.bar(
        x_inds + i * width - width, crlb_array[:, i], width, label=f"CRLB_{labels[i]}"
    )
plt.xticks(x_inds, all_keys, rotation=45)
plt.ylabel("CRLB (std dev)")
plt.title("CRLB: Comparison of New Approaches + MaxDet")
plt.legend()
plt.tight_layout()
plt.show()

# 6) Plot the three loss curves in one figure, normalized to [0..1]
plt.figure(figsize=(7, 5))
for ap in approaches:
    raw_curve = loss_histories[ap]
    mn = min(raw_curve)
    mx = max(raw_curve)
    rng = mx - mn if mx > mn else 1e-12
    norm_curve = [(val - mn) / rng for val in raw_curve]
    plt.plot(norm_curve, label=f"{ap} (normed)")
plt.xlabel("Iteration")
plt.ylabel("Normalized Loss")
plt.title("Optimization Loss Curves (Normalized)")
plt.legend()
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.show()

# 7) Pareto Frontier Plot Including All Wavefronts
nuisance_keys = ["LogLin", "OffDiag", "SubDet"]
all_keys = list(wavefronts_dict.keys())
crlb_dict = {}
for key in all_keys:
    wf = wavefronts_dict[key]
    J_mat = compute_fisher_matrix_for_wavefront(wf, F_tensor_torch)
    crlb = np.sqrt(np.diag(np.linalg.inv(J_mat)))
    crlb_dict[key] = crlb
crlb_array = np.array([crlb_dict[key] for key in all_keys])
markers = []
for key in all_keys:
    markers.append("o" if key in nuisance_keys else "s")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x1 = crlb_array[:, 0] + crlb_array[:, 1]
y1 = crlb_array[:, 2]
ax1 = axes[0]
for i, key in enumerate(all_keys):
    ax1.scatter(
        x1[i],
        y1[i],
        s=80,
        marker=markers[i],
        color="blue" if key not in nuisance_keys else "red",
    )
    ax1.annotate(
        key,
        (x1[i], y1[i]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.5, alpha=0.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.3),
    )
ax1.set_xlabel("CRLB_x + CRLB_z")
ax1.set_ylabel("CRLB_θ")
ax1.set_title("Sum: (CLRB_x + CRLB_z) vs. CRLB_θ")
ax1.grid(True)
x3 = crlb_array[:, 0]
y3 = crlb_array[:, 1]
ax3 = axes[1]
for i, key in enumerate(all_keys):
    ax3.scatter(
        x3[i],
        y3[i],
        s=80,
        marker=markers[i],
        color="orange" if key not in nuisance_keys else "purple",
    )
    ax3.annotate(
        key,
        (x3[i], y3[i]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.5, alpha=0.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.3),
    )
ax3.set_xlabel("CLRB_x")
ax3.set_ylabel("CRLB_z")
ax3.set_title("CLRB_x vs. CRLB_z")
ax3.grid(True)
plt.tight_layout()
plt.show()


# %% 12) MIS vs Direct CRLB vs Direct J_xx Optimization for x

print("\n--- MIS vs Direct CRLB vs Direct J_xx Optimization for x ---")

# Photon budget (consistent with previous sections)
photon_budget = 1

# MIS wavefront for x (assumed from section 5), normalized
wavefront_x_norm = normalize_wavefront(wavefront_x, photon_budget)

# --- Optimization Functions ---

# 1. Minimize (J^-1)_{xx} (CRLB for x)
def torch_optimize_crlb_xx(
    F_tensor, N_in, steps=4000, lr=1e-3, photon_budget=1, num_runs=10
):
    best_loss = float("inf")
    best_vec = None
    all_loss_curves = []
    for run in range(num_runs):
        E_param = torch.randn((N_in, 2), dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            norm_init = torch.linalg.norm(E_param)
            E_param *= (photon_budget ** 0.5) / norm_init
        opt = torch.optim.Adam([E_param], lr=lr)
        loss_curve = []
        for step in range(steps):
            opt.zero_grad()
            E_norm = photon_budget ** 0.5 * E_param / torch.linalg.norm(E_param)
            J = fisher_3x3(
                E_norm, F_tensor
            )  # Assumed function to compute 3x3 Fisher matrix
            J_inv = torch.linalg.inv(J)
            loss = J_inv[0, 0]  # CRLB for x
            loss.backward()
            opt.step()
            with torch.no_grad():
                E_norm *= (photon_budget ** 0.5) / torch.linalg.norm(E_norm)
            loss_curve.append(loss.item())
            if step % 500 == 0:
                print(
                    f"CRLB Run {run+1}/{num_runs}, Step {step}: CRLB_x = {loss.item():.5f}"
                )
        all_loss_curves.append(loss_curve)
        if loss_curve[-1] < best_loss:
            best_loss = loss_curve[-1]
            best_vec = E_norm.detach().clone()
    E_opt_cplx = best_vec[:, 0].cpu().numpy() + 1j * best_vec[:, 1].cpu().numpy()
    return E_opt_cplx, best_loss, all_loss_curves


# 2. Maximize J_xx directly
def torch_optimize_J_xx(
    F_tensor, N_in, steps=4000, lr=1e-3, photon_budget=1, num_runs=10
):
    best_J_xx = -float("inf")
    best_vec = None
    all_loss_curves = []
    for run in range(num_runs):
        E_param = torch.randn((N_in, 2), dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            norm_init = torch.linalg.norm(E_param)
            E_param *= (photon_budget ** 0.5) / norm_init
        opt = torch.optim.Adam([E_param], lr=lr)
        loss_curve = []
        for step in range(steps):
            opt.zero_grad()
            E_norm = photon_budget ** 0.5 * E_param / torch.linalg.norm(E_param)
            J = fisher_3x3(E_norm, F_tensor)
            J_xx = J[0, 0]  # Fisher information for x
            loss = -J_xx  # Minimize -J_xx to maximize J_xx
            loss.backward()
            opt.step()
            with torch.no_grad():
                E_norm *= (photon_budget ** 0.5) / torch.linalg.norm(E_norm)
            loss_curve.append(J_xx.item())  # Store J_xx value
            if step % 500 == 0:
                print(
                    f"J_xx Run {run+1}/{num_runs}, Step {step}: J_xx = {J_xx.item():.5f}"
                )
        all_loss_curves.append(loss_curve)
        if loss_curve[-1] > best_J_xx:
            best_J_xx = loss_curve[-1]
            best_vec = E_norm.detach().clone()
    E_opt_cplx = best_vec[:, 0].cpu().numpy() + 1j * best_vec[:, 1].cpu().numpy()
    return E_opt_cplx, best_J_xx, all_loss_curves


print("\nOptimizing for CRLB_x...")
E_opt_crlb_xx, best_crlb_xx, all_loss_curves_crlb = torch_optimize_crlb_xx(
    F_tensor_torch,
    N_in=N_in,
    steps=4000,
    lr=1e-3,
    photon_budget=photon_budget,
    num_runs=multi_runs,
)
E_opt_crlb_xx_norm = normalize_wavefront(E_opt_crlb_xx, photon_budget)
print(f"\nBest CRLB_x from optimization: {best_crlb_xx:.3e}")

print("\nOptimizing for J_xx...")
E_opt_J_xx, best_J_xx, all_loss_curves_J_xx = torch_optimize_J_xx(
    F_tensor_torch,
    N_in=N_in,
    steps=4000,
    lr=1e-3,
    photon_budget=photon_budget,
    num_runs=multi_runs,
)
E_opt_J_xx_norm = normalize_wavefront(E_opt_J_xx, photon_budget)
print(f"\nBest J_xx from optimization: {best_J_xx:.3e}")

# Plot loss curves for both optimizations
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i, loss_curve in enumerate(all_loss_curves_crlb):
    plt.plot(loss_curve, label=f"Run {i+1}", alpha=0.7)
plt.xlabel("Iteration")
plt.ylabel("CRLB_x")
plt.title("CRLB_x Optimization Loss Curves")
plt.yscale("log")
plt.grid(True, linestyle=":", alpha=0.5)

plt.subplot(1, 2, 2)
for i, loss_curve in enumerate(all_loss_curves_J_xx):
    plt.plot(loss_curve, label=f"Run {i+1}", alpha=0.7)
plt.xlabel("Iteration")
plt.ylabel("J_xx")
plt.title("J_xx Optimization Loss Curves")
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()

# Visualize all three wavefronts
wavefronts = {
    "MIS_x": wavefront_x_norm,
    "Optimized CRLB_x": E_opt_crlb_xx_norm,
    "Optimized J_xx": E_opt_J_xx_norm,
}

for name, wf in wavefronts.items():
    print(f"\nVisualizing {name} wavefront:")
    base_sim.reset()
    base_sim.set_input_field_kx(kx_array, wf)
    base_sim.solve_fields()
    base_sim.visualize_intensity()

# Compute CRLB for all three wavefronts
crlb_mis = compute_crlb_for_wavefront(
    wavefront_x_norm, F_tensor_torch
)  # Assumed function
crlb_opt_crlb = compute_crlb_for_wavefront(E_opt_crlb_xx_norm, F_tensor_torch)
crlb_opt_J_xx = compute_crlb_for_wavefront(E_opt_J_xx_norm, F_tensor_torch)

# Prepare data for bar plot
crlb_values = np.array([crlb_mis, crlb_opt_crlb, crlb_opt_J_xx])
wavefront_names = ["MIS_x", "Optimized CRLB_xx", "Optimized J_xx"]
param_labels = ["x", "z", "θ"]

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))
x_idx = np.arange(len(wavefront_names))
width = 0.2
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

for i in range(3):  # For each parameter (x, z, theta)
    ax.bar(
        x_idx + i * width,
        crlb_values[:, i],
        width,
        label=param_labels[i],
        color=colors[i],
    )

ax.set_xticks(x_idx + width)
ax.set_xticklabels(wavefront_names)
ax.set_ylabel("CRLB")
ax.set_title("CRLB Comparison: MIS_x vs. Direct CRLB_xx vs. Direct J_xx Optimization")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()

# Print CRLB values
for name, crlb in zip(wavefront_names, [crlb_mis, crlb_opt_crlb, crlb_opt_J_xx]):
    print(f"\nCRLB Values for {name} Wavefront:")
    for i, param in enumerate(param_labels):
        print(f"  {param}: {crlb[i]:.3e}")
