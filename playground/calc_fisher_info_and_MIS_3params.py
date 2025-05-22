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
10) Visualize the 3x3 Fisher Information Matrix (J) and its inverse (J⁻¹) for each wavefront, along with condition numbers.
11) Multi-objective optimization considering nuisance parameters using different strategies ("LogLin", "OffDiag", "SubDet").
    Includes result analysis, wavefront visualization, condition number comparison, and CRLB comparison.
12) Compare single-parameter MIS for 'x' against direct optimization of CRLB_x and direct optimization of J_xx.
    Includes visualization of optimized wavefronts and CRLB values.
13) Optimize input state by maximizing det(J_interest|nuisance_rotation) (multi-run), where rotation is the nuisance parameter
    and (x, z) are parameters of interest. Visualize loss curves and the resulting wavefront.
14) Optimize input state by minimizing Tr((J_interest|nuisance_rotation)⁻¹) (multi-run).
    Visualize loss curves and the resulting wavefront.
15) Optimize input state by maximizing the smallest eigenvalue of J_interest|nuisance_rotation (multi-run).
    Visualize loss curves and the resulting wavefront.
16) Update and re-visualize comprehensive comparisons: multi-parameter Fisher metrics, CRLBs,
    and J and J⁻¹ matrices, now including the nuisance-aware optimized wavefronts from sections 13-15.
17) Generate an updated Pareto frontier plot (CRLB_x+CRLB_z vs CRLB_theta and CRLB_x vs CRLB_z)
    including all previously computed and nuisance-aware wavefronts, using different markers/colors for categories.
18) Generate a detailed Pareto frontier plot with individual point labels for all wavefronts,
    facilitating direct identification of each wavefront's performance in the CRLB space.
"""

# %% Imports and functions

import math

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
    corners = np.array(
        [(-half, -half), (-half, +half), (+half, -half), (+half, +half)]
    )
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
    S_minus, x_cam = sim_minus.build_scattering_matrix(
        angles_deg, flux_normalize=True
    )

    # T_plus
    scatterers_plus = shift_and_rotate_dipoles(
        base_positions, dx_plus, dz_plus, dth_plus
    )
    sim_plus = CoupledDipoleSimulator(**sim_kwargs)
    sim_plus.scatterers = scatterers_plus
    S_plus, _ = sim_plus.build_scattering_matrix(
        angles_deg, flux_normalize=True
    )

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
    print(
        f"Built derivative wrt {param_names[i]}, shape T_deriv = {S_deriv.shape}"
    )

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

    print(
        f"\nSingle-param MIS => {param_names[i]}, largest eigenvalue = {lam_max:.3g}"
    )
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
            val = (torch.conj(E_c).unsqueeze(0) @ (Fmn @ E_c.unsqueeze(1)))[
                0, 0
            ]
            J_mat[mu, nu] = val.real
    return J_mat


def torch_optimize_det(F_tensor, N_in, steps=300, lr=1e-3, photon_budget=1):
    E_param = torch.randn((N_in, 2), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        norm_init = torch.linalg.norm(E_param)
        E_param *= photon_budget**0.5 / norm_init

    opt = torch.optim.Adam([E_param], lr=lr)
    best_det_val = None
    best_vec = None

    for step in range(steps):
        opt.zero_grad()
        E_norm = photon_budget**0.5 * E_param / torch.linalg.norm(E_param)
        J = fisher_3x3(E_norm, F_tensor)
        det_val = torch.linalg.det(J)
        loss = -det_val
        loss.backward()
        opt.step()

        with torch.no_grad():
            cur_norm = torch.linalg.norm(E_norm)
            E_norm *= photon_budget**0.5 / cur_norm

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
        E_param *= (photon_budget**0.5) / norm_init

    opt = torch.optim.Adam([E_param], lr=lr)
    best_min_eig = None
    best_vec = None

    for step in range(steps):
        opt.zero_grad()
        E_norm = photon_budget**0.5 * E_param / torch.linalg.norm(E_param)
        J = fisher_3x3(E_norm, F_tensor)
        eigvals = torch.linalg.eigvalsh(J)
        min_eig = eigvals[0]  # smallest eigenvalue of J
        loss = -min_eig
        loss.backward()
        opt.step()

        with torch.no_grad():
            cur_norm = torch.linalg.norm(E_norm)
            E_norm *= (photon_budget**0.5) / cur_norm

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
        E_param = torch.randn(
            (N_in, 2), dtype=torch.float64, requires_grad=True
        )
        with torch.no_grad():
            norm_init = torch.linalg.norm(E_param)
            E_param *= photon_budget**0.5 / norm_init

        opt = torch.optim.Adam([E_param], lr=lr)
        run_loss_curve = []
        best_run_loss = None
        best_run_vec = None

        for step in range(steps):
            opt.zero_grad()
            E_norm = photon_budget**0.5 * E_param / torch.linalg.norm(E_param)
            J = fisher_3x3(E_norm, F_tensor)
            J_inv = torch.linalg.inv(J)
            trace_val = torch.trace(J_inv)
            loss = trace_val  # we want to minimize trace(J^-1)

            loss.backward()
            opt.step()

            with torch.no_grad():
                cur_norm = torch.linalg.norm(E_norm)
                E_norm *= photon_budget**0.5 / cur_norm

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
    E_opt_cplx = (
        best_vec[:, 0].cpu().numpy() + 1j * best_vec[:, 1].cpu().numpy()
    )
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
    F_tensor_torch,
    N_in=N_in,
    steps=4000,
    lr=1e-3,
    photon_budget=1,
    runs=multi_runs,
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
            val = (torch.conj(E_c).unsqueeze(0) @ (Fmn @ E_c.unsqueeze(1)))[
                0, 0
            ]
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
        results[wname] = {
            "diag": diag_vals,
            "trace": trace_val,
            "det": det_val,
        }

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
        x_idx + offset_2 / 2,
        det_data,
        width=offset_2,
        color=det_color,
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
E_opt_spec_complex_norm = normalize_wavefront(
    E_opt_spec_complex, photon_budget
)
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
    wavefronts_dict,
    F_tensor_torch,
    figure_title="Comparison of all wavefronts",
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


def compute_crlb_for_random_wavefronts(
    F_tensor, photon_budget, num_samples, N_in
):
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
        crlb_vals = compute_crlb_for_wavefront(
            wf_cplx_normalized, F_tensor_torch
        )
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
        random_wavefront_data = [
            random_wavefront_crlbs[:, i] for i in range(3)
        ]

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
    wavefronts_dict,
    F_tensor,
    wavefronts_to_show=None,
    title="Comparison of J and J^-1",
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
    F_tensor,
    N_in,
    steps,
    lr,
    photon_budget,
    approach_name,
    alpha=0.1,
    beta=1.0,
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
        E_param *= (photon_budget**0.5) / norm_init

    opt = torch.optim.Adam([E_param], lr=lr)
    best_loss = None
    best_vec = None
    loss_curve = []

    for _ in range(steps):
        opt.zero_grad()
        # Re-normalize wavefront each iteration
        E_norm = photon_budget**0.5 * E_param / torch.linalg.norm(E_param)
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
            E_param *= (photon_budget**0.5) / cur_norm

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
    J_ = compute_fisher_matrix_for_wavefront(
        wavefronts_dict[wf_name], F_tensor_torch
    )
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
print(
    "\nVisualizing multi-param fisher, CRLB, and J-and-inverse for all wavefronts."
)
visualize_multi_param_fisher_comparison(
    wavefronts_dict,
    F_tensor_torch,
    figure_title="All Wavefronts: Multi-Param Fisher",
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
        J_mdet = compute_fisher_matrix_for_wavefront(
            E_traceinv_norm, F_tensor_torch
        )
        crlb_array.append(np.sqrt(np.diag(np.linalg.inv(J_mdet))))
    else:
        crlb_array.append(results[k]["CRLB"])
crlb_array = np.array(crlb_array)
plt.figure(figsize=(6, 4))
x_inds = np.arange(len(all_keys))
width = 0.25
for i in range(3):
    plt.bar(
        x_inds + i * width - width,
        crlb_array[:, i],
        width,
        label=f"CRLB_{labels[i]}",
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
        E_param = torch.randn(
            (N_in, 2), dtype=torch.float64, requires_grad=True
        )
        with torch.no_grad():
            norm_init = torch.linalg.norm(E_param)
            E_param *= (photon_budget**0.5) / norm_init
        opt = torch.optim.Adam([E_param], lr=lr)
        loss_curve = []
        for step in range(steps):
            opt.zero_grad()
            E_norm = photon_budget**0.5 * E_param / torch.linalg.norm(E_param)
            J = fisher_3x3(
                E_norm, F_tensor
            )  # Assumed function to compute 3x3 Fisher matrix
            J_inv = torch.linalg.inv(J)
            loss = J_inv[0, 0]  # CRLB for x
            loss.backward()
            opt.step()
            with torch.no_grad():
                E_norm *= (photon_budget**0.5) / torch.linalg.norm(E_norm)
            loss_curve.append(loss.item())
            if step % 500 == 0:
                print(
                    f"CRLB Run {run+1}/{num_runs}, Step {step}: CRLB_x = {loss.item():.5f}"
                )
        all_loss_curves.append(loss_curve)
        if loss_curve[-1] < best_loss:
            best_loss = loss_curve[-1]
            best_vec = E_norm.detach().clone()
    E_opt_cplx = (
        best_vec[:, 0].cpu().numpy() + 1j * best_vec[:, 1].cpu().numpy()
    )
    return E_opt_cplx, best_loss, all_loss_curves


# 2. Maximize J_xx directly
def torch_optimize_J_xx(
    F_tensor, N_in, steps=4000, lr=1e-3, photon_budget=1, num_runs=10
):
    best_J_xx = -float("inf")
    best_vec = None
    all_loss_curves = []
    for run in range(num_runs):
        E_param = torch.randn(
            (N_in, 2), dtype=torch.float64, requires_grad=True
        )
        with torch.no_grad():
            norm_init = torch.linalg.norm(E_param)
            E_param *= (photon_budget**0.5) / norm_init
        opt = torch.optim.Adam([E_param], lr=lr)
        loss_curve = []
        for step in range(steps):
            opt.zero_grad()
            E_norm = photon_budget**0.5 * E_param / torch.linalg.norm(E_param)
            J = fisher_3x3(E_norm, F_tensor)
            J_xx = J[0, 0]  # Fisher information for x
            loss = -J_xx  # Minimize -J_xx to maximize J_xx
            loss.backward()
            opt.step()
            with torch.no_grad():
                E_norm *= (photon_budget**0.5) / torch.linalg.norm(E_norm)
            loss_curve.append(J_xx.item())  # Store J_xx value
            if step % 500 == 0:
                print(
                    f"J_xx Run {run+1}/{num_runs}, Step {step}: J_xx = {J_xx.item():.5f}"
                )
        all_loss_curves.append(loss_curve)
        if loss_curve[-1] > best_J_xx:
            best_J_xx = loss_curve[-1]
            best_vec = E_norm.detach().clone()
    E_opt_cplx = (
        best_vec[:, 0].cpu().numpy() + 1j * best_vec[:, 1].cpu().numpy()
    )
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
ax.set_title(
    "CRLB Comparison: MIS_x vs. Direct CRLB_xx vs. Direct J_xx Optimization"
)
ax.legend()
ax.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()

# Print CRLB values
for name, crlb in zip(
    wavefront_names, [crlb_mis, crlb_opt_crlb, crlb_opt_J_xx]
):
    print(f"\nCRLB Values for {name} Wavefront:")
    for i, param in enumerate(param_labels):
        print(f"  {param}: {crlb[i]:.3e}")


# %% 13) Optimization Maximizing det(J_interest|nuisance_rotation) [MULTI-RUN]


def compute_fisher_interest_nuisance(J_full, nuisance_param_index=2):
    """
    Computes the Fisher information matrix for parameters of interest,
    accounting for a single nuisance parameter.
    Assumes J_full is a 3x3 torch tensor (classical FIM).
    nuisance_param_index = 0 for x, 1 for z, 2 for theta (rotation).
    Returns the 2x2 FIM for the parameters of interest.
    """
    if nuisance_param_index == 0:  # x is nuisance, interest are z, theta
        J11 = J_full[1:, 1:]
        J12 = J_full[1:, 0].unsqueeze(1)
        J21 = J_full[0, 1:].unsqueeze(0)
        J22 = J_full[0, 0].unsqueeze(0).unsqueeze(0)  # Make it 1x1
    elif nuisance_param_index == 1:  # z is nuisance, interest are x, theta
        indices_interest = [0, 2]
        J11 = J_full[
            torch.tensor(indices_interest, device=J_full.device)[:, None],
            torch.tensor(indices_interest, device=J_full.device),
        ]
        J12 = J_full[indices_interest, nuisance_param_index].unsqueeze(1)
        J21 = J_full[nuisance_param_index, indices_interest].unsqueeze(0)
        J22 = (
            J_full[nuisance_param_index, nuisance_param_index]
            .unsqueeze(0)
            .unsqueeze(0)
        )  # Make it 1x1
    elif nuisance_param_index == 2:  # theta is nuisance, interest are x, z
        J11 = J_full[:2, :2]
        J12 = J_full[:2, 2].unsqueeze(1)
        J21 = J_full[2, :2].unsqueeze(0)
        J22 = J_full[2, 2].unsqueeze(0).unsqueeze(0)  # Make it 1x1
    else:
        raise ValueError("nuisance_param_index must be 0, 1, or 2")

    # Check determinant of J22 before inversion
    det_J22 = torch.det(J22)
    if (
        abs(det_J22.item()) < 1e-12
    ):  # Using a small threshold for near-singularity
        # print(
        #     "Warning: J22 is singular or near singular in compute_fisher_interest_nuisance. Applying regularization."
        # )
        # Apply regularization to J22. Ensure eye matrix is on the same device and dtype.
        J22_inv = torch.linalg.inv(
            J22
            + torch.eye(J22.shape[0], device=J22.device, dtype=J22.dtype)
            * 1e-12
        )
    else:
        J22_inv = torch.linalg.inv(J22)

    J_interest_nuisance = J11 - J12 @ J22_inv @ J21
    return J_interest_nuisance


torch.manual_seed(13)  # Consistent seed for this section's start


def torch_optimize_det_interest_nuisance_multirun(
    F_tensor,
    N_in,
    steps=1000,
    lr=1e-3,
    photon_budget=1,
    nuisance_idx=2,
    runs=5,
):
    best_det_val_overall = None
    best_vec_overall = None
    all_loss_curves_all_runs = []

    for run_i in range(runs):
        E_param = torch.randn(
            (N_in, 2),
            dtype=torch.float64,
            requires_grad=True,
            device=F_tensor.device,
        )
        with torch.no_grad():
            norm_init = torch.linalg.norm(E_param)
            if norm_init > 1e-9:
                E_param *= (photon_budget**0.5) / norm_init
            else:
                E_param = torch.randn(
                    (N_in, 2),
                    dtype=torch.float64,
                    requires_grad=True,
                    device=F_tensor.device,
                )
                E_param *= (photon_budget**0.5) / torch.linalg.norm(E_param)

        opt = torch.optim.Adam([E_param], lr=lr)

        current_run_losses = []
        best_det_this_run = None
        best_vec_this_run = None

        print(f"\nDet Interest (Nuisance) Opt - Run {run_i+1}/{runs}")
        for step in range(steps):
            opt.zero_grad()
            # Normalize E_param before using it to calculate J
            with torch.no_grad():  # Ensure normalization doesn't affect gradients for E_param
                current_norm_for_E_norm = torch.linalg.norm(E_param)
                if current_norm_for_E_norm > 1e-9:
                    E_norm = (
                        (photon_budget**0.5)
                        * E_param
                        / current_norm_for_E_norm
                    )
                else:  # If E_param somehow became zero, re-initialize or handle
                    E_norm = E_param  # Avoid division by zero, loss will likely be NaN or inf

            # Ensure E_norm requires grad if E_param does, or use E_param and normalize inside loss calculation
            # For simplicity, let's re-attach E_norm to the graph if E_param is part of it
            # Or, more directly, calculate loss based on E_param and handle normalization within
            # Let's use E_param directly and normalize inside the path to loss
            E_param_normalized_for_loss = (
                (photon_budget**0.5) * E_param / torch.linalg.norm(E_param)
            )

            J_full = fisher_3x3(E_param_normalized_for_loss, F_tensor)
            J_interest = compute_fisher_interest_nuisance(
                J_full, nuisance_param_index=nuisance_idx
            )

            current_det_val_interest = torch.linalg.det(J_interest)
            loss = -current_det_val_interest  # We want to maximize determinant

            if torch.isnan(loss) or torch.isinf(loss):
                pass  # Skip backward and step if loss is invalid
            else:
                loss.backward()
                opt.step()

            with torch.no_grad():
                cur_norm_e_param = torch.linalg.norm(E_param)
                if cur_norm_e_param > 1e-9:
                    E_param *= (photon_budget**0.5) / cur_norm_e_param

                # For reporting and best_vec, use the current state of E_param, normalized
                E_norm_report = (
                    (photon_budget**0.5) * E_param / torch.linalg.norm(E_param)
                )
                J_full_report = fisher_3x3(E_norm_report, F_tensor)
                J_interest_report = compute_fisher_interest_nuisance(
                    J_full_report, nuisance_param_index=nuisance_idx
                )
                current_det_val_report_item = torch.linalg.det(
                    J_interest_report
                ).item()

            if not math.isnan(current_det_val_report_item) and not math.isinf(
                current_det_val_report_item
            ):
                if (
                    best_det_this_run is None
                    or current_det_val_report_item > best_det_this_run
                ):
                    best_det_this_run = current_det_val_report_item
                    best_vec_this_run = E_norm_report.detach().clone()
                current_run_losses.append(-current_det_val_report_item)
            else:
                current_run_losses.append(float("nan"))

            if step % 200 == 0:
                # Corrected print statement
                val_to_print = (
                    f"{current_det_val_report_item:.3e}"
                    if not (
                        math.isnan(current_det_val_report_item)
                        or math.isinf(current_det_val_report_item)
                    )
                    else "NaN/Inf"
                )
                print(f"  Step {step:4d} => det(J_interest)={val_to_print}")

        all_loss_curves_all_runs.append(current_run_losses)
        val_this_run_to_print = (
            f"{best_det_this_run:.3e}"
            if best_det_this_run is not None
            and not (
                math.isnan(best_det_this_run) or math.isinf(best_det_this_run)
            )
            else "NaN/Inf or No Valid Found"
        )
        print(
            f"Run {run_i+1}/{runs} final best det(J_interest) = {val_this_run_to_print}"
        )

        if best_vec_this_run is not None and best_det_this_run is not None:
            if not (
                math.isnan(best_det_this_run) or math.isinf(best_det_this_run)
            ):
                if (
                    best_det_val_overall is None
                    or best_det_this_run > best_det_val_overall
                ):
                    best_det_val_overall = best_det_this_run
                    best_vec_overall = best_vec_this_run

    return (
        (
            best_vec_overall.cpu().numpy()
            if best_vec_overall is not None
            else None
        ),
        best_det_val_overall,
        all_loss_curves_all_runs,
    )


print("\n--- Maximizing Det(J_interest|nuisance_rotation) [MULTI-RUN] ---")
# Assuming multi_runs is defined in your script (e.g., multi_runs = 5)
# Also ensure F_tensor_torch, N_in, photon_budget, kx_array, base_sim are defined
(
    E_det_interest_nuisance_parts,
    best_det_interest_nuisance,
    all_losses_det_interest_nuisance,
) = torch_optimize_det_interest_nuisance_multirun(
    F_tensor_torch,
    N_in=N_in,
    steps=2000,
    lr=1e-3,
    photon_budget=photon_budget,
    nuisance_idx=2,
    runs=multi_runs,
)

if (
    E_det_interest_nuisance_parts is not None
    and best_det_interest_nuisance is not None
):
    E_det_interest_nuisance = (
        E_det_interest_nuisance_parts[:, 0]
        + 1j * E_det_interest_nuisance_parts[:, 1]
    )
    print(
        f"Overall Best det(J_interest|nuisance_rotation) after multi-run = {best_det_interest_nuisance:.3e}"
    )

    base_sim.reset()
    base_sim.set_input_field_kx(kx_array, E_det_interest_nuisance)
    base_sim.solve_fields()
    base_sim.visualize_intensity()

    plt.figure(figsize=(7, 5))
    colors_loss_det_interest = plt.cm.plasma(
        np.linspace(0, 1, len(all_losses_det_interest_nuisance))
    )
    for i, lc in enumerate(all_losses_det_interest_nuisance):
        lc_plot = [val if not math.isnan(val) else np.nan for val in lc]
        plt.plot(
            lc_plot,
            color=colors_loss_det_interest[i],
            alpha=0.7,
            label=f"Run {i+1}",
        )
    plt.xlabel("Iteration")
    plt.ylabel("-det(J_interest|nuisance_rotation)")
    plt.title(
        "Loss Curves for Max Det(J_interest|nuisance_rotation) [Multi-Run]"
    )

    valid_losses_flat = [
        x
        for x_list in all_losses_det_interest_nuisance
        for x in x_list
        if not (math.isnan(x) or math.isinf(x))
    ]
    if valid_losses_flat:
        min_abs_loss = np.min(np.abs(valid_losses_flat))
        linthresh_val = (
            min_abs_loss / 10 if min_abs_loss > 1e-9 else 1e-5
        )  # ensure linthresh is positive
    else:
        linthresh_val = 1e-5  # Default if all are NaN/Inf
    plt.yscale(
        "symlog", linthresh=linthresh_val, linscale=0.5
    )  # Adjusted linthresh logic
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()
else:
    print(
        "Optimization for Max Det(J_interest|nuisance_rotation) [Multi-Run] failed to find a valid wavefront or result."
    )
    E_det_interest_nuisance = None

key_for_dict_det_psi_phi = r"Max Det($J_{\psi|\phi}$)"
if E_det_interest_nuisance is not None:
    wavefronts_dict[key_for_dict_det_psi_phi] = normalize_wavefront(
        E_det_interest_nuisance, photon_budget
    )
else:
    if key_for_dict_det_psi_phi not in wavefronts_dict:
        wavefronts_dict[key_for_dict_det_psi_phi] = np.zeros(
            N_in, dtype=complex
        )


# %% 14) Optimization Minimizing Tr(J_interest|nuisance_rotation ^ -1)

torch.manual_seed(14)  # New seed


def torch_optimize_traceinv_interest_nuisance(
    F_tensor,
    N_in,
    steps=3000,
    lr=1e-3,
    photon_budget=1,
    nuisance_idx=2,
    runs=5,
):
    best_loss_overall = None
    best_vec_overall = None
    all_loss_curves_all_runs = []

    for run_i in range(runs):
        E_param = torch.randn(
            (N_in, 2),
            dtype=torch.float64,
            requires_grad=True,
            device=F_tensor.device,
        )
        with torch.no_grad():
            norm_init = torch.linalg.norm(E_param)
            E_param *= (photon_budget**0.5) / norm_init

        opt = torch.optim.Adam([E_param], lr=lr)
        current_run_losses = []
        best_loss_this_run = float("inf")
        best_vec_this_run = None

        for step in range(steps):
            opt.zero_grad()
            E_norm = (
                (photon_budget**0.5) * E_param / torch.linalg.norm(E_param)
            )
            J_full = fisher_3x3(E_norm, F_tensor)
            J_interest = compute_fisher_interest_nuisance(
                J_full, nuisance_param_index=nuisance_idx
            )

            current_loss_val = float("nan")
            try:
                if torch.det(J_interest) == 0:  # Check for singularity
                    # print(f"TraceInv Interest Opt Run {run_i+1} Step {step:4d} => J_interest singular, assign high loss.")
                    loss = torch.tensor(
                        float("inf"), device=F_tensor.device
                    )  # High loss for singular matrix
                else:
                    J_interest_inv = torch.linalg.inv(J_interest)
                    loss = torch.trace(
                        J_interest_inv
                    )  # Minimize trace of inverse

                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    opt.step()
                current_loss_val = loss.item()
            except torch.linalg.LinAlgError:
                # print(f"TraceInv Interest Opt Run {run_i+1} Step {step:4d} => LinAlgError, assign high loss.")
                current_loss_val = float("inf")

            with torch.no_grad():
                cur_norm_e = torch.linalg.norm(E_param)
                E_param *= (photon_budget**0.5) / cur_norm_e

            current_run_losses.append(current_loss_val)
            if current_loss_val < best_loss_this_run:
                best_loss_this_run = current_loss_val
                # Update best_vec based on the current E_param after normalization
                E_norm_update = (
                    (photon_budget**0.5) * E_param / torch.linalg.norm(E_param)
                )
                best_vec_this_run = E_norm_update.detach().clone()

            if step % 500 == 0:
                print(
                    f"TraceInv Interest Opt Run {run_i+1}/{runs} Step {step:4d} => Tr(J_interest^-1)={current_loss_val:.3e}"
                )

        all_loss_curves_all_runs.append(current_run_losses)
        print(
            f"Run {run_i+1}/{runs} final Tr(J_interest^-1) = {best_loss_this_run:.3e}"
        )

        if (
            best_vec_this_run is not None
        ):  # Check if a valid vector was found for this run
            if (
                best_loss_overall is None
                or best_loss_this_run < best_loss_overall
            ):
                best_loss_overall = best_loss_this_run
                best_vec_overall = best_vec_this_run

    return (
        (
            best_vec_overall.cpu().numpy()
            if best_vec_overall is not None
            else None
        ),
        best_loss_overall,
        all_loss_curves_all_runs,
    )


print("\n--- Minimizing Tr(J_interest|nuisance_rotation ^ -1) ---")
(
    E_traceinv_interest_nuisance_parts,
    best_traceinv_interest_nuisance,
    all_traceinv_interest_loss_curves,
) = torch_optimize_traceinv_interest_nuisance(
    F_tensor_torch,
    N_in=N_in,
    steps=3000,
    lr=1e-3,
    photon_budget=photon_budget,
    nuisance_idx=2,
    runs=multi_runs,
)

if E_traceinv_interest_nuisance_parts is not None:
    E_traceinv_interest_nuisance = (
        E_traceinv_interest_nuisance_parts[:, 0]
        + 1j * E_traceinv_interest_nuisance_parts[:, 1]
    )
    print(
        f"Best Tr(J_interest|nuisance_rotation ^ -1) = {best_traceinv_interest_nuisance:.3e}"
    )

    base_sim.reset()
    base_sim.set_input_field_kx(kx_array, E_traceinv_interest_nuisance)
    base_sim.solve_fields()
    base_sim.visualize_intensity()
    plt.title(
        f"Wavefront for Min Tr(J_interest|nuisance_rotation ^-1)\nTr = {best_traceinv_interest_nuisance:.2e}"
    )
    plt.show()

    plt.figure(figsize=(7, 5))
    colors = plt.cm.viridis(
        np.linspace(0, 1, len(all_traceinv_interest_loss_curves))
    )
    for i, lc in enumerate(all_traceinv_interest_loss_curves):
        plt.plot(lc, color=colors[i], alpha=0.7, label=f"Run {i+1}")
    plt.xlabel("Iteration")
    plt.ylabel("Tr(J_interest|nuisance_rotation ^ -1)")
    plt.title("Loss for Min Tr(J_interest|nuisance_rotation ^ -1)")
    plt.yscale("symlog", linthresh=1e-1)
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()
else:
    print("Optimization for Min Tr(J_interest|nuisance_rotation ^ -1) failed.")
    E_traceinv_interest_nuisance = None

if E_traceinv_interest_nuisance is not None:
    wavefronts_dict[r"Min Tr($J_{\psi|\phi}^{-1}$)"] = normalize_wavefront(
        E_traceinv_interest_nuisance, photon_budget
    )
else:
    wavefronts_dict[r"Min Tr($J_{\psi|\phi}^{-1}$)"] = np.zeros(
        N_in, dtype=complex
    )

# %% 15) Optimization Maximizing Smallest Eigenvalue of J_interest|nuisance_rotation [MULTI-RUN]

torch.manual_seed(15)  # Consistent seed for this section's start


def torch_optimize_spec_interest_nuisance_multirun(
    F_tensor,
    N_in,
    steps=2000,
    lr=1e-3,
    photon_budget=1,
    nuisance_idx=2,
    runs=5,
):
    best_min_eig_val_overall = None
    best_vec_overall = None
    all_loss_curves_all_runs = []

    for run_i in range(runs):
        E_param = torch.randn(
            (N_in, 2),
            dtype=torch.float64,
            requires_grad=True,
            device=F_tensor.device,
        )
        with torch.no_grad():
            norm_init = torch.linalg.norm(E_param)
            if (
                norm_init > 1e-12
            ):  # Avoid division by zero for very small initial norms
                E_param *= (photon_budget**0.5) / norm_init
            else:  # Re-initialize if norm is too small
                E_param = torch.randn(
                    (N_in, 2),
                    dtype=torch.float64,
                    requires_grad=True,
                    device=F_tensor.device,
                )
                E_param *= (photon_budget**0.5) / torch.linalg.norm(
                    E_param
                )  # Re-normalize

        opt = torch.optim.Adam([E_param], lr=lr)

        current_run_losses = []
        best_min_eig_this_run = None
        best_vec_this_run = None

        print(f"\nSpec Interest (Nuisance) Opt - Run {run_i+1}/{runs}")
        for step in range(steps):
            opt.zero_grad()

            current_norm_e_param_for_loss = torch.linalg.norm(E_param)
            # Let potential division by zero or small norm propagate if it occurs here, per instructions
            E_param_normalized_for_loss = (
                (photon_budget**0.5) * E_param / current_norm_e_param_for_loss
            )

            J_full = fisher_3x3(E_param_normalized_for_loss, F_tensor)
            J_interest = compute_fisher_interest_nuisance(
                J_full, nuisance_param_index=nuisance_idx
            )

            # Calculate eigenvalues. If J_interest is not positive semi-definite or singular,
            # eigvalsh might behave in specific ways or error out. Let it happen.
            eigvals_interest = torch.linalg.eigvalsh(J_interest)
            min_eig_interest = eigvals_interest[0]  # Smallest eigenvalue
            loss = (
                -min_eig_interest
            )  # We want to maximize the smallest eigenvalue

            # Per instructions, let NaN/Inf propagate. Do not skip backward() or step().
            loss.backward()
            opt.step()

            # Post-step operations
            with torch.no_grad():
                cur_norm_e_param_after_step = torch.linalg.norm(E_param)
                if (
                    cur_norm_e_param_after_step > 1e-12
                ):  # Check to avoid division by zero if E_param became zero
                    E_param *= (
                        photon_budget**0.5
                    ) / cur_norm_e_param_after_step
                # If E_param becomes NaN/Inf, this normalization will also become NaN/Inf.

                E_norm_report = (
                    (photon_budget**0.5) * E_param / torch.linalg.norm(E_param)
                )
                # Recalculate J_full_report and J_interest_report for accurate reporting if E_param changed
                J_full_report = fisher_3x3(E_norm_report, F_tensor)
                J_interest_report = compute_fisher_interest_nuisance(
                    J_full_report, nuisance_param_index=nuisance_idx
                )

                eigvals_report = torch.linalg.eigvalsh(J_interest_report)
                current_min_eig_report_item = eigvals_report[
                    0
                ].item()  # Convert to float for Python logic

            # Store loss for plotting (it's -min_eig)
            # If min_eig_interest was NaN/Inf, loss.item() will be NaN/Inf
            current_run_losses.append(loss.item())

            # Update best for this run (use float values for comparison)
            if not (
                math.isnan(current_min_eig_report_item)
                or math.isinf(current_min_eig_report_item)
            ):
                if (
                    best_min_eig_this_run is None
                    or current_min_eig_report_item > best_min_eig_this_run
                ):
                    best_min_eig_this_run = current_min_eig_report_item
                    best_vec_this_run = E_norm_report.detach().clone()

            if step % 200 == 0:
                val_to_print = (
                    f"{current_min_eig_report_item:.3e}"
                    if not (
                        math.isnan(current_min_eig_report_item)
                        or math.isinf(current_min_eig_report_item)
                    )
                    else "NaN/Inf"
                )
                print(
                    f"  Step {step:4d} => min_eig(J_interest)={val_to_print}"
                )

        all_loss_curves_all_runs.append(current_run_losses)
        val_this_run_to_print = (
            f"{best_min_eig_this_run:.3e}"
            if best_min_eig_this_run is not None
            and not (
                math.isnan(best_min_eig_this_run)
                or math.isinf(best_min_eig_this_run)
            )
            else "NaN/Inf or No Valid Found"
        )
        print(
            f"Run {run_i+1}/{runs} final best min_eig(J_interest) = {val_this_run_to_print}"
        )

        if best_vec_this_run is not None and best_min_eig_this_run is not None:
            if not (
                math.isnan(best_min_eig_this_run)
                or math.isinf(best_min_eig_this_run)
            ):
                if (
                    best_min_eig_val_overall is None
                    or best_min_eig_this_run > best_min_eig_val_overall
                ):
                    best_min_eig_val_overall = best_min_eig_this_run
                    best_vec_overall = best_vec_this_run

    return (
        (
            best_vec_overall.cpu().numpy()
            if best_vec_overall is not None
            else None
        ),
        best_min_eig_val_overall,
        all_loss_curves_all_runs,
    )


print(
    "\n--- Maximizing Min Eigenvalue of J_interest|nuisance_rotation [MULTI-RUN] ---"
)
# Assuming multi_runs is defined in your script
(
    E_spec_interest_nuisance_parts,
    best_spec_interest_nuisance,
    all_losses_spec_interest_nuisance,
) = torch_optimize_spec_interest_nuisance_multirun(
    F_tensor_torch,
    N_in=N_in,
    steps=2000,
    lr=1e-3,
    photon_budget=photon_budget,
    nuisance_idx=2,
    runs=multi_runs,
)

if (
    E_spec_interest_nuisance_parts is not None
    and best_spec_interest_nuisance is not None
):
    E_spec_interest_nuisance = (
        E_spec_interest_nuisance_parts[:, 0]
        + 1j * E_spec_interest_nuisance_parts[:, 1]
    )
    print(
        f"Overall Best min_eig(J_interest|nuisance_rotation) after multi-run = {best_spec_interest_nuisance:.3e}"
    )

    base_sim.reset()
    base_sim.set_input_field_kx(kx_array, E_spec_interest_nuisance)
    base_sim.solve_fields()
    base_sim.visualize_intensity()
    plt.title(
        f"Wavefront for Max MinEig(J_interest|nuisance_rotation) [Multi-Run]\nMinEig = {best_spec_interest_nuisance:.2e}"
    )
    plt.show()

    plt.figure(figsize=(7, 5))
    colors_loss_spec_interest = plt.cm.viridis(
        np.linspace(0, 1, len(all_losses_spec_interest_nuisance))
    )  # Changed colormap for variety
    for i, lc in enumerate(all_losses_spec_interest_nuisance):
        lc_plot = [val if not math.isnan(val) else np.nan for val in lc]
        plt.plot(
            lc_plot,
            color=colors_loss_spec_interest[i],
            alpha=0.7,
            label=f"Run {i+1}",
        )
    plt.xlabel("Iteration")
    plt.ylabel(
        "-min_eig(J_interest|nuisance_rotation)"
    )  # Loss is negative of the value we maximize
    plt.title(
        "Loss Curves for Max MinEig(J_interest|nuisance_rotation) [Multi-Run]"
    )

    valid_losses_flat = [
        x
        for x_list in all_losses_spec_interest_nuisance
        for x in x_list
        if not (math.isnan(x) or math.isinf(x))
    ]
    if valid_losses_flat:
        # For symlog, linthresh must be positive. Calculate based on min absolute non-zero value.
        abs_losses = np.abs(valid_losses_flat)
        min_abs_loss = (
            np.min(abs_losses[np.nonzero(abs_losses)])
            if len(abs_losses[np.nonzero(abs_losses)]) > 0
            else 1e-5
        )
        linthresh_val = min_abs_loss / 10 if min_abs_loss > 1e-9 else 1e-5
    else:
        linthresh_val = 1e-5
    plt.yscale("symlog", linthresh=linthresh_val, linscale=0.5)
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()
else:
    print(
        "Optimization for Max MinEig(J_interest|nuisance_rotation) [Multi-Run] failed to find a valid wavefront or result."
    )
    E_spec_interest_nuisance = None

key_for_dict_spec_psi_phi = r"Max MinEig($J_{\psi|\phi}$)"
if E_spec_interest_nuisance is not None:
    wavefronts_dict[key_for_dict_spec_psi_phi] = normalize_wavefront(
        E_spec_interest_nuisance, photon_budget
    )
else:
    if key_for_dict_spec_psi_phi not in wavefronts_dict:
        wavefronts_dict[key_for_dict_spec_psi_phi] = np.zeros(
            N_in, dtype=complex
        )


# %% 16) Updated Comparisons: Multi-Param Fisher, CRLB, J and J^-1

print(
    "\n--- Updated Full Wavefront Comparisons (including nuisance-aware) ---"
)

# Ensure all wavefronts are normalized and added to the dictionary correctly
# The individual sections 13, 14, 15 should have handled adding their respective wavefronts.
# Check if any are None and replace with a zero vector if so, to avoid errors in visualization functions,
# though ideally, optimization should yield a valid vector.
for key in [
    r"Max Det($J_{\psi|\phi}$)",
    r"Min Tr($J_{\psi|\phi}^{-1}$)",
    r"Max MinEig($J_{\psi|\phi}$)",
]:
    if key not in wavefronts_dict or wavefronts_dict[key] is None:
        print(
            f"Warning: Wavefront for {key} was not successfully optimized. Using placeholder for visualization."
        )
        wavefronts_dict[key] = normalize_wavefront(
            np.random.randn(N_in) + 1j * np.random.randn(N_in), photon_budget
        )


# Re-run visualization_multi_param_fisher_comparison
print(
    "\nVisualizing Multi-Param Fisher properties for ALL wavefronts (including nuisance-aware):"
)
visualize_multi_param_fisher_comparison(
    wavefronts_dict,
    F_tensor_torch,
    figure_title="Comparison of ALL Wavefronts (incl. Nuisance-Aware)",
)

# Re-run visualize_crlb_comparison
print("\nVisualizing CRLB for ALL wavefronts (including nuisance-aware):")
visualize_crlb_comparison(
    wavefronts_dict,
    F_tensor_torch,
    sim_kwargs,  # Make sure sim_kwargs is defined
    base_dipoles,  # Make sure base_dipoles is defined
    angles_deg,  # Make sure angles_deg is defined
    photon_budget,
    kx_array,  # Make sure kx_array is defined
    figure_title="CRLB Comparison for ALL Wavefronts (incl. Nuisance-Aware)",
    show_plane_waves=True,
    show_random_wavefronts=True,
)

# Re-run visualize_J_and_inverse
print(
    "\nVisualizing J and J^-1 for ALL wavefronts (including nuisance-aware):"
)
visualize_J_and_inverse(
    wavefronts_dict,
    F_tensor_torch,
    title="Comparison of J and J^-1 for ALL Wavefronts (incl. Nuisance-Aware)",
)

# %% 17) Updated Pareto Frontier Plot (including nuisance-aware wavefronts)

print("\n--- Updated Pareto Frontier Plot (including nuisance-aware) ---")

# Ensure all_keys for CRLB calculations and plotting includes the new wavefronts
all_keys_pareto_updated = list(
    wavefronts_dict.keys()
)  # Use all keys from the dictionary

crlb_dict_pareto_updated = {}
for key in all_keys_pareto_updated:
    wf = wavefronts_dict[key]
    if wf is None or np.all(
        wf == 0
    ):  # Handle cases where optimization might have failed
        print(
            f"Warning: Wavefront for key '{key}' is None or zeros. Assigning NaN CRLBs for Pareto plot."
        )
        crlb_dict_pareto_updated[key] = np.array(
            [float("nan"), float("nan"), float("nan")]
        )
        continue
    try:
        J_mat_pareto = compute_fisher_matrix_for_wavefront(wf, F_tensor_torch)
        if np.linalg.det(J_mat_pareto) == 0:
            crlb_dict_pareto_updated[key] = np.array(
                [float("inf"), float("inf"), float("inf")]
            )
        else:
            crlb_pareto = np.sqrt(np.diag(np.linalg.inv(J_mat_pareto)))
            crlb_dict_pareto_updated[key] = crlb_pareto
    except Exception as e:
        print(f"Error computing CRLB for {key} in Pareto: {e}")
        crlb_dict_pareto_updated[key] = np.array(
            [float("nan"), float("nan"), float("nan")]
        )


crlb_array_pareto_updated = np.array(
    [crlb_dict_pareto_updated[key] for key in all_keys_pareto_updated]
)

# Identify nuisance-aware and other multi-objective optimization wavefronts for different markers/colors
nuisance_aware_keys = [
    r"Max Det($J_{\psi|\phi}$)",
    r"Min Tr($J_{\psi|\phi}^{-1}$)",
    r"Max MinEig($J_{\psi|\phi}$)",
]
multi_obj_original_keys = [
    "LogLin",
    "OffDiag",
    "SubDet",
]  # From your section 11

markers_pareto_updated = []
colors_pareto_updated = []

for key in all_keys_pareto_updated:
    if key in nuisance_aware_keys:
        markers_pareto_updated.append(
            "^"
        )  # Triangle up for new nuisance-aware
        colors_pareto_updated.append("green")
    elif key in multi_obj_original_keys:
        markers_pareto_updated.append(
            "s"
        )  # Square for original multi-objective
        colors_pareto_updated.append("red")
    elif key in [
        r"Max $\mathcal{J}_x$",
        r"Max $\mathcal{J}_z$",
        r"Max $\mathcal{J}_\theta$",
    ]:
        markers_pareto_updated.append("x")  # Cross for single-param MIS
        colors_pareto_updated.append("purple")
    else:
        markers_pareto_updated.append(
            "o"
        )  # Circle for others (Max Tr(J), Max Det(J), etc.)
        colors_pareto_updated.append("blue")

fig_pareto_updated, axes_pareto_updated = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: (CRLB_x + CRLB_z) vs CRLB_theta
# Filter out NaN values for plotting
valid_indices_plot1 = (
    ~np.isnan(crlb_array_pareto_updated[:, 0])
    & ~np.isnan(crlb_array_pareto_updated[:, 1])
    & ~np.isnan(crlb_array_pareto_updated[:, 2])
)
x1_pareto = (
    crlb_array_pareto_updated[valid_indices_plot1, 0]
    + crlb_array_pareto_updated[valid_indices_plot1, 1]
)
y1_pareto = crlb_array_pareto_updated[valid_indices_plot1, 2]
keys_plot1 = [
    all_keys_pareto_updated[i]
    for i, valid in enumerate(valid_indices_plot1)
    if valid
]
markers_plot1 = [
    markers_pareto_updated[i]
    for i, valid in enumerate(valid_indices_plot1)
    if valid
]
colors_plot1 = [
    colors_pareto_updated[i]
    for i, valid in enumerate(valid_indices_plot1)
    if valid
]


ax1_pareto = axes_pareto_updated[0]
for i in range(len(keys_plot1)):
    ax1_pareto.scatter(
        x1_pareto[i],
        y1_pareto[i],
        s=100,
        marker=markers_plot1[i],
        color=colors_plot1[i],
        label=keys_plot1[i] if i < 10 else None,
        alpha=0.7,  # Limit labels to avoid clutter
    )
# Annotations can be very cluttered; consider adding them selectively for key points or using interactive plots
# for i, key in enumerate(keys_plot1):
#     ax1_pareto.annotate(key, (x1_pareto[i], y1_pareto[i]), xytext=(5,5), textcoords="offset points", fontsize=8, alpha=0.7)

ax1_pareto.set_xlabel("CRLB_x + CRLB_z (Sum of Position CRLBs)")
ax1_pareto.set_ylabel("CRLB_θ (Rotation CRLB)")
ax1_pareto.set_title("Pareto: Position Estimation vs. Rotation Estimation")
ax1_pareto.grid(True, linestyle=":")
ax1_pareto.set_xscale("log")
ax1_pareto.set_yscale("log")
# ax1_pareto.legend(fontsize='small', loc='best') # Legend can be very crowded

# Plot 2: CRLB_x vs CRLB_z
valid_indices_plot2 = ~np.isnan(crlb_array_pareto_updated[:, 0]) & ~np.isnan(
    crlb_array_pareto_updated[:, 1]
)
x2_pareto = crlb_array_pareto_updated[valid_indices_plot2, 0]
y2_pareto = crlb_array_pareto_updated[valid_indices_plot2, 1]
keys_plot2 = [
    all_keys_pareto_updated[i]
    for i, valid in enumerate(valid_indices_plot2)
    if valid
]
markers_plot2 = [
    markers_pareto_updated[i]
    for i, valid in enumerate(valid_indices_plot2)
    if valid
]
colors_plot2 = [
    colors_pareto_updated[i]
    for i, valid in enumerate(valid_indices_plot2)
    if valid
]

ax2_pareto = axes_pareto_updated[1]
for i in range(len(keys_plot2)):
    ax2_pareto.scatter(
        x2_pareto[i],
        y2_pareto[i],
        s=100,
        marker=markers_plot2[i],
        color=colors_plot2[i],
        label=keys_plot2[i] if i < 10 else None,
        alpha=0.7,
    )

ax2_pareto.set_xlabel("CRLB_x")
ax2_pareto.set_ylabel("CRLB_z")
ax2_pareto.set_title("Pareto: CRLB_x vs. CRLB_z")
ax2_pareto.grid(True, linestyle=":")
ax2_pareto.set_xscale("log")
ax2_pareto.set_yscale("log")
# ax2_pareto.legend(fontsize='small', loc='best')


# Create a single legend for the figure
handles, labels = [], []
# Collect unique legend entries
unique_labels_for_legend = {}
for i in range(len(all_keys_pareto_updated)):
    # Use a generic label for each marker type to simplify the legend
    marker_type_label = "Other Optimal"
    if all_keys_pareto_updated[i] in nuisance_aware_keys:
        marker_type_label = "Nuisance-Aware ($J_{\psi|\phi}$)"
    elif all_keys_pareto_updated[i] in multi_obj_original_keys:
        marker_type_label = "Multi-Objective (Original)"
    elif all_keys_pareto_updated[i] in [
        r"Max $\mathcal{J}_x$",
        r"Max $\mathcal{J}_z$",
        r"Max $\mathcal{J}_\theta$",
    ]:
        marker_type_label = "Single-Param MIS"

    if marker_type_label not in unique_labels_for_legend:
        # Create a dummy scatter plot point for the legend entry
        unique_labels_for_legend[marker_type_label] = plt.Line2D(
            [0],
            [0],
            marker=markers_pareto_updated[i],
            color=colors_pareto_updated[i],
            linestyle="None",
            markersize=10,
        )

fig_pareto_updated.legend(
    unique_labels_for_legend.values(),
    unique_labels_for_legend.keys(),
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 0.03),
)

plt.tight_layout(
    rect=[0, 0.05, 1, 0.95]
)  # Adjust layout to make space for legend
plt.suptitle(
    "Updated Pareto Frontier Comparison (incl. Nuisance-Aware Wavefronts)",
    fontsize=14,
)
plt.show()

# %% 18) Pareto Frontier Plot with Individual Point Labels

print("\n--- Pareto Frontier Plot with Individual Labels ---")

all_keys_for_plot_18 = list(wavefronts_dict.keys())
crlb_data_for_plot_18 = {}

for key in all_keys_for_plot_18:
    wf = wavefronts_dict[key]
    try:
        crlb_vals = compute_crlb_for_wavefront(wf, F_tensor_torch)
        # Ensure crlb_vals are float, not complex due to numerical precision issues
        crlb_data_for_plot_18[key] = np.real(crlb_vals)
    except np.linalg.LinAlgError:  # Catching this specifically for plotting
        print(
            f"Warning: LinAlgError computing CRLB for '{key}'. Assigning [inf, inf, inf]."
        )
        crlb_data_for_plot_18[key] = np.array(
            [float("inf"), float("inf"), float("inf")]
        )
    except (
        Exception
    ) as e:  # Catch any other error during CRLB computation for a specific WF
        print(
            f"Warning: Could not compute CRLB for '{key}' due to {e}. Assigning [nan, nan, nan]."
        )
        crlb_data_for_plot_18[key] = np.array(
            [float("nan"), float("nan"), float("nan")]
        )


crlb_array_plot_18 = np.array(
    [crlb_data_for_plot_18[key] for key in all_keys_for_plot_18]
)

nuisance_aware_keys_plot_18 = [
    k for k in all_keys_for_plot_18 if "$J_{\psi|\phi}$" in k
]
multi_obj_original_keys_plot_18 = ["LogLin", "OffDiag", "SubDet"]
single_param_mis_keys_plot_18 = [
    r"Max $\mathcal{J}_x$",
    r"Max $\mathcal{J}_z$",
    r"Max $\mathcal{J}_\theta$",
]

markers_plot_18 = []
colors_plot_18 = []
sizes_plot_18 = []

for key in all_keys_for_plot_18:
    if key in nuisance_aware_keys_plot_18:
        markers_plot_18.append("^")
        colors_plot_18.append("green")
        sizes_plot_18.append(80)
    elif key in multi_obj_original_keys_plot_18:
        markers_plot_18.append("s")
        colors_plot_18.append("red")
        sizes_plot_18.append(80)
    elif key in single_param_mis_keys_plot_18:
        markers_plot_18.append("x")
        colors_plot_18.append("purple")
        sizes_plot_18.append(100)
    elif (
        "Plane Waves" in key or "Random WF" in key
    ):  # Assuming these might be added to dict
        markers_plot_18.append(".")
        colors_plot_18.append("gray")
        sizes_plot_18.append(30)
    else:
        markers_plot_18.append("o")
        colors_plot_18.append("blue")
        sizes_plot_18.append(80)

fig_pareto_labeled, axes_pareto_labeled = plt.subplots(
    1, 2, figsize=(18, 7)
)  # Increased figure size

# Plot 1: (CRLB_x + CRLB_z) vs CRLB_theta
ax1_labeled = axes_pareto_labeled[0]
x_data_ax1 = crlb_array_plot_18[:, 0] + crlb_array_plot_18[:, 1]
y_data_ax1 = crlb_array_plot_18[:, 2]

for i, key in enumerate(all_keys_for_plot_18):
    # Only plot if data is finite
    if np.isfinite(x_data_ax1[i]) and np.isfinite(y_data_ax1[i]):
        ax1_labeled.scatter(
            x_data_ax1[i],
            y_data_ax1[i],
            s=sizes_plot_18[i],
            marker=markers_plot_18[i],
            color=colors_plot_18[i],
            alpha=0.7,
            label=key,  # Can add label here if you want a legend too, or remove for pure annotation
        )
        ax1_labeled.annotate(
            key,
            (x_data_ax1[i], y_data_ax1[i]),
            xytext=(5, 5),  # Offset text slightly
            textcoords="offset points",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", alpha=0.5
            ),  # Add a box for readability
        )

ax1_labeled.set_xlabel("CRLB_x + CRLB_z (Sum of Position Errors)")
ax1_labeled.set_ylabel("CRLB_θ (Rotation Error)")
ax1_labeled.set_title("Pareto: Sum of Position Errors vs. Rotation Error")
ax1_labeled.grid(True, linestyle=":", alpha=0.7)
ax1_labeled.set_xscale("log")
ax1_labeled.set_yscale("log")

# Plot 2: CRLB_x vs CRLB_z
ax2_labeled = axes_pareto_labeled[1]
x_data_ax2 = crlb_array_plot_18[:, 0]
y_data_ax2 = crlb_array_plot_18[:, 1]

for i, key in enumerate(all_keys_for_plot_18):
    if np.isfinite(x_data_ax2[i]) and np.isfinite(y_data_ax2[i]):
        ax2_labeled.scatter(
            x_data_ax2[i],
            y_data_ax2[i],
            s=sizes_plot_18[i],
            marker=markers_plot_18[i],
            color=colors_plot_18[i],
            alpha=0.7,
            label=key,  # Optional: for legend
        )
        ax2_labeled.annotate(
            key,
            (x_data_ax2[i], y_data_ax2[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5),
        )

ax2_labeled.set_xlabel("CRLB_x (x-shift Error)")
ax2_labeled.set_ylabel("CRLB_z (z-shift Error)")
ax2_labeled.set_title("Pareto: x-shift Error vs. z-shift Error")
ax2_labeled.grid(True, linestyle=":", alpha=0.7)
ax2_labeled.set_xscale("log")
ax2_labeled.set_yscale("log")

# Optional: Add a legend for marker types if desired, though direct labels are primary
# Creating a custom legend for marker types:
legend_elements = []
seen_labels_for_legend = set()

marker_map = {
    "Nuisance-Aware ($J_{\psi|\phi}$)": ("^", "green"),
    "Multi-Objective (Original)": ("s", "red"),
    "Single-Param MIS": ("x", "purple"),
    "Other Optimal/Benchmark": ("o", "blue"),
    "Reference (Plane/Random)": (".", "gray"),
}

for key_type, (marker, color) in marker_map.items():
    if key_type not in seen_labels_for_legend:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=key_type,
                markerfacecolor=color,
                markersize=10,
            )
        )
        seen_labels_for_legend.add(key_type)

# fig_pareto_labeled.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.03))
# Decided against a general legend as individual annotations are primary.
# If a legend is desired, it can be added back.

plt.suptitle(
    "Pareto Frontier Comparison with Individual Labels", fontsize=16, y=0.99
)
plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)  # Adjust rect to make space for suptitle
plt.show()
