# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:43:43 2025

@author: 5914787
"""
# %%
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
# delta = 1e-3

# dE_dtheta_manual = np.zeros((3, sim.grid_points_x), dtype=complex)


# loss_function = [lf for lf in loss_functions if lf.name == "det"][0]
# wavefront = loss_function.wavefront
# sim.set_input_field_kx(kx_array, wavefront, flux_normalize=True)

# for p in range(3):

#     dtheta_vec = delta * np.eye(3)[p]
#     shifted_dipoles = placedipoles.shift_and_rotate_dipoles(
#         base_dipoles, *dtheta_vec, frozen_dipoles=frozen_dipoles
#     )
#     sim.reset()
#     sim.scatterers = shifted_dipoles
#     E_plus = sim.compute_zL_fields_direct()

#     shifted_dipoles = placedipoles.shift_and_rotate_dipoles(
#         base_dipoles, *(-dtheta_vec), frozen_dipoles=frozen_dipoles
#     )
#     sim.reset()
#     sim.scatterers = shifted_dipoles
#     E_minus = sim.compute_zL_fields_direct()

#     dE_dtheta_manual[p] = (E_plus - E_minus) / (2 * delta)
# %%
"""
Calculate dE/dtheta, E(theta) and E0 = E(0)
"""


def calc_dE_dtheta(dS, wavefront):
    dE_dtheta = [dS[p] @ wavefront for p in range(len(dS))]
    return np.array(dE_dtheta)


def calc_E0(sim, base_dipoles, frozen_dipoles):
    sim.reset()
    sim.scatterers = np.hstack([base_dipoles, frozen_dipoles])
    E0 = sim.compute_zL_fields_direct()
    return E0


def calc_E_range_theta(sim, theta_range, base_dipoles, frozen_dipoles):
    E_output = np.zeros((len(theta_range), 3, sim.grid_points_x), dtype=complex)
    for p in range(3):
        for i, dtheta in enumerate(theta_range):
            dtheta_vec = dtheta * np.eye(3)[p]
            shifted_dipoles = placedipoles.shift_and_rotate_dipoles(
                base_dipoles, *dtheta_vec, frozen_dipoles=frozen_dipoles
            )
            sim.reset()
            sim.scatterers = shifted_dipoles
            E_output[i, p] = sim.compute_zL_fields_direct()
    return E_output


n_theta = 200
dE_dtheta = np.zeros((3, sim.grid_points_x), dtype=complex)
theta_range = 0.1
theta_in = np.linspace(-theta_range, theta_range, num=n_theta)


# loss_function = [lf for lf in loss_functions if lf.name == "det"][0]
loss_function = loss_functions["det"]
wavefront = loss_function.wavefront


sim.set_input_field_kx(kx_array, wavefront, flux_normalize=True)

dE_dtheta = calc_dE_dtheta(fishermatrix.dS, wavefront)
E0 = calc_E0(sim, base_dipoles, frozen_dipoles)
E_output = calc_E_range_theta(sim, theta_in, base_dipoles, frozen_dipoles)


# %%
"""
Plot E(theta)
"""


def plot_E_theta_range(E, theta_range, zmin, zmax):
    for p in range(3):
        plt.figure()
        # ims = plt.imshow(np.abs(E_output[p].T), extent=[np.min(sim.z), np.max(sim.z), np.min(theta_in), np.max(theta_in)])
        ims = plt.imshow(
            np.abs(E[:, p].T),
            # extent=[np.min(theta_in), np.max(theta_in), np.min(sim.z), np.max(sim.z)],
            extent=[np.min(theta_range), np.max(theta_range), zmin, zmax],
        )

        plt.gca().set_aspect("auto")
        divider = make_axes_locatable(plt.gca())
        plt.xlabel(r"$\theta-\theta_0$")
        plt.ylabel(r"$x / \lambda$")
        plt.title(f"Output field for {fishermatrix.param_names[p]}")
        cax = divider.append_axes("right", size="4%", pad=0.1)
        plt.colorbar(ims, cax=cax)
        plt.show()


plot_E_theta_range(E_output, theta_in, np.min(sim.z), np.max(sim.z))
# %%
"""
Plot E0 + dE/dtheta * theta - E(theta), how big is the linear region?
"""
for p in range(3):
    plt.figure()
    ims = plt.imshow(
        np.abs(E0 + dE_dtheta[p] * theta_in[:, np.newaxis] - E_output[:, p, :]).T,
        extent=[np.min(theta_in), np.max(theta_in), np.min(sim.z), np.max(sim.z)],
    )
    # ims = plt.imshow(
    #     # np.abs(theta_in[np.newaxis, :]),
    #     # np.abs(theta_in[np.newaxis, :]),
    #     # np.abs(E0[p, :, np.newaxis]),
    #     # np.abs(dE_dtheta[p, :, np.newaxis]),
    #     extent=[np.min(theta_in), np.max(theta_in), np.min(sim.z), np.max(sim.z)],
    # )

    plt.gca().set_aspect("auto")
    divider = make_axes_locatable(plt.gca())
    plt.xlabel(r"$\theta-\theta_0$")
    plt.ylabel(r"$x / \lambda$")
    plt.title(
        r"$\left|E_{\mathrm{predict}}-E\right|$" + f" for {fishermatrix.param_names[p]}"
    )
    cax = divider.append_axes("right", size="4%", pad=0.1)
    plt.colorbar(ims, cax=cax)
    plt.show()


# %%
"""
Plot best estimator. Use the following linear model:
    E = E0 + d_theta E0 (theta - theta0)
To solve for theta:
    ==> theta - theta0 = d_theta E0^* . (E - E0) /|d_theta E0|^2
"""


def best_estimate_simple(E, dE_dtheta, E0):
    """
    Version with no numpy trickery for debugging and readability. Can only do a
    single parameter or estimate at the same time.

    Parameters
    ----------
    E: numpy array of shape (nout,)
        Array of complex output state.
    dE_theta: numpy array of shape (nout,)
        Partial derivative of E with respect to the parameters. Evaluated at
        theta = theta0.
    E0: numpy array of shape (nout,)
        Array of complex output state evaluated at theta = theta0.

    Returns
    -------
    theta_est: float
        Estimate of theta
    """
    theta_est = (dE_dtheta.conj()) @ (E - E0) / (dE_dtheta.conj() @ dE_dtheta)
    theta_est = np.real(theta_est)
    return theta_est


def best_estimate(E, dE_dtheta, E0):
    """
    Version with numpy trickery for faster computation. Can do multiple
    estimates and parameters at the same time. Here
        ntheta = Number of estimates of theta. Can also be a more complex shape.
        np = Number of parameters
        nout = Size of output states

    Parameters
    ----------
    E: numpy array of shape (ntheta, np, nout)
        Array of complex output state.
    dE_theta: numpy array of shape (np, nout)
        Partial derivative of E with respect to the parameters. Evaluated at
        theta = theta0.
    E0: numpy array of shape (nout,)
        Array of complex output state evaluated at theta = theta0.

    Returns
    -------
    theta_est: numpy array of shape (ntheta, np)
        Array of estimates

    """
    theta_est = np.vecdot(dE_dtheta, E - E0) / np.vecdot(dE_dtheta, dE_dtheta)
    theta_est = np.real(theta_est)
    return theta_est


def best_estimate_multilinear_simple(E, dE_dtheta, E0):
    H = dE_dtheta.T
    Hdag = H.conj().T
    print(f"E {E.shape}")
    print(f"H {H.shape}")
    print(f"Hdag {Hdag.shape}")
    print(f"E0 {E0.shape}")
    theta_est = np.real(np.linalg.inv(Hdag @ H)) @ np.real(Hdag @ (E - E0))
    # theta_est = Hdag @ (E - E0)
    # theta_est = (E - E0)
    print(f"theta_est {theta_est.shape}")
    return theta_est


p = 2
dtheta = best_estimate_simple(E_output[0, p], dE_dtheta[p], E0)
dtheta = best_estimate(E_output, dE_dtheta, E0)
dtheta = best_estimate_multilinear_simple(E_output[0, p], dE_dtheta, E0)

# for p in range(3):
#     i = 0

#     plt.figure()
#     plt.plot(theta_in, dtheta[:, p])
#     theta_exact = theta_in[n_theta // 2 - 30 : n_theta // 2 + 30]
#     plt.plot(theta_exact, theta_exact)
#     param = fishermatrix.param_names[p]
#     plt.xlabel(r"${}$".format(param))
#     plt.ylabel(r"predicted ${}$".format(param))
#     plt.title(f"Best linear estimator for {fishermatrix.param_names[p]}")
#     # plt.ylim(np.min(dtheta), np.max(dtheta))
#     plt.show()


# %%
"""
Now use the statistical nature of our field for low photon counts to get
errorbars on the measurements. I first wrote the poisson camera, but then I
realized that was stupid because we need complex field detection. After some
correspondence with Chat GPT I believe I used the right way to sample, but I
would have to check, because I'm not very confident
"""


def measure_poisson(I, integration_time, nsamples=None):
    """
    Models a camera with shot noise. Here I is an array representing the
    intensity in photons / second at every pixel of the camera. The integration
    time determines for how long photons are collected. Returns an array with
    photon counts.

    Parameters
    ----------
    I: array
        Intensity in units of photons / second.
    integration_time: float
        Integration time in seconds
    nsamples: int, default: None
        Perform multiple measurements if necessary.

    Returns
    -------
        Array of photon counts of type int with shape (nsamples, I.shape) if
        nsamples is given or otherwise with same shape as I
    """
    I = np.array(I)
    # I_flat = I.flatten()
    if nsamples is None:
        photon_counts = stats.poisson.rvs(I * integration_time)
    else:
        photon_counts = stats.poisson.rvs(
            I * integration_time, size=(nsamples, *I.shape)
        )
    return photon_counts


def measure_heterodyne(E, integration_time, nsamples=None):
    """
    https://chatgpt.com/share/681242ae-2044-8004-b701-f3d14a45781b
    Models a camera with complex field detection Here E is an array
    representing the field strength in units of square root of photons per
    coherence time (???). integration_time is the integration time in units of
    coherence lengths ~ 1 / linewidth. Returns an array of field samples.

    Parameters
    ----------
    E: complex array
        Field strength in units of sqrt(photons) / coherence times(???).
    integration_time: float
        Integration time in coherence times.
    nsamples: int, default: None
        Perform multiple measurements if necessary.

    Returns
    -------
        Array of complex fields with shape (nsamples, E.shape) if
        nsamples is given or otherwise E.shape.
    """
    E = np.array(E)
    E_real = E.view(E.real.dtype)  # convert complex array to real array
    if nsamples is None:
        field_samples = stats.norm.rvs(loc=E_real, scale=1 / (2 * integration_time))
    else:
        field_samples = stats.norm.rvs(
            loc=E_real, scale=1 / (2 * integration_time), size=(nsamples, *E_real.shape)
        )
    field_samples = field_samples.view(E.dtype)  # convert back to complex array
    return field_samples


def measure_heterodyne2(E, integration_time, nsamples=None):
    """
    https://chatgpt.com/share/681242ae-2044-8004-b701-f3d14a45781b
    Models a camera with complex field detection Here E is an array
    representing the field strength in units of square root of photons per
    coherence time (???). integration_time is the integration time in units of
    coherence lengths ~ 1 / linewidth. Returns an array of field samples.

    Parameters
    ----------
    E: complex array
        Field strength in units of sqrt(photons) / coherence times(???).
    integration_time: float
        Integration time in coherence times.
    nsamples: int, default: None
        Perform multiple measurements if necessary.

    Returns
    -------
        Array of complex fields with shape (nsamples, E.shape) if
        nsamples is given or otherwise E.shape.
    """
    E = np.array(E)
    E_real = E.view(E.real.dtype)  # convert complex array to real array

    sigma = np.sqrt(1 / (2 * integration_time))
    if nsamples is None:
        field_samples = stats.norm.rvs(loc=E_real, scale=sigma)
    else:
        field_samples = stats.norm.rvs(
            loc=E_real, scale=sigma, size=(nsamples, *E_real.shape)
        )
    field_samples = field_samples.view(E.dtype)  # convert back to complex array
    return field_samples


int_time = 1.0
nsamples = 100
E_output_noise = measure_heterodyne(
    E_output, integration_time=int_time, nsamples=nsamples
)


# plot_E_theta_range(E_output_poisson[0], theta_in, np.min(sim.z), np.max(sim.z))
# dtheta = 0

# dtheta_poisson = np.zeros((nsamples, p, len(E0)), dtype=np.complex128)

# for s in range(nsamples):
#     for p in range(3):
#         i = 0
dtheta_poisson = best_estimate(E_output_noise, dE_dtheta, E0)
dtheta_mean = np.mean(dtheta_poisson, axis=0)
dtheta_var = np.std(dtheta_poisson, axis=0)
for p in range(3):
    plt.figure()
    plt.plot(theta_in, dtheta_mean[:, p], label="Linear Estimator")
    mean = dtheta_mean[:, p]
    var = dtheta_var[:, p]
    plt.fill_between(theta_in, mean - var, mean + var, alpha=0.5)
    theta_exact = theta_in[n_theta // 2 - 30 : n_theta // 2 + 30]
    plt.plot(theta_exact, theta_exact, zorder=500, label="Exact")
    param = fishermatrix.param_names[p]
    plt.xlabel(r"${}$".format(param))
    plt.ylabel(r"predicted ${}$".format(param))
    plt.title(f"Best linear estimator for {fishermatrix.param_names[p]}")
    plt.legend()
    # plt.ylim(np.min(dtheta), np.max(dtheta))
    plt.show()

# %%

plt.figure()
max_var = np.max(var)
plt.plot(theta_in, var)
plt.show()
