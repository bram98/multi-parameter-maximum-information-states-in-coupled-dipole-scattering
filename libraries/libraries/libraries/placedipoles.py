# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:29:22 2025

@author: 5914787
"""
import numpy as np
import matplotlib.pyplot as plt
from libraries import CoupledDipoleSimulator
import traceback


def square_of_dipoles(center=(4.0, 0.0), side=1.0):
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


def line_of_dipoles(center=(4.0, 0.0), dist=1.0, number=5):
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


def shift_and_rotate_dipoles_3params(original_positions, dx, dz=None, dtheta=None):
    """
    Convenience function for generated shifted dipole pairs, to be used in
    finite difference.

    original_positions: shape(2,Ns), each col is one dipole
    dx, dz: translations in x,z
    dtheta: rotation in radians about the shape's centroid
    Returns array [(scatters_plus_x, scatters_minus_x), ...] with shape
    (3, 2, 2, Ns)
    """
    if dz is None:
        dz = dx
    if dtheta is None:
        dtheta = dx

    shifted_dipoles = []
    for dx_, dz_, dtheta_ in zip([dx, 0, 0], [0, dz, 0], [0, 0, dtheta]):
        dipoles_p = shift_and_rotate_dipoles(original_positions, dx_, dz_, dtheta_)
        dipoles_m = shift_and_rotate_dipoles(original_positions, -dx_, -dz_, -dtheta_)
        shifted_dipoles.append([dipoles_p, dipoles_m])

    return np.array(shifted_dipoles)


def simulate_dS(sim_kwargs, scatterers_plus, scatterers_minus, angles_deg, delta=1e-3):
    """
    Calculates the partial derivative dS/dtheta of the scattering matrix using
    the CoupledDipoleSimulator class. Performs a finite difference by first
    calculating the S-matrix with the scatterers in the scatterers_plus positions
    and then in the scatterers_minus positions.

    Parameters
    ----------
    sim_kwargs: dict
        Dictionary that can be directly passed to a CoupledDipoleSimulator class
    scatterers_plus, scatters_minus: array of shape (2, Ns)
        Positions of scatterers .
    angles_deg: array of shape (nhilbert,)
        Array of angles (in degrees) that are to be sampled in the scattering
        matrix.
    delta: float (default is 1e-3)
        Stepsize to be used in finite difference

    Returns
    -------
    dS: array of shape (nhilbert, nhilbert)
        Partial derivative of the scattering matrix
    """
    scatterers_plus = np.array(scatterers_plus)
    scatterers_minus = np.array(scatterers_minus)
    try:
        assert scatterers_plus.shape == scatterers_minus.shape
        assert sim_kwargs["Ns"] == scatterers_plus.shape[-1]
    except AssertionError:
        # traceback.print_exc()
        raise ValueError(
            (
                f"Shape of dipoles_plus {scatterers_plus.shape}, dipoles_minus "
                + f'{scatterers_minus.shape} and Ns ({sim_kwargs["Ns"]}) are incompatible'
            )
        )

    sim_minus = CoupledDipoleSimulator(**sim_kwargs)
    sim_minus.scatterers = scatterers_minus
    S_minus, _ = sim_minus.build_scattering_matrix(angles_deg, flux_normalize=True)

    sim_plus = CoupledDipoleSimulator(**sim_kwargs)
    sim_plus.scatterers = scatterers_plus
    S_plus, _ = sim_plus.build_scattering_matrix(angles_deg, flux_normalize=True)

    return (S_plus - S_minus) / (2 * delta)


def simulate_dE(
    sim_kwargs,
    scatterers_plus,
    scatterers_minus,
    kx_array,
    wavefront,
    delta=1e-3,
):
    """
    Calculates the partial derivative dE/dtheta for the field using
    the CoupledDipoleSimulator class. Performs a finite difference by first
    calculating E with the scatterers in the scatterers_plus positions
    and then in the scatterers_minus positions

    Parameters
    ----------
    sim_kwargs: dict
        Dictionary that can be directly passed to a CoupledDipoleSimulator class
    scatterers_plus, scatters_minus: array of shape (2, Ns)
        Positions of scatterers .
    angles_deg: array of shape (nhilbert,)
        Array of angles (in degrees) that are to be sampled in the scattering
        matrix.
    kx_array: array of shape (nhilbert,)
        Array with kx components. The input field is seen as a superposition
        of planewaves, whose kx components are given by this array.
    wavefront: complex array of shape (nhilbert,)
        Complex coefficients for the input wavefront.
    delta: float (default is 1e-3)
        Step to be used in finite difference

    Returns
    -------
    dE: array with shape as specified in sim_kwargs
        Partial derivative of E
    """
    assert len(scatterers_plus) == len(scatterers_minus)
    assert sim_kwargs["Ns"] == len(scatterers_plus)

    # E_minus
    sim_minus = CoupledDipoleSimulator(**sim_kwargs)
    sim_minus.scatterers = scatterers_minus
    sim_minus.set_input_field_kx(kx_array, wavefront)
    E_minus = sim_minus.compute_total_field()

    # E_plus
    sim_plus = CoupledDipoleSimulator(**sim_kwargs)
    sim_plus.scatterers = scatterers_plus
    sim_plus.set_input_field_kx(kx_array, wavefront)
    E_plus = sim_plus.compute_total_field()

    dE = (E_plus - E_minus) / (2 * delta)
    return dE


def simulate_dS_3params(sim_kwargs, angles_deg, base_dipoles, dx, dz=None, dtheta=None):
    """
    Calculates the partial derivative dS/dtheta of the scattering matrix using
    the CoupledDipoleSimulator class. Performs a finite difference by first
    calculating the S-matrix with the scatterers in the scatterers_plus positions
    and then in the scatterers_minus positions.

    Parameters
    ----------
    sim_kwargs: dict
        Dictionary that can be directly passed to a CoupledDipoleSimulator class
    scatterers_plus, scatters_minus: array of shape (2, Ns)
        Positions of scatterers .
    angles_deg: array of shape (nhilbert,)
        Array of angles (in degrees) that are to be sampled in the scattering
        matrix.
    delta: float (default is 1e-3)
        Stepsize to be used in finite difference

    Returns
    -------
    dS: array of shape (nhilbert, nhilbert)
        Partial derivative of the scattering matrix
    """
    shifted_dipoles = shift_and_rotate_dipoles_3params(
        base_dipoles, dx=dx, dz=dz, dtheta=dtheta
    )

    dS = []

    for dipoles_plus, dipoles_minus in shifted_dipoles:
        dS_cur = simulate_dS(sim_kwargs, dipoles_plus, dipoles_minus, angles_deg)
        dS.append(dS_cur)

    return np.array(dS)
