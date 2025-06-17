# scatteringlibs/placedipoles.py

import numpy as np

from scatteringlibs.scatteringsimulation import CoupledDipoleSimulator


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


def _fibonacci_chain(iterations):
    """
    Generates the Fibonacci word using L-system rules with L and S:
      - Axiom: 'L'
      - Production rules:
          'L' -> 'LS'
          'S' -> 'L'
    After n iterations, returns the L-system string (i.e., the Fibonacci word in L/S form).
    """
    axiom = "L"
    rules = {"L": "LS", "S": "L"}

    result = axiom
    for _ in range(iterations):
        next_result = []
        for symbol in result:
            next_result.append(rules.get(symbol, symbol))
        result = "".join(next_result)
    return result


def place_quasi_random_line_of_dipoles(
    center=(4.0, 0.0), small_dist=1.0, number=5, start_index=0
):

    fib_chain = "LSLLSLSLLSLLSLSLLSLSLLSLLSLSLLSLLSLSLLSLSLLSLLSLSLLSLSL"
    fib_chain = fib_chain[start_index : start_index + number]
    print(fib_chain)
    golden_ratio = (1 + np.sqrt(5)) / 2

    y = 0
    pos = np.zeros((2, number))
    for i in range(number):
        pos[0, i] = center[0]
        pos[1, i] = y
        if fib_chain[i] == "L":
            y += golden_ratio * small_dist
        elif fib_chain[i] == "S":
            y += small_dist
    pos[1, :] += center[1] - np.mean(pos[1, :])
    return pos


def shift_and_rotate_dipoles(
    original_positions, dx, dz, dtheta, frozen_dipoles_positions=None
):
    """
    original_positions: shape(2,N) with each column a dipole coordinate (z, x) for movable dipoles.
    dx, dz: translations in x,z
    dtheta: rotation in radians about the movable dipoles' centroid.
    frozen_dipoles_positions: shape(2,M) for fixed dipoles, or None.

    Return the new positions of *all* dipoles, shape(2, N+M).
    If original_positions represent two lines (e.g., from place_double_line_of_dipoles),
    this function assumes the first half are frozen and the second half are movable.
    This interpretation should be handled by the calling function by splitting
    the `base_dipoles` appropriately. This function will shift/rotate `original_positions`.
    """
    out = original_positions.copy()

    # Shift/rotate only the 'original_positions'
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

    shifted_rotated_dipoles = np.vstack((z_rot, x_rot))

    if (
        frozen_dipoles_positions is not None
        and frozen_dipoles_positions.size > 0
    ):
        return np.hstack((frozen_dipoles_positions, shifted_rotated_dipoles))
    else:
        return shifted_rotated_dipoles


def calculate_dS_for_3_parameters(
    sim_kwargs,
    angles_deg,
    movable_dipoles,
    frozen_dipoles=None,
    delta_param=1e-3,
    flux_normalize=True,
):
    """
    Calculates dS/d(param) for three parameters: shiftX, shiftZ, and rotation.
    The shifts/rotation are applied *only* to the `movable_dipoles`.
    `frozen_dipoles` remain fixed.

    Args:
        sim_kwargs (dict): Keyword arguments for CoupledDipoleSimulator.
        angles_deg (np.ndarray): Array of incident/output angles in degrees.
        movable_dipoles (np.ndarray): Positions (2, N_movable) of the dipoles to be perturbed.
        frozen_dipoles (np.ndarray, optional): Positions (2, N_frozen) of fixed dipoles. Defaults to None.
        delta_param (float, optional): Finite difference step. Defaults to 1e-3.
        flux_normalize (bool, optional): Whether to flux-normalize input for S-matrix. Defaults to True.

    Returns:
        np.ndarray: dS array of shape (3, N_out, N_in), where 3 corresponds to dx, dz, dtheta.
    """
    dS_list = []
    param_shifts = [
        (delta_param, 0.0, 0.0),  # dx
        (0.0, delta_param, 0.0),  # dz
        (0.0, 0.0, delta_param),  # dtheta
    ]

    # Ensure sim_kwargs reflects the total number of dipoles
    total_Ns = movable_dipoles.shape[1]
    if frozen_dipoles is not None and frozen_dipoles.size > 0:
        total_Ns += frozen_dipoles.shape[1]

    # Create a temporary sim_kwargs for this function to avoid modifying the original
    current_sim_kwargs = sim_kwargs.copy()
    current_sim_kwargs["Ns"] = total_Ns

    for dx, dz, dtheta in param_shifts:
        # Positions for S_plus
        scatterers_plus = shift_and_rotate_dipoles(
            movable_dipoles,
            dx,
            dz,
            dtheta,
            frozen_dipoles_positions=frozen_dipoles,
        )
        sim_plus = CoupledDipoleSimulator(**current_sim_kwargs)
        sim_plus.scatterers = scatterers_plus
        S_plus, _ = sim_plus.build_scattering_matrix(
            angles_deg, flux_normalize=flux_normalize
        )

        # Positions for S_minus
        scatterers_minus = shift_and_rotate_dipoles(
            movable_dipoles,
            -dx,
            -dz,
            -dtheta,
            frozen_dipoles_positions=frozen_dipoles,
        )
        sim_minus = CoupledDipoleSimulator(**current_sim_kwargs)
        sim_minus.scatterers = scatterers_minus
        S_minus, _ = sim_minus.build_scattering_matrix(
            angles_deg, flux_normalize=flux_normalize
        )

        S_deriv = (S_plus - S_minus) / (2 * delta_param)
        dS_list.append(S_deriv)

    return np.array(dS_list)
