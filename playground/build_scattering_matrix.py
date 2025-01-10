# build_scattering_matrix.py
# %%
"""
Example script that builds a scattering matrix T using the CoupledDipoleSimulator,
mapping a discrete set of incident plane-wave angles (input) to the direct
camera line at z=Lz (output). This is a so-called "direct basis" on the output
(not plane-wave or angle channels, but simply sampled field points). The matrix
T thus has shape (N_out, N_in), where N_in is the number of input angles, and
N_out is the number of camera sample points. For many tasks (e.g. maximizing
information, designing wavefronts), such a direct representation is perfectly
valid.

We also do a simple flux normalization on the *input* channels, so each
incident plane wave has roughly unit flux in 2D. On the output we do no further
projection or normalization—just store the field samples at z=Lz.
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from ScatteringSimulation import CoupledDipoleSimulator

# %% Define a helper function for flux-normalizing plane-wave channels in 2D
def flux_amplitude_2d(k0, kx):
    """
    Return amplitude so that the plane wave e^{i kx x + i kz z} in 2D
    has ~ unit flux. The flux for a 2D plane wave is proportional to sqrt(kz),
    where kz = sqrt(k0^2 - kx^2). We also include a / sqrt(N) or / sqrt(\Delta kx)
    factor if we want an orthonormal set, but for a small discrete set we usually
    just do sqrt(kz / k0).
    """
    kz = np.sqrt(k0**2 - kx**2 + 0j)  # +0j to keep it in complex domain
    return np.sqrt(kz / k0)  # kind of equivalent to normalize with sqrt(cos)

# %% Build scattering matrix in the "direct" output basis
def build_scattering_matrix_direct(
    simulator: CoupledDipoleSimulator,
    angles_deg: np.ndarray
):
    """
    Build a scattering matrix T of shape (N_out, N_in) by sending in each angle
    from angles_deg (each as a plane wave), capturing the output E-field at z=Lz
    on the simulator's x-grid. The output basis is thus the direct sampling
    (grid_points_x) at z=Lz. We do flux normalization on the input wave only.

    simulator: a CoupledDipoleSimulator instance with the desired scatterers
               already placed.
    angles_deg: array of angles (in degrees) for which we want an input plane wave.

    Returns:
        T: np.ndarray, shape (N_out, N_in), the scattering matrix
        x_out: array of length N_out (the x sampling at z=Lz),
               so T[i,n] is the field at x_out[i] for input mode n
    """
    # The "camera" array at z=Lz
    x_out = simulator.x.copy()  # shape (grid_points_x,)

    # Decide how many input channels
    N_in = len(angles_deg)
    N_out = x_out.size

    # Prepare an empty scattering matrix
    T = np.zeros((N_out, N_in), dtype=complex)

    # We will loop over each input angle
    k0 = simulator.k  # base wavenumber
    for n, theta_deg in enumerate(angles_deg):
        theta_rad = np.deg2rad(theta_deg)
        kx = k0 * np.sin(theta_rad)

        # flux amplitude so each plane wave is ~ unit flux in 2D
        amp = flux_amplitude_2d(k0, kx)

        # Set the simulator input to this single plane wave (with flux amplitude)
        simulator.set_input_field_kx([kx], [amp])

        # Solve fields
        simulator.solve_fields()

        # Extract field at z=Lz directly
        E_zL = simulator.compute_zL_fields_direct()  # shape: (N_out,)

        # Store in the T matrix
        T[:, n] = E_zL

        # Reset the simulator for the next angle
        simulator.reset()

    return T, x_out

# %% Main usage example
if __name__ == "__main__":
    # 1) Create a CoupledDipoleSimulator with some scatterers
    sim = CoupledDipoleSimulator(
        lambda_0=1.0,
        Lz=80.0,   # a shorter slab for quick testing
        Lx=80.0,
        Ns=1,     # a few random scatterers
        seed=42   # for reproducibility
    )

    # 2) Choose some input angles
    max_angle = 15
    angles = np.linspace(-max_angle, max_angle, 100)

    # 3) Build the scattering matrix in the direct output basis
    T_matrix, x_cam = build_scattering_matrix_direct(sim, angles)

    # T_matrix has shape (N_out=sim.grid_points_x, N_in=len(angles))

    # %% Quick sanity checks and plots

    # -- Check scatterers
    plt.figure()
    plt.scatter(sim.scatterers[0]/sim.lambda_0, sim.scatterers[1]/sim.lambda_0,
                c='r', s=40, label='Scatterers')
    plt.xlabel("z / λ")
    plt.ylabel("x / λ")
    plt.title("Scatterers (z-x plane in 2D)")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -- Plot magnitude of each output column in T_matrix
    if len(angles) < 21:
        fig, axes = plt.subplots(nrows=len(angles), ncols=1, figsize=(7, 2*len(angles)))
        if len(angles) == 1:
            axes = [axes]  # to unify indexing
        for i, ax in enumerate(axes):
            ax.plot(x_cam/sim.lambda_0, np.abs(T_matrix[:, i])**2, 'b-')
            ax.set_title(f"Output intensity for θ_in = {angles[i]:.1f}°")
            ax.set_xlabel("x / λ  (at z = Lz)")
            ax.set_ylabel(r"$|E|^2$")
            ax.grid(True)
        plt.tight_layout()
        plt.show()

    print("Scattering matrix shape:", T_matrix.shape)
    print("Angles (deg):", angles)
    print("x_cam array shape:", x_cam.shape)

    # %% 1) Perform SVD of the scattering matrix T
    # T has shape (N_out, N_in):
    #     T = U @ S @ V.conjugate().T
    # where:
    #   U is (N_out x N_out) unitary
    #   S is (N_out x N_in) diagonal-like (with min(N_out, N_in) singular values on diag)
    #   V is (N_in x N_in) unitary
    U, svals, Vh = np.linalg.svd(T_matrix, full_matrices=False)

    # singular values (descending)
    print("Singular values (descending):", svals)

    # %% 2) Plot the singular values
    plt.figure()
    plt.plot(svals, 'o-', label='Singular Values')
    plt.title("Singular Values of the Scattering Matrix T")
    plt.xlabel("Index")
    plt.ylabel("σ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # %% 3) Inspect the dominant singular vector(s)
    # The "right" singular vectors (columns of V) correspond to input-mode coefficients
    # in angle space (each vector is length N_in).
    # The "left" singular vectors (columns of U) correspond to output fields in camera space.

    # a) Let's pick the strongest mode = the 1st singular vector
    mode_index = 0  # 0 => first (largest) singular value
    input_vector = Vh.conjugate().T[:, mode_index]  # shape (N_in,)
    output_vector = U[:, mode_index]                 # shape (N_out,)

    print(f"\n--- Dominant singular mode (index=0) ---")
    print("Input-vector (angle-space) = ", input_vector)
    print("Output-vector (camera-space) = ", output_vector)

    # b) Plot the input coefficients vs. angle
    plt.figure(figsize=(6,4))
    plt.plot(angles, np.abs(input_vector)**2, 'bo-', label="|input|^2")
    plt.title("Dominant Input Sing. Vector (Angle Space)")
    plt.xlabel("Angle (deg)")
    plt.ylabel("|Coefficient|^2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # c) Plot the corresponding output field pattern vs. x at z=Lz
    plt.figure(figsize=(8,4))
    plt.plot(x_cam/sim.lambda_0, np.abs(output_vector)**2, 'r-')
    plt.title("Dominant Output Sing. Vector (Camera Space)")
    plt.xlabel("x / λ  (at z=Lz)")
    plt.ylabel("|Field|^2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # b) Plot the input coefficients vs. angle
    plt.figure(figsize=(6,4))
    plt.plot(angles, np.angle(input_vector), 'bo-', label="|input|^2")
    plt.title("Dominant Input Sing. Vector (Angle Space)")
    plt.xlabel("Angle (deg)")
    plt.ylabel("angle(Coefficient)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # c) Plot the corresponding output field pattern vs. x at z=Lz
    plt.figure(figsize=(8,4))
    plt.plot(x_cam/sim.lambda_0, np.angle(output_vector), 'r-')
    plt.title("Dominant Output Sing. Vector (Camera Space)")
    plt.xlabel("x / λ  (at z=Lz)")
    plt.ylabel("angle(Coefficient)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # %% 4) (Optional) Double-check by simulating that singular input on the actual simulator
    # If you want to see if feeding that linear combination of angles physically
    # reproduces the output_vector, you can do:


    # We re-init or reuse a simulator with same geometry + scatterers
    test_sim = CoupledDipoleSimulator(
        lambda_0=sim.lambda_0,
        Lz=sim.Lz/sim.lambda_0,  # pass in wavelengths
        Lx=sim.Lx/sim.lambda_0,
        Ns=sim.Ns,
        seed=sim.seed
    )

    # "input_vector" are the amplitude coefficients for each angle in 'angles'
    # We convert them to kx, multiply by amplitude[i], etc.
    kx_array = test_sim.k * np.sin(np.deg2rad(angles))
    test_sim.set_input_field_kx(kx_array, input_vector)
    test_sim.solve_fields()
    E_test = test_sim.compute_zL_fields_direct()  # shape (N_out,)

    # Compare E_test to svals[mode_index]*output_vector (up to a global phase)
    # Because physically T * input_vector = svals[0] * output_vector
    predicted = svals[mode_index] * output_vector

    # Quick plot to see if they match in magnitude
    plt.figure(figsize=(8,4))
    plt.plot(x_cam/sim.lambda_0, np.abs(E_test), 'b-', label="|E_test| from sim")
    plt.plot(x_cam/sim.lambda_0, np.abs(predicted), 'r--', label="|predicted| from SVD")
    plt.title("Check: Summed Input Coeff => output (compare to SVD prediction)")
    plt.xlabel("x / λ  (at z=Lz)")
    plt.ylabel("|E|")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # show whole intensity distribution in slab
    test_sim.visualize_intensity()

    # %% 5) Test with a random input field
    # Generate a random input field
    random_input = np.random.rand(len(angles)) + 1j * 2* np.pi * np.random.rand(len(angles))

    # Predict output using the scattering matrix
    predicted_output_random = T_matrix @ random_input  # matrix-vector multiplication

    # Simulate output for the same random input
    test_sim = CoupledDipoleSimulator(
        lambda_0=sim.lambda_0,
        Lz=sim.Lz / sim.lambda_0,
        Lx=sim.Lx / sim.lambda_0,
        Ns=sim.Ns,
        seed=sim.seed
    )

    kx_array = test_sim.k * np.sin(np.deg2rad(angles))
    test_sim.set_input_field_kx(kx_array, random_input)
    test_sim.solve_fields()
    simulated_output_random = test_sim.compute_zL_fields_direct()

    # Compare predicted and simulated outputs
    plt.figure(figsize=(8, 4))
    plt.plot(x_cam / sim.lambda_0, np.abs(simulated_output_random), 'b-', label="|E_simulated|")
    plt.plot(x_cam / sim.lambda_0, np.abs(predicted_output_random), 'r--', label="|E_predicted|")
    plt.title("Random Input: Simulated vs Predicted Output")
    plt.xlabel("x / λ  (at z = Lz)")
    plt.ylabel("|E|")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # show whole intensity distribution in slab
    test_sim.visualize_intensity()