#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scattering Simulation

A class to simulate wave scattering in a 2D slab using the Coupled Dipole Method (CDM).
It handles multiple scatterers, computes induced electric fields, and visualizes intensity distributions.

Authors: Jacob Seifert - j.seifert@uu.nl, Bram Verreussel
"""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import hankel1

from scatteringlibs.storable import Storable


def _flux_amplitude_2d(k0, kx):
    """
    Optional helper that returns amplitude so that e^{i(kx*x + kz*z)} in 2D
    has ~ unit flux. k0 = 2π/λ, kx in [-k0, k0], and
    kz = sqrt(k0^2 - kx^2).
    """
    kz = np.sqrt(k0**2 - kx**2 + 0j)
    return np.sqrt((kz / k0).real)


class CoupledDipoleSimulator(Storable):
    """Simulates wave scattering in a 2D slab using the Coupled Dipole Method (CDM).

    Attributes:
        lambda_0 (float): Wavelength.
        k (float): Wavenumber.
        Lz (float): Length of the slab along the z-axis in wavelengths.
        Lx (float): Length of the slab along the x-axis in wavelengths.
        Ns (int): Number of scatterers.
        alpha (complex): Polarizability of scatterers.
        exclusion_radius (float): Minimum spacing between scatterers.
        grid_points_x (int): Number of grid points along the x-axis.
        grid_points_z (int): Number of grid points along the z-axis (derived).
        z (numpy.ndarray): Array of z-coordinates.
        x (numpy.ndarray): Array of x-coordinates.
        Z_grid (numpy.ndarray): Meshgrid for z-coordinates.
        X_grid (numpy.ndarray): Meshgrid for x-coordinates.
        scatterers (numpy.ndarray): Positions of scatterers (shape: (2, Ns)).
        fields (numpy.ndarray): Induced electric fields at scatterers.
        E_total (numpy.ndarray): Total electric field on the grid.
        E_output (numpy.ndarray): Electric field at z = Lz plane.
        theta_inc_deg (float): Incident angle in degrees.
        theta_inc (float): Incident angle in radians.
        seed (int): Random seed for scatterer placement.
        k_x_array (numpy.ndarray): Array of k_x components for arbitrary input field.
        E_kx_array (numpy.ndarray): Corresponding amplitudes for k_x components.
    """

    def __init__(
        self,
        lambda_0=1.0,
        Lz=16.0,
        Lx=8.0,
        Ns=10,
        alpha=4j / (2 * np.pi / 1.0) ** 2,
        exclusion_radius=0.637,  # Derived from sigma_s ≈ 0.637λ₀
        grid_points_x=400,
        theta_inc_deg=None,
        theta_inc=None,
        seed=0,
        k_x_array=None,
        E_kx_array=None,
    ):
        """Initializes the CoupledDipoleSimulator with default or user-defined parameters.

        Args:
            lambda_0 (float, optional): Wavelength (normalized units). Defaults to 1.0.
            Lz (float, optional): Length of the slab along the z-axis (in wavelengths). Defaults to 16.0.
            Lx (float, optional): Length of the slab along the x-axis (in wavelengths). Defaults to 8.0.
            Ns (int, optional): Number of scatterers. Defaults to 10.
            alpha (complex, optional): Polarizability of scatterers. Defaults to 4j / k**2.
            exclusion_radius (float, optional): Minimum spacing between scatterers (in wavelengths). Defaults to 0.637.
            grid_points_x (int, optional): Number of grid points along the x-axis. Defaults to 400.
            theta_inc_deg (float, optional): Incident angle in degrees. Defaults to 0.0.
            theta_inc (float, optional): Incident angle in radians. Defaults to 0.0.
            seed (int, optional): Random seed for scatterer placement. Defaults to 0.
            k_x_array (numpy.ndarray, optional): Array of k_x components for arbitrary input field. Defaults to None.
            E_kx_array (numpy.ndarray, optional): Corresponding amplitudes for k_x components. Defaults to None.
        """
        self.lambda_0 = lambda_0
        self.k = 2 * np.pi / self.lambda_0
        self.Lz = Lz * self.lambda_0
        self.Lx = Lx * self.lambda_0
        self.Ns = Ns
        self.alpha = alpha
        self.exclusion_radius = exclusion_radius * self.lambda_0
        self.grid_points_x = grid_points_x
        if theta_inc_deg is not None:
            self.theta_inc_deg = theta_inc_deg
        elif theta_inc is not None:
            self.theta_inc = theta_inc
        else:
            self.theta_inc = 0.0
        self.seed = seed

        # Derive grid points along z based on x sampling
        # Note: for endpoint=True, you need to subtract one from grid_points
        delta_x = self.Lx / (self.grid_points_x - 1)
        self.grid_points_z = int(np.ceil(self.Lz / delta_x))

        # Generate grid
        self.z = np.linspace(0, self.Lz, self.grid_points_z, endpoint=True)
        self.x = np.linspace(
            -self.Lx / 2, self.Lx / 2, self.grid_points_x, endpoint=True
        )
        self.Z_grid, self.X_grid = np.meshgrid(self.z, self.x)

        # Initialize scatterer positions
        self.scatterers = self._generate_scatterers()

        # Initialize fields as None; compute when needed
        self.fields = None
        self.E_total = None
        self.E_output = None

        # For arbitrary input fields
        self.k_x_array = k_x_array
        self.E_kx_array = E_kx_array

    def _generate_scatterers(self):
        """Generates scatterer positions with the first at the center and others random.

        Returns:
            numpy.ndarray: Array of scatterer positions (shape: (2, Ns)).
        """
        if self.Ns == 0:
            return np.array([[], []])

        scatterers = [
            np.array([self.Lz / 2, 0])
        ]  # First scatterer always centered
        np.random.seed(self.seed)  # For reproducibility
        while len(scatterers) < self.Ns:
            candidate = np.array(
                [
                    np.random.uniform(0, self.Lz),
                    np.random.uniform(-self.Lx / 2, self.Lx / 2),
                ]
            )
            if all(
                np.linalg.norm(candidate - s) > self.exclusion_radius
                for s in scatterers
            ):
                scatterers.append(candidate)

        return np.array(scatterers).T  # Shape: (2, Ns)

    @staticmethod
    def _G0(r, r_prime, k):
        """Computes the Green's function for 2D wave propagation.

        Args:
            r (numpy.ndarray): Observation points (shape: (2, N)).
            r_prime (numpy.ndarray): Source points (shape: (2, M)).
            k (float): Wavenumber.

        Returns:
            numpy.ndarray: Green's function matrix (shape: (N, M)).
        """
        # Compute pairwise distances using cdist
        distances = cdist(r.T, r_prime.T)  # Shape: (N, M)
        G = (1j / 4) * hankel1(0, k * distances)
        G[distances == 0] = 0.0  # Handle singularity
        return G

    def theta2k_x(self, theta, degree=False):
        if degree:
            theta = np.deg2rad(theta)
        k_x = self.k * np.sin(theta)
        return k_x

    def k_x2theta(self, k_x, degree=False):
        theta = np.arcsin(k_x / self.k)
        if degree:
            theta = np.rad2deg(theta)
        return theta

    def compute_kz(self, k_x_array):
        k_z_array = np.sqrt(self.k**2 - k_x_array**2, dtype=complex)
        return k_z_array

    def set_input_field_kx(self, k_x_array, E_kx_array, flux_normalize=True):
        k_x_array = np.array(k_x_array)
        E_kx_array = np.array(E_kx_array)

        if k_x_array.shape != E_kx_array.shape:
            raise ValueError(
                "k_x_array and E_kx_array must have the same shape."
            )

        if flux_normalize:
            amp = _flux_amplitude_2d(self.k, k_x_array)
        else:
            amp = 1.0

        self.k_x_array = k_x_array
        self.E_kx_array = E_kx_array * amp
        self.reset()  # Reset computed fields

    def compute_incident_field(self, x, z):
        """Computes the incident field at given positions.

        Args:
            x (numpy.ndarray): x-coordinates.
            z (numpy.ndarray): z-coordinates.

        Returns:
            numpy.ndarray: Incident field at the given positions.
        """
        if self.k_x_array is not None and self.E_kx_array is not None:
            # Arbitrary input field specified in k_x components
            E_incident = np.zeros_like(x, dtype=complex)
            k_z_array = self.compute_kz(self.k_x_array)
            for k_x, k_z, E_kx in zip(
                self.k_x_array, k_z_array, self.E_kx_array
            ):
                E_incident += E_kx * np.exp(1j * (k_z * z + k_x * x))
            return E_incident
        else:
            # Default to plane wave
            k_z = self.k * np.cos(self.theta_inc)
            k_x = self.k * np.sin(self.theta_inc)
            return np.exp(1j * (k_z * z + k_x * x))

    def solve_fields(self):
        """Solves the coupled dipole equations to find induced electric fields.

        Returns:
            numpy.ndarray: Induced electric fields at scatterers (shape: (Ns,)).
        """
        if self.fields is not None:
            return self.fields

        if self.Ns == 0:
            self.fields = np.array([])
            return self.fields

        # Incident field at scatterers
        E0_vec = self.compute_incident_field(
            self.scatterers[1, :], self.scatterers[0, :]
        )

        # Interaction matrix
        A = np.eye(self.Ns, dtype=complex)
        G_matrix = self._G0(self.scatterers, self.scatterers, self.k)
        A -= self.k**2 * self.alpha * G_matrix

        # Solve for induced fields
        self.fields = np.linalg.solve(A, E0_vec)
        return self.fields

    def compute_total_field(self):
        """Computes the total electric field on the grid by summing incident and scattered fields.

        Returns:
            numpy.ndarray: Total electric field on the grid (shape: (grid_points_x, grid_points_z)).
        """
        if self.E_total is not None:
            return self.E_total

        # Ensure that induced fields are computed
        if self.fields is None:
            self.solve_fields()

        # Compute incident field
        E_incident = self.compute_incident_field(self.X_grid, self.Z_grid)

        # Initialize total field with incident field
        E_total = np.copy(E_incident)

        # Compute scattered fields from each scatterer
        for n in range(self.Ns):
            r_prime = self.scatterers[:, n].reshape(
                2, 1, 1
            )  # Shape: (2, 1, 1)
            delta_r = np.sqrt(
                (self.Z_grid - r_prime[0]) ** 2
                + (self.X_grid - r_prime[1]) ** 2
            )
            G = (1j / 4) * hankel1(0, self.k * delta_r)
            G[delta_r == 0] = 0.0  # Avoid singularity
            E_total += self.k**2 * self.alpha * self.fields[n] * G

        self.E_total = E_total
        return self.E_total

    def compute_zL_fields_direct(self):
        """Directly computes the electric field at z = Lz plane without solving the entire slab.

        Returns:
            numpy.ndarray: Electric field at z = Lz (shape: (grid_points_x,)).
        """
        if self.E_output is not None:
            return self.E_output

        # Ensure that induced fields are computed
        if self.fields is None:
            self.solve_fields()

        # Define z = Lz
        z_L = self.Lz

        # Define x coordinates along z = Lz
        x_points = self.x  # Shape: (grid_points_x,)

        # Compute incident field at z = Lz for all x
        E_incident = self.compute_incident_field(x_points, z_L)

        # Initialize total field at z = Lz with incident field
        E_output = np.copy(E_incident)

        # Compute scattered fields from each scatterer
        for n in range(self.Ns):
            # Position of scatterer n
            r_n = self.scatterers[:, n]  # Shape: (2,)

            # Observation points at z = Lz, all x
            # Vectorized computation
            delta_z = z_L - r_n[0]  # Scalar
            delta_x = x_points - r_n[1]  # Shape: (grid_points_x,)
            distances = np.sqrt(
                delta_z**2 + delta_x**2
            )  # Shape: (grid_points_x,)

            # Compute Green's function
            G = (1j / 4) * hankel1(0, self.k * distances)
            G[distances == 0] = 0.0  # Avoid singularity

            # Scattered field contribution from scatterer n
            E_scatt = (
                self.k**2 * self.alpha * self.fields[n] * G
            )  # Shape: (grid_points_x,)

            # Add to total field
            E_output += E_scatt

        self.E_output = E_output
        return self.E_output

    def reset(self):
        """Resets computed fields."""
        self.fields = None
        self.E_output = None
        self.E_total = None

    def visualize_scatterers(self):
        """Visualizes the positions of scatterers within the slab."""
        plt.figure(figsize=(12, 12 * self.Lx / self.Lz))
        plt.scatter(
            self.scatterers[0, :] / self.lambda_0,
            self.scatterers[1, :] / self.lambda_0,
            c="red",
            s=100,
            label="Scatterers",
        )
        plt.title("Scatterer Positions")
        plt.xlabel(r"$z/\lambda$")
        plt.ylabel(r"$x/\lambda$")
        plt.legend()
        plt.gca().set_aspect("equal")
        plt.xlim(0, self.Lz / self.lambda_0)
        plt.ylim(-self.Lx / self.lambda_0 / 2, self.Lx / self.lambda_0 / 2)
        plt.tight_layout()
        plt.show()

    def _visualize(self, display_function, title, dpi=100, figsize=(10, 8)):
        """
        Helper function that applies `display_function` to the entire field
        before plotting it.
        """
        # Ensure that E_total is computed
        if self.E_total is None:
            self.compute_total_field()

        plt.figure(
            figsize=(figsize[0], figsize[1] * self.Lx / self.Lz), dpi=dpi
        )
        plt.pcolormesh(
            self.Z_grid / self.lambda_0,
            self.X_grid / self.lambda_0,
            display_function(self.E_total),
            shading="auto",
            cmap="viridis",
        )
        plt.colorbar(label=r"$|E|^2$  (a.u.)")
        plt.scatter(
            self.scatterers[0, :] / self.lambda_0,
            self.scatterers[1, :] / self.lambda_0,
            c="red",
            s=100,
            label="Scatterers",
        )
        plt.title(title)
        plt.xlabel(r"$z/\lambda$")
        plt.ylabel(r"$x/\lambda$")
        plt.legend()
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        plt.show()

    def visualize_intensity(self, title="Custom Input State", dpi=100.0):
        """Visualizes the intensity distribution over the entire slab."""

        if self.k_x_array is not None and self.E_kx_array is not None:
            pass
        else:
            title = f"Electric Field Intensity Over Slab (θ = {self.theta_inc_deg:.1f}°)"

        self._visualize(
            display_function=lambda field: np.abs(field) ** 2,
            title=title,
            dpi=dpi,
        )

    def visualize_abs(self):
        """Visualizes the absolute value of the field over the entire slab."""

        if self.k_x_array is not None and self.E_kx_array is not None:
            title = "Custom Input State: Absolute Electric Field"
        else:
            title = f"Absolute Value of Electric Field Over Slab (θ = {self.theta_inc_deg:.1f}°)"

        self._visualize(
            display_function=np.abs,
            title=title,
        )

    def visualize_real(self):
        """Visualizes the real value of the field over the entire slab."""

        if self.k_x_array is not None and self.E_kx_array is not None:
            title = "Custom Input State: Real Part of Electric Field"
        else:
            title = f"Real Value of Electric Field Over Slab (θ = {self.theta_inc_deg:.1f}°)"

        self._visualize(
            display_function=np.real,
            title=title,
        )

    def visualize_output_field(
        self, plot_intensity_only=False, direct_computation=True
    ):
        """Visualizes the electric field at z = Lz plane.

        Args:
            plot_intensity_only (bool, optional): If True, plots only the intensity.
                                                 If False, plots both intensity and phase. Defaults to True.
            direct_computation (bool, optional): If True, computes E_zL directly without full field. Defaults to True.
        """

        if (self.E_output is not None) and (self.E_total is not None):
            if direct_computation:
                E_output = self.compute_zL_fields_direct()
            else:
                # Extract fields at z = Lz
                E_output = self.get_zL_fields()
        else:
            E_output = self.get_zL_fields()

        if not plot_intensity_only:
            # Plot both intensity and phase
            x_values = self.x / self.lambda_0
            plt.figure(figsize=(12, 6))

            # Intensity plot
            plt.subplot(1, 2, 1)
            plt.plot(
                x_values, np.abs(E_output) ** 2, "b-", label=r"$|E(z=L_z)|^2$"
            )
            plt.title("Electric Field Intensity at $z = L_z$")
            plt.xlabel(r"$x/\lambda$")
            plt.ylabel(r"$|E|^2$")
            plt.grid(True)
            plt.legend()

            # Phase plot
            plt.subplot(1, 2, 2)
            plt.plot(
                x_values, np.angle(E_output), "r-", label=r"$\arg(E(z=L_z))$"
            )
            plt.title("Electric Field Phase at $z = L_z$")
            plt.xlabel(r"$x/\lambda$")
            plt.ylabel("Phase (radians)")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.show()
        else:
            # Plot only intensity
            x_values = self.x / self.lambda_0
            plt.figure(figsize=(12, 6))
            plt.plot(
                x_values, np.abs(E_output) ** 2, "g-", label=r"$|E(z=L_z)|^2$"
            )
            if self.k_x_array is not None and self.E_kx_array is not None:
                plt.title(
                    "Custom Input State: Electric Field Intensity at $z = L_z$"
                )
            else:
                plt.title("Electric Field Intensity at $z = L_z$")
            plt.xlabel(r"$x/\lambda$")
            plt.ylabel(r"$|E|^2$")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def get_zL_fields(self):
        """Returns the electric field at z = Lz plane.

        Returns:
            numpy.ndarray: Electric field at z = Lz (shape: (grid_points_x,)).
        """
        if self.E_output is not None:
            return self.E_output

        # Ensure that E_total is computed
        if self.E_total is None:
            self.compute_total_field()

        self.E_output = self.E_total[:, -1]
        return self.E_output

    @property
    def theta_inc_deg(self):
        return np.rad2deg(self.theta_inc)

    @theta_inc_deg.setter
    def theta_inc_deg(self, deg):
        self.theta_inc = np.deg2rad(deg)

    def animate_time_evolution(
        self,
        frames=60,
        fps=30,
        figsize=(8, 6),
        save_mp4=True,
        save_gif=True,
        mp4_filename="figures/time_evolution.mp4",
        gif_filename="figures/time_evolution.gif",
        blit=False,
        title=None,
        **plt_kwargs,
    ):
        if self.E_total is None:
            self.compute_total_field()

        fig, ax = plt.subplots(figsize=figsize)
        # Use the full 2D field over the slab
        pcm = ax.pcolormesh(
            self.Z_grid / self.lambda_0,
            self.X_grid / self.lambda_0,
            self.E_total.real,
            shading="auto",
            cmap="RdBu_r",
            **plt_kwargs,
        )
        ax.set_xlabel("z / λ")
        ax.set_ylabel("x / λ")
        if title is None:
            title = "Real Part of E over Full Slab"
        ax.set_title(title)
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Re(E)")
        # Plot scatterer positions (scatterers array: [z, x])
        ax.scatter(
            self.scatterers[0] / self.lambda_0,
            self.scatterers[1] / self.lambda_0,
            c="red",
            s=40,
            label="Scatterers",
        )
        ax.legend()

        # Define the angular frequency for one optical period (normalized units)
        c = 1.0
        omega = 2 * np.pi * c / self.lambda_0

        def update(frame):
            t = frame / frames * (2 * np.pi / omega)
            phase_factor = np.exp(-1j * omega * t)
            new_field = (self.E_total * phase_factor).real
            pcm.set_array(new_field.ravel())
            return (pcm,)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=1000 / fps,
            blit=blit,
            repeat=True,
        )

        # Ensure output directories exist before saving
        os.makedirs(os.path.dirname(mp4_filename), exist_ok=True)
        os.makedirs(os.path.dirname(gif_filename), exist_ok=True)

        if save_mp4:
            ani.save(mp4_filename, writer="ffmpeg", fps=fps)
        if save_gif:
            ani.save(gif_filename, writer="imagemagick", fps=fps)

        plt.show()

    def build_scattering_matrix(self, angles_deg, flux_normalize=True):
        """
        Build a scattering matrix S of shape (N_out, N_in) where:
        - N_in = len(angles_deg)
        - N_out = self.grid_points_x (the direct samples at z=Lz)
        For each angle in angles_deg, we set a single plane wave input,
        solve, read off E_zL. Then store in S.
        Returns:
            S: shape (N_out, N_in)
            x_cam: array of x-values at z=Lz
        """
        N_in = len(angles_deg)
        N_out = self.grid_points_x
        S = np.zeros((N_out, N_in), dtype=complex)

        # We'll do a loop over angles
        for n, th_deg in enumerate(angles_deg):
            th_rad = np.deg2rad(th_deg)
            kx = self.k * np.sin(th_rad)
            # if flux_normalize:
            #     amp = _flux_amplitude_2d(self.k, kx)
            # else:
            #     amp = 1.0

            # set input
            self.set_input_field_kx([kx], [1.0], flux_normalize=flux_normalize)
            self.solve_fields()
            E_zL = self.compute_zL_fields_direct()
            S[:, n] = E_zL
        # re-initialize if needed
        return S, self.x


if __name__ == "__main__":
    import time  # Import the time module for performance measurement

    # Start timer
    start_time = time.time()

    # Initialize the simulator with a specific incident angle
    simulator = CoupledDipoleSimulator(theta_inc_deg=45, Ns=7)

    # Display scatterer positions using the new method
    simulator.visualize_scatterers()

    # Visualize the intensity over the full slab
    simulator.visualize_intensity()

    # Visualize the fields at z = Lz with cross-section plots
    simulator.visualize_output_field(direct_computation=True)

    # Optionally, retrieve and print the electric field at z = Lz
    E_zL = simulator.compute_zL_fields_direct()

    # %% ------------------------ Arbitrary Input Field Demo

    # Define arbitrary input field in terms of k_x components
    angles_deg = np.array([-30, 0, 30])  # Incident angles in degrees
    angles_rad = np.deg2rad(angles_deg)  # Convert to radians
    k_x_array = simulator.k * np.sin(angles_rad)  # k_x values
    E_kx_array = np.array([1.0, 2, 1.0])  # Corresponding amplitudes

    # Initialize a new simulator instance for arbitrary input field with scatterers
    simulator_arbitrary = CoupledDipoleSimulator(Ns=5)

    # Set the arbitrary input field
    simulator_arbitrary.set_input_field_kx(k_x_array, E_kx_array)

    # Visualize scatterer positions
    simulator_arbitrary.visualize_scatterers()

    # Visualize the intensity over the full slab with arbitrary input field
    simulator_arbitrary.visualize_intensity()

    # Visualize the fields at z = Lz with cross-section plots
    simulator_arbitrary.visualize_output_field(direct_computation=True)

    # Optionally, retrieve and print the electric field at z = Lz
    E_zL_arbitrary = simulator_arbitrary.compute_zL_fields_direct()

    # ------------------------ Arbitrary Input Field Without Scatterers Demo
    # Initialize a new simulator instance for arbitrary input field with no scatterers
    simulator_arbitrary_no_scatterers = CoupledDipoleSimulator(Ns=0)

    # Set the same arbitrary input field
    simulator_arbitrary_no_scatterers.set_input_field_kx(k_x_array, E_kx_array)

    # Visualize the intensity over the full slab with arbitrary input field and no scatterers
    simulator_arbitrary_no_scatterers.visualize_intensity()

    # Visualize the fields at z = Lz with cross-section plots
    simulator_arbitrary_no_scatterers.visualize_output_field(
        direct_computation=True
    )

    # Optionally, retrieve and print the electric field at z = Lz
    E_zL_arbitrary_no_scatterers = (
        simulator_arbitrary_no_scatterers.compute_zL_fields_direct()
    )
    print(
        f"Electric field at z = Lz with arbitrary input and no scatterers (first 10 points): {E_zL_arbitrary_no_scatterers[:10]}"
    )

    # Stop timer and print elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nTotal Execution time: {elapsed_time:.3f} seconds")

    # %% Test animation method for time evolution (sometimes nice to look at)

    # Create a new simulator with a custom scatterer configuration
    sim = CoupledDipoleSimulator(
        lambda_0=1.0,
        Lz=8.0,  # slab thickness in wavelengths
        Lx=8.0,  # slab width in wavelengths
        Ns=20,  # number of scatterers
        seed=123,  # random seed for reproducibility
        grid_points_x=400,
        theta_inc_deg=0.0,  # incident plane wave at 20 degrees
    )

    # Solve for the induced fields
    sim.solve_fields()

    # Animate the time evolution of the real part of the field at the mid-slice (z-index defaults to mid-plane)

    sim.animate_time_evolution(
        frames=60,
        fps=30,
        figsize=(8, 6),
        save_mp4=True,
        save_gif=True,
        mp4_filename="figures/my_time_evolution.mp4",
        gif_filename="figures/my_time_evolution.gif",
        blit=False,
    )

    print("Time evolution animations have been saved to the figures/ folder.")
