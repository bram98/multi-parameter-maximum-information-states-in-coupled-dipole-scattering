#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scattering Simulation

A class to simulate wave scattering in a 2D slab using the Coupled Dipole Method (CDM).
It handles multiple scatterers, computes induced electric fields, and visualizes intensity distributions.

Authors: Jacob Seifert - j.seifert@uu.nl, Bram Verreussel
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import hankel1


class CoupledDipoleSimulator:
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
        theta_inc_deg=0.0,
        seed=0,
    ):
        """Initializes the CoupledDipoleSimulator with default or user-defined parameters.

        Args:
            lambda_0 (float, optional): Wavelength (normalized units). Defaults to 1.0.
            Lz (float, optional): Length of the slab along the z-axis (in wavelengths). Defaults to 16.0.
            Lx (float, optional): Length of the slab along the x-axis (in wavelengths). Defaults to 8.0.
            Ns (int, optional): Number of scatterers. Defaults to 10.
            alpha (complex, optional): Polarizability of scatterers. Defaults to 4j / k**2.
            exclusion_radius (float, optional): Minimum spacing between scatterers (in wavelengths). Defaults to 0.5.
            grid_points_x (int, optional): Number of grid points along the x-axis. Defaults to 400.
            theta_inc_deg (float, optional): Incident angle in degrees. Defaults to 0.0.
            seed (int, optional): Random seed for scatterer placement. Defaults to 0.
        """

        self.lambda_0 = lambda_0
        self.k = 2 * np.pi / self.lambda_0
        self.Lz = Lz * self.lambda_0
        self.Lx = Lx * self.lambda_0
        self.Ns = Ns
        self.alpha = alpha
        self.exclusion_radius = exclusion_radius * self.lambda_0
        self.grid_points_x = grid_points_x
        self.theta_inc_deg = theta_inc_deg
        self.theta_inc = np.deg2rad(self.theta_inc_deg)
        self.seed = seed

        # Derive grid points along z based on x sampling
        delta_x = self.Lx / self.grid_points_x
        self.grid_points_z = int(np.ceil(self.Lz / delta_x))

        # Generate grid
        self.z = np.linspace(0, self.Lz, self.grid_points_z, endpoint=True)
        self.x = np.linspace(0, self.Lx, self.grid_points_x, endpoint=True)
        self.Z_grid, self.X_grid = np.meshgrid(self.z, self.x)

        # Initialize scatterer positions
        self.scatterers = self._generate_scatterers()

        # Initialize fields as None; compute when needed
        self.fields = None
        self.E_total = None
        self.E_output = None

    def _generate_scatterers(self):
        """Generates scatterer positions with the first at the center and others random.

        Returns:
            numpy.ndarray: Array of scatterer positions (shape: (2, Ns)).
        """
        scatterers = [
            np.array([self.Lz / 2, self.Lx / 2])
        ]  # First scatterer always centered
        np.random.seed(self.seed)  # For reproducibility
        while len(scatterers) < self.Ns:
            candidate = np.array(
                [np.random.uniform(0, self.Lz), np.random.uniform(0, self.Lx)]
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

    def solve_fields(self):
        """Solves the coupled dipole equations to find induced electric fields.

        Returns:
            numpy.ndarray: Induced electric fields at scatterers (shape: (Ns,)).
        """
        if self.fields is not None:
            return self.fields

        # Compute wavevector components
        k_z = self.k * np.cos(self.theta_inc)
        k_x = self.k * np.sin(self.theta_inc)

        # Incident field at scatterers
        E0_vec = np.exp(
            1j * (k_z * self.scatterers[0, :] + k_x * self.scatterers[1, :])
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

        # Compute wavevector components
        k_z = self.k * np.cos(self.theta_inc)
        k_x = self.k * np.sin(self.theta_inc)

        # Compute incident field
        E_incident = np.exp(1j * (k_z * self.Z_grid + k_x * self.X_grid))

        # Initialize total field with incident field
        E_total = np.copy(E_incident)

        # Compute scattered fields from each scatterer
        for n in range(self.Ns):
            r_prime = self.scatterers[:, n].reshape(2, 1, 1)  # Shape: (2, 1, 1)
            delta_r = np.sqrt(
                (self.Z_grid - r_prime[0]) ** 2 + (self.X_grid - r_prime[1]) ** 2
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

        # Compute wavevector components
        k_z = self.k * np.cos(self.theta_inc)
        k_x = self.k * np.sin(self.theta_inc)

        # Compute incident field at z = Lz for all x
        E_incident = np.exp(
            1j * (k_z * z_L + k_x * x_points)
        )  # Shape: (grid_points_x,)

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
            distances = np.sqrt(delta_z**2 + delta_x**2)  # Shape: (grid_points_x,)

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
        plt.ylim(0, self.Lx / self.lambda_0)
        plt.tight_layout()
        plt.show()

    def visualize_intensity(self):
        """Visualizes the intensity distribution over the entire slab."""
        # Ensure that E_total is computed
        if self.E_total is None:
            self.compute_total_field()

        plt.figure(figsize=(10, 8 * self.Lx / self.Lz))
        plt.pcolormesh(
            self.Z_grid / self.lambda_0,
            self.X_grid / self.lambda_0,
            np.abs(self.E_total) ** 2,
            shading="auto",
            cmap="viridis",
        )
        plt.colorbar(label=r"$|E|^2$")
        plt.scatter(
            self.scatterers[0, :] / self.lambda_0,
            self.scatterers[1, :] / self.lambda_0,
            c="red",
            s=100,
            label="Scatterers",
        )
        plt.title(f"Electric Field Intensity Over Slab (θ = {self.theta_inc_deg}°)")
        plt.xlabel(r"$z/\lambda$")
        plt.ylabel(r"$x/\lambda$")
        plt.legend()
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        plt.show()

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
            plt.plot(x_values, np.abs(E_output) ** 2, "b-", label=r"$|E(z=L_z)|^2$")
            plt.title("Electric Field Intensity at $z = L_z$")
            plt.xlabel(r"$x/\lambda$")
            plt.ylabel(r"$|E|^2$")
            plt.grid(True)
            plt.legend()

            # Phase plot
            plt.subplot(1, 2, 2)
            plt.plot(x_values, np.angle(E_output), "r-", label=r"$\arg(E(z=L_z))$")
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
            plt.plot(x_values, np.abs(E_output) ** 2, "g-", label=r"$|E(z=L_z)|^2$")
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


if __name__ == "__main__":
    import time  # Import the time module for performance measurement

    # Start timer
    start_time = time.time()

    # Initialize the simulator with a specific incident angle
    simulator = CoupledDipoleSimulator(theta_inc_deg=45)

    # Display scatterer positions using the new method
    simulator.visualize_scatterers()

    # Visualize the intensity over the full slab
    simulator.visualize_intensity()

    # Visualize the fields at z = Lz with cross-section plots
    simulator.visualize_output_field(direct_computation=True)

    # Optionally, retrieve and print the electric field at z = Lz
    E_zL = simulator.compute_zL_fields_direct()

    # Stop timer and print elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nExecution time: {elapsed_time:.3f} seconds")
