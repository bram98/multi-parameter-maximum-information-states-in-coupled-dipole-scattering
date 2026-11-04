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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal.windows import tukey
from scipy.spatial.distance import cdist
from scipy.special import hankel1

from scatteringlibs.matplotlibmagic import register_turbo
from scatteringlibs.storable import Storable

register_turbo()


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
        NA_out (float, optional): Numerical aperture of the output objective for filtering.
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
        Lz=8.0,
        Lx=8.0,
        Lz_vis=8.0,
        Lx_vis=8.0,
        grid_points_x=16,
        grid_points_x_vis=200,
        grid_points_z_vis=None,
        alpha=4j / (2 * np.pi / 1.0) ** 2,
        # exclusion_radius=0.637,  # Derived from sigma_s ≈ 0.637λ₀
        NA_out=0.95,
        theta_inc_deg=None,
        theta_inc=None,
        # seed=0,
        k_x_array=None,
        E_kx_array=None,
        movable_scatterers=None,
        frozen_scatterers=None,
    ):
        """Initializes the CoupledDipoleSimulator with default or user-defined parameters."""
        self.lambda_0 = lambda_0
        self.k = 2 * np.pi / self.lambda_0
        self.Lz = Lz  # Lz is saved in its native units: it has same units as wavelength
        self.Lx = Lx

        self.grid_points_x = grid_points_x
        self.x = np.linspace(
            -self.Lx / 2, self.Lx / 2, self.grid_points_x, endpoint=False
        )
        self.alpha = alpha
        # self.exclusion_radius = exclusion_radius * self.lambda_0
        self.NA_out = NA_out

        self.set_input_field_angles(
            k_x_array=k_x_array,
            theta_inc=theta_inc,
            theta_inc_deg=theta_inc_deg,
        )

        # self.seed = seed

        self.Lz_vis = (
            Lz_vis / self.lambda_0
        )  # distances are displayed in units of wavelength
        self.Lx_vis = Lx_vis / self.lambda_0
        self.grid_points_x_vis = grid_points_x_vis
        if grid_points_z_vis is None:
            # Derive grid points along z based on x sampling
            # Note: for endpoint=True, you need to subtract one from grid_points
            delta_x = self.Lx_vis / (self.grid_points_x_vis - 1)
            self.grid_points_z_vis = int(np.round(self.Lz_vis / delta_x + 1))
        else:
            self.grid_points_z_vis = grid_points_z_vis

        # Generate grid for visualization
        self.z_vis = np.linspace(
            -self.Lz_vis / 2,
            self.Lz_vis / 2,
            self.grid_points_z_vis,
            endpoint=True,
        )
        self.x_vis = np.linspace(
            -self.Lx_vis / 2,
            self.Lx_vis / 2,
            self.grid_points_x_vis,
            endpoint=True,
        )
        self.Z_grid, self.X_grid = np.meshgrid(self.z_vis, self.x_vis)

        self.movable_scatterers = movable_scatterers
        self.frozen_scatterers = frozen_scatterers
        # Initialize scatterer positions
        # if movable_scatterers is None:
        #     self.movable_scatterers = self._generate_scatterers()
        # else:
        #     self.movable_scatterers = movable_scatterers

        # if frozen_scatterers is not None:
        #     self.frozen_scatterers = frozen_scatterers
        #   Note: self.scatterers is defined via a getter

        # Initialize fields as None; compute when needed
        self.fields = None
        self.E_total = None
        self.E_output = None

        # For arbitrary input fields
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

    def set_scatterers(self, movable_scatterers=None, frozen_scatterers=None):
        if movable_scatterers is not None:
            try:
                movable_scatterers = np.asarray(movable_scatterers)
                assert movable_scatterers.shape[0] == 2
            except Exception as e:
                print(
                    f"Expected array of shape (2, N) where N is the number of movable scatters. Instead got movable_scatterers={movable_scatterers}"
                )
                raise e
            self.movable_scatterers = movable_scatterers

        if frozen_scatterers is not None:
            try:
                frozen_scatterers = np.asarray(frozen_scatterers)
                assert frozen_scatterers.shape[0] == 2
            except Exception as e:
                print(
                    f"Expected array of shape (2, N) where N is the number of frozen_scatterers scatters. Instead got frozen_scatterers={movable_scatterers}"
                )
                raise e
            self.frozen_scatterers = frozen_scatterers

    def set_input_field_angles(
        self, k_x_array=None, theta_inc=None, theta_inc_deg=None
    ):
        """
        Sets the input field angles. The input field is written as a superposition of
        plane waves. The angles are input via k_x_array, which describes the kx
        components of each planewave. The amplitudes are give by E_kx_array,
        which should have the same shape as k_x_array.
        """

        if k_x_array is not None:
            self.k_x_array = np.array(k_x_array)
        elif theta_inc_deg is not None:
            self.theta_inc_deg = np.array(theta_inc_deg)
        elif theta_inc is not None:
            self.theta_inc = np.array(theta_inc)
        else:
            self.theta_inc = 0.0

    def set_input_field_amplitudes(self, E_kx_array, flux_normalize=True):
        """
        Sets the input field amplitudes. The input field is written as a
        superposition of plane waves. The angles are input via k_x_array, which describes the kx
        components of each planewave. The amplitudes are give by E_kx_array,
        which should have the same shape as k_x_array. The amplitudes can
        be automatically flux_normalized by passing flux_normalized=True. This
        means each plane wave has unit flux through the simulation box. This
        method also calls reset().

        Args:
            k_x_array (numpy.ndarray): Array of kx values.
            E_kx_array (numpy.ndarray): Array of amplitudes
            flux_normalize (bool): Use flux normalization or use raw amplitudes
        """

        E_kx_array = np.array(E_kx_array)

        if flux_normalize:
            amp = _flux_amplitude_2d(self.k, self.k_x_array)
        else:
            amp = 1.0

        self.E_kx_array = E_kx_array * amp
        self.reset()  # Reset computed fields

    def set_input_field_kx(self, k_x_array, E_kx_array, flux_normalize=True):
        """
        Sets both k_x_array and E_kx_array. The name is for backwards
        compatibility
        """
        self.set_input_field_angles(k_x_array=k_x_array)
        self.set_input_field_amplitudes(
            E_kx_array, flux_normalize=flux_normalize
        )

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

    def _NA_filter(
        self, E, NA: float, x: np.ndarray = None, window_start: int = None
    ):
        """
        Performs an NA filter equivalent to a 4F system with a limiting
        numerical aperture given by NA. If window_start is provided, the region
        outside [window_start:-window_start] is used as padding, which greatly
        reduces ringing on the edges of the domain.

        Args:
            E (np.ndarray): input field
            NA (float): numerical aperture
            x (np.ndarray): x positions of input field. If None, the simulation
                            x is used. Should have same dimension as E
            window_start (int): the valid region is defined by the slice
                                [window_start:-window_start].
        """
        if window_start is not None:
            alpha = window_start / len(E)
            E *= tukey(len(E), alpha=alpha)
        # Apply NA filter in Fourier space
        E_kx = np.fft.fft(E, norm="ortho")

        if x is None:
            x = self.x

        assert len(x) == len(E)

        # Define spatial frequency vector
        dx = x[1] - x[0]
        kx_vec = np.fft.fftfreq(len(x), d=dx) * 2 * np.pi

        # Define cutoff frequency and mask
        k_max = NA * self.k
        mask = np.abs(kx_vec) <= k_max

        # Apply mask and inverse transform
        E_kx_filtered = E_kx * mask
        E_output_filtered = np.fft.ifft(E_kx_filtered, norm="ortho")

        if window_start is not None:
            E_output_filtered = E_output_filtered[window_start:-window_start]
        return E_output_filtered

    def compute_output_fields_direct(self):
        """
        Directly computes the electric field at z = Lz/2 plane.
        If self.NA_out is specified, it applies a Fourier-space filter to model
        a finite numerical aperture objective.
        """
        if self.E_output is not None:
            print(
                "Warning: using precomputed values. Use sim.reset() to prevent this."
            )
            return self.E_output

        # Ensure that induced fields are computed
        if self.fields is None:
            self.solve_fields()

        # Define z = Lz / 2
        z_out = self.Lz / 2

        # Define x coordinates along z = Lz
        if self.NA_out is None:
            x_points = self.x  # Shape: (grid_points_x,)
        else:
            """
            Here we create an array that is twice as large as the domain. We
            can use this extra padding to improve the Fourier filtering. We use
            a Tukey window to reduce leakage. The lobes of the Tukey window
            fall in this extra padding.
            """
            dx = self.x[1] - self.x[0]
            n_half = len(self.x) // 2
            x_points = np.linspace(
                self.x[0] - dx * n_half,
                self.x[-1] + dx * n_half,
                len(self.x) + 2 * n_half,
                endpoint=True,
            )  # this awkward line prevents one-off errors.

        # Compute incident field at z = Lz for all x
        E_incident = self.compute_incident_field(x_points, z_out)
        E_output_unfiltered = np.copy(E_incident)

        # Compute scattered fields from each scatterer
        for n in range(self.Ns):
            # Position of scatterer n
            r_n = self.scatterers[:, n]  # Shape: (2,)

            # Observation points at z = Lz, all x
            # Vectorized computation
            delta_z = z_out - r_n[0]  # Scalar
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
            E_output_unfiltered += E_scatt

        if self.NA_out is None:
            self.E_output = E_output_unfiltered
            return self.E_output
        else:
            self.E_output = self._NA_filter(
                E_output_unfiltered,
                NA=self.NA_out,
                x=x_points,
                window_start=n_half,
            )
            return self.E_output

    def reset(self):
        """Resets computed fields."""
        self.fields = None
        self.E_output = None
        self.E_total = None

    def visualize_scatterers(
        self,
        movable_kwargs=None,
        frozen_kwargs=None,
        ax=None,
        figsize=None,
        title=None,
    ):
        """Visualizes the positions of scatterers within the slab."""
        movable_plot = None
        frozen_plot = None
        newfig = ax is None
        if newfig:
            if figsize is None:
                figsize = (12, 12 * self.Lx_vis / self.Lz_vis)
            fig, ax = plt.subplots(figsize=figsize)

        if self.movable_scatterers is not None:
            if movable_kwargs is None:
                movable_kwargs = {}
            movable_kwargs_default = dict(
                facecolors="none",
                edgecolors="r",
                s=80,
                marker="o",
                label="Scatterers",
            )
            movable_kwargs_default.update(movable_kwargs)
            movable_plot = ax.scatter(
                self.movable_scatterers[0, :] / self.lambda_0,
                self.movable_scatterers[1, :] / self.lambda_0,
                **movable_kwargs_default,
            )

        if self.frozen_scatterers is not None:
            if frozen_kwargs is None:
                frozen_kwargs = {}
            frozen_kwargs_default = dict(
                facecolors="none",
                edgecolors="r",
                s=80,
                marker="s",
                label="Static scatterers",
            )
            frozen_kwargs_default.update(frozen_kwargs)
            frozen_plot = ax.scatter(
                self.frozen_scatterers[0, :] / self.lambda_0,
                self.frozen_scatterers[1, :] / self.lambda_0,
                **frozen_kwargs_default,
            )

        if newfig:
            ax.set_title(title)
            ax.set_xlabel(r"$z/\lambda$")
            ax.set_ylabel(r"$x/\lambda$")
            plt.legend()
            ax.set_aspect("equal")
            ax.set_xlim(
                -self.Lz_vis / self.lambda_0 / 2,
                self.Lz_vis / self.lambda_0 / 2,
            )
            ax.set_ylim(
                -self.Lx_vis / self.lambda_0 / 2,
                self.Lx_vis / self.lambda_0 / 2,
            )
            plt.tight_layout()
            plt.show()
        return dict(movable_plot=movable_plot, frozen_plot=frozen_plot)

    def _visualize(
        self,
        display_function,
        title,
        dpi=100,
        figsize=(10, 8),
        ax=None,
        use_scatterers=True,
        use_colorbar=True,
        use_legend=True,
        imshow_kwargs=None,
        movable_kwargs=None,
        frozen_kwargs=None,
        legend_kwargs=None,
        colorbar_label=None,
    ):
        """
        Helper function that applies `display_function` to the entire field
        before plotting it.
        """
        newfig = ax is None
        if newfig:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Ensure that E_total is computed
        if self.E_total is None:
            self.compute_total_field()

        if imshow_kwargs is None:
            imshow_kwargs = {}

        imshow_kwargs_default = dict(
            shading="auto",
            cmap="turbo",
        )
        imshow_kwargs_default.update(imshow_kwargs)

        # field_img = ax.pcolormesh(
        #     self.Z_grid / self.lambda_0,
        #     self.X_grid / self.lambda_0,
        #     display_function(self.E_total),
        #     **imshow_kwargs,
        # )
        dz = self.Z_grid[0, 1] - self.Z_grid[0, 0]
        dx = self.X_grid[1, 0] - self.X_grid[0, 0]
        extent = [
            np.min(self.Z_grid) - 0.5 * dz,
            np.max(self.Z_grid) + 0.5 * dz,
            np.min(self.X_grid) - 0.5 * dx,
            np.max(self.X_grid) + 0.5 * dx,
        ]
        field_img = ax.imshow(
            display_function(self.E_total),
            extent=extent,
            origin="lower",
            **imshow_kwargs,
        )

        if use_scatterers:
            scatter_plots = self.visualize_scatterers(
                movable_kwargs=movable_kwargs,
                frozen_kwargs=frozen_kwargs,
                ax=ax,
            )
        else:
            scatter_plots = dict(movable_plot=None, frozen_plot=None)

        ax.set_title(title)

        if use_legend:
            if legend_kwargs is None:
                legend_kwargs = {}
            legend = ax.legend(**legend_kwargs)
        else:
            legend = None
        ax.set_xlabel(r"$z/\lambda$")
        ax.set_ylabel(r"$x/\lambda$")
        ax.set_aspect("equal")
        if use_colorbar:

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(field_img, cax=cax, label=colorbar_label)
            # plt.colorbar(field_img, ax=ax)
        if newfig:
            plt.tight_layout()
            plt.show()
        return dict(
            ax=ax,
            field_img=field_img,
            movable_plot=scatter_plots["movable_plot"],
            frozen_plot=scatter_plots["frozen_plot"],
            legend=legend,
        )

    def visualize_intensity(
        self,
        title="Custom Input State",
        dpi=100,
        figsize=(10, 8),
        ax=None,
        use_scatterers=True,
        use_colorbar=True,
        use_legend=True,
        imshow_kwargs=None,
        movable_kwargs=None,
        frozen_kwargs=None,
        legend_kwargs=None,
        colorbar_label=r"$|E|^2$  (a.u.)",
    ):
        """Visualizes the intensity distribution over the entire slab."""

        if self.k_x_array is not None and self.E_kx_array is not None:
            pass
        else:
            title = f"Electric Field Intensity Over Slab (θ = {self.theta_inc_deg:.1f}°)"

        return self._visualize(
            display_function=lambda field: np.abs(field) ** 2,
            title=title,
            dpi=dpi,
            figsize=figsize,
            ax=ax,
            use_scatterers=use_scatterers,
            use_colorbar=use_colorbar,
            use_legend=use_legend,
            imshow_kwargs=imshow_kwargs,
            movable_kwargs=movable_kwargs,
            frozen_kwargs=frozen_kwargs,
            legend_kwargs=legend_kwargs,
            colorbar_label=colorbar_label,
        )

    def visualize_abs(
        self,
        title="Custom Input State",
        dpi=100.0,
        ax=None,
        colorbar=True,
        figsize=(10, 8),
    ):
        """Visualizes the absolute value of the field over the entire slab."""

        if self.k_x_array is not None and self.E_kx_array is not None:
            title = "Custom Input State: Absolute Electric Field"
        else:
            title = f"Absolute Value of Electric Field Over Slab (θ = {self.theta_inc_deg:.1f}°)"

        return self._visualize(
            display_function=np.abs,
            title=title,
            dpi=dpi,
            ax=ax,
            figsize=figsize,
        )

    def visualize_real(
        self,
        title="Custom Input State",
        dpi=100.0,
        ax=None,
        colorbar=True,
        figsize=(10, 8),
    ):
        """Visualizes the real value of the field over the entire slab."""

        # if self.k_x_array is not None and self.E_kx_array is not None:
        #     title = "Custom Input State: Real Part of Electric Field"
        # else:
        #     title = f"Real Value of Electric Field Over Slab (θ = {self.theta_inc_deg:.1f}°)"

        return self._visualize(
            display_function=np.real,
            title=title,
            dpi=dpi,
            ax=ax,
            figsize=figsize,
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
                E_output = self.compute_output_fields_direct()
            else:
                # Extract fields at z = Lz
                E_output = self.get_output_fields()
        else:
            E_output = self.get_output_fields()

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

    def get_output_fields(self):
        """Returns the electric field at z = Lz/2 plane.

        Returns:
            numpy.ndarray: Electric field at z = Lz/2 (shape: (grid_points_x,)).
        """
        if self.E_output is not None:
            return self.E_output

        # Ensure that E_total is computed
        if self.E_total is None:
            self.compute_total_field()

        self.E_output = self.E_total[:, -1]
        return self.E_output

    # k_x_array is the variable that's actually stored. theta_inc and
    # theta_inc_deg just shadow k_x_array.
    @property
    def theta_inc_deg(self):
        return self.k_x2theta(self.k_x_array, degree=True)

    @theta_inc_deg.setter
    def theta_inc_deg(self, theta_deg):
        self.k_x_array = self.theta2k_x(theta_deg, degree=True)

    @property
    def theta_inc(self):
        return self.k_x2theta(self.k_x_array, degree=False)

    @theta_inc.setter
    def theta_inc(self, theta):
        self.k_x_array = self.theta2k_x(theta, degree=False)

    def get_kwargs(self):

        keys = [
            "lambda_0",
            "Lz",
            "Lx",
            "Lz_vis",
            "Lx_vis",
            "grid_points_x",
            "grid_points_x_vis",
            "grid_points_z_vis",
            # "Ns",
            "alpha",
            "NA_out",
            # "exclusion_radius",
            # "theta_inc_deg",
            # "theta_inc",
            # "seed",
            "k_x_array",
            "E_kx_array",
            "movable_scatterers",
            "frozen_scatterers",
        ]
        kwargs = {
            key: self.__getattribute__(key)
            for key in keys
            if self.__getattribute__(key) is not None
        }
        return kwargs

    def copy(self):
        return CoupledDipoleSimulator(**self.get_kwargs())

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

    @staticmethod
    def calc_critical_sampling_at_N_modes(
        N_modes=45, NA_full=0.95, NA_in=None, lambda_0=1.0
    ):
        k0 = 2 * np.pi / lambda_0
        n = np.ceil(N_modes / 2)
        dk = k0 * NA_full / n

        dx = 2 * np.pi / (N_modes * dk)

        k_x_array = np.fft.fftfreq(N_modes, d=dx)

        k_x_array = np.fft.fftshift(k_x_array)

        if NA_in is not None:
            k_x_array = k_x_array[np.where(np.abs(k_x_array) <= NA_in)]

        k_x_array *= 2 * np.pi

        Lx = dx * N_modes

        return dict(k_x_array=k_x_array, Lx=Lx, grid_points_x=N_modes)

    @staticmethod
    def calc_critical_sampling_at_Lx(self, NA_in=None, Lx=None, k=None):
        """
        Creates a kx array that is critically sampled, which means that the
        the input states form an orthogonal basis if we assume that the
        simulation box is a waveguide with periodic boundary conditions.
        More precisely, it creates the array
        [ -(n-1)*dkx, ..., -dkx, 0, dkx, ..., (n-1)*dkx ], where dkx allows for
        periodic boundary conditions in the box and (n-1)*dkx is smaller than
        or equal to k.
        """
        if NA_in is None:
            NA_in = self.NA_in
            if NA_in is None:
                NA_in = 1.0
        if Lx is None:
            Lx = self.Lx
        if k is None:
            k = self.k

        dkx = 2 * np.pi / Lx
        n = np.floor(k * NA_in / dkx) + 1
        n = int(n)
        max_kx = (n - 1) * dkx
        # print(dkx, n)
        return np.linspace(-max_kx, max_kx, num=2 * n - 1)

    def build_scattering_matrix(self, flux_normalize=True):
        """
        Build a scattering matrix S of shape (N_out, N_in) where:
        - N_in = len(angles_deg)
        - N_out = self.grid_points_x (the direct samples at z=Lz)
        For each angle in angles_deg, we set a single plane wave input,
        solve, read off E_zL. Then store in S.
        Returns:
            tuple (S, x_cam):
                S: shape (N_out, N_in)
                x_cam: array of x-values at z=Lz/2
        """
        N_in = len(self.k_x_array)
        N_out = self.grid_points_x
        S = np.zeros((N_out, N_in), dtype=complex)
        k_in = self.k_x_array.copy()

        # We'll do a loop over angles
        for n, kx in enumerate(self.k_x_array):
            # set input
            self.set_input_field_kx([kx], [1.0], flux_normalize=flux_normalize)
            self.solve_fields()
            E_output = self.compute_output_fields_direct()
            S[:, n] = E_output
        S = np.fft.fft(S, axis=0, norm="ortho")
        S = np.fft.fftshift(S, axes=0)

        dx = self.x[1] - self.x[0]
        k_out = np.fft.fftfreq(N_out, d=dx) * 2 * np.pi
        k_out = np.fft.fftshift(k_out)

        self.k_x_array = k_in  # restore k_x_array

        return S, (k_in, k_out)

    def calculate_dS(
        self,
        shifts: np.ndarray,
        deltas: np.ndarray,
        symmetric: bool = True,
        flux_normalize: bool = True,
    ):
        """
        Calculates the d_\mu S tensor. Calculates a numerical derivative by
        applying the shifts to the movable dipoles and calculating the
        scattering matrix. Calculates a symmetric derivate when symmetric==True
        and a forward derivative otherwise. Assumes `deltas` is the step size.

        Args:
            shifts: array of shape (p, 2, len(self.movable_dipoles[0]))
                shifts to apply to each dipole. p is the number of parameters
            deltas: array of shape (p,)
                The step size to use in numerical derivative. If
                symmetric==True, will be multiplied by 2
            symmetric:
                whether to use symmetric or forward derivative. Default: True

        """
        shifts = np.asarray(shifts)
        deltas = np.asarray(deltas)
        p = len(deltas)

        if symmetric:
            deltas *= 2

        sim_plus = self.copy()
        sim_minus = self.copy()
        movable_dipoles = sim.movable_scatterers.copy()
        # movable_plus = movable_dipoles + shifts

        N_in = len(self.k_x_array)
        N_out = self.grid_points_x
        dS = np.zeros((p, N_out, N_in), dtype=complex)

        if not symmetric:
            S_minus, _ = sim_minus.build_scattering_matrix(
                flux_normalize=flux_normalize
            )

        for i in range(p):
            movable_plus = movable_dipoles + shifts[i]
            sim_plus.set_scatterers(movable_scatterers=movable_plus)

            S_plus, _ = sim_plus.build_scattering_matrix(
                flux_normalize=flux_normalize
            )

            if symmetric:
                movable_minus = movable_dipoles - shifts[i]
                sim_minus.set_scatterers(movable_scatterers=movable_minus)

                S_minus, _ = sim_minus.build_scattering_matrix(
                    flux_normalize=flux_normalize
                )

            dS[i] = (S_plus[i] - S_minus[i]) / deltas[i]
        return dS[i]

    @property
    def Ns(self):
        """
        Getter that returns the number of scatterers.
        """
        _Ns = 0
        if self.movable_scatterers is not None:
            _Ns += self.movable_scatterers.shape[1]

        if self.frozen_scatterers is not None:
            _Ns += self.frozen_scatterers.shape[1]

        return _Ns

    @property
    def scatterers(self):
        """
        Getter that returns movable and frozen scatterers as a single (2, N)
        array. If both are None, returns None.
        """
        if self.movable_scatterers is None and self.frozen_scatterers is None:
            return None

        scat = np.array([[], []])
        if self.movable_scatterers is not None:
            scat = np.append(scat, self.movable_scatterers, axis=1)
        if self.frozen_scatterers is not None:
            scat = np.append(scat, self.frozen_scatterers, axis=1)

        return scat


if __name__ == "__main__":
    import time

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
    E_zL = simulator.compute_output_fields_direct()

    # %% ------------------------ Arbitrary Input Field Demo

    # Define arbitrary input field in terms of k_x components
    angles_deg = np.array([-30, 0, 30])  # Incident angles in degrees
    angles_rad = np.deg2rad(angles_deg)  # Convert to radians
    k_x_array = simulator.k * np.sin(angles_rad)  # k_x values
    E_kx_array = np.array([1.0, 2, 1.0])  # Corresponding amplitudes

    # Initialize a new simulator instance for arbitrary input field with scatterers
    simulator_arbitrary = CoupledDipoleSimulator(Ns=5)

    # Set the arbitrary input field
    simulator_arbitrary.set_input_field_angles(angles_deg)
    simulator_arbitrary.set_input_field_amplitudes(
        E_kx_array, flux_normalize=False
    )

    # Visualize scatterer positions
    simulator_arbitrary.visualize_scatterers()

    # Visualize the intensity over the full slab with arbitrary input field
    simulator_arbitrary.visualize_intensity()

    # Visualize the fields at z = Lz with cross-section plots
    simulator_arbitrary.visualize_output_field(direct_computation=True)

    # Optionally, retrieve and print the electric field at z = Lz
    E_zL_arbitrary = simulator_arbitrary.compute_output_fields_direct()

    # ------------------------ Arbitrary Input Field Without Scatterers Demo
    # Initialize a new simulator instance for arbitrary input field with no scatterers
    simulator_arbitrary_no_scatterers = CoupledDipoleSimulator(Ns=0)

    # Set the same arbitrary input field

    simulator_arbitrary_no_scatterers.set_input_field_angles(angles_deg)
    simulator_arbitrary_no_scatterers.set_input_field_amplitudes(E_kx_array)

    # Visualize the intensity over the full slab with arbitrary input field and no scatterers
    simulator_arbitrary_no_scatterers.visualize_intensity()

    # Visualize the fields at z = Lz with cross-section plots
    simulator_arbitrary_no_scatterers.visualize_output_field(
        direct_computation=True
    )

    # Optionally, retrieve and print the electric field at z = Lz
    E_zL_arbitrary_no_scatterers = (
        simulator_arbitrary_no_scatterers.compute_output_fields_direct()
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
