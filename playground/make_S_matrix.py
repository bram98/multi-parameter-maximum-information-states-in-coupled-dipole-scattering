# %%
import sys

import fourierfun as ff
import matplotlib.pyplot as plt
import numpy as np
from libraries.ScatteringSimulation import CoupledDipoleSimulator
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, MultipleLocator

# %%


class SMatrixCalculator:
    left = 1
    right = 2
    both = 3

    def __init__(self, N, input_states_l=None, input_states_r=None):
        self.N = N
        self.input_states_l = (
            None  # input states left (can be angles or vectors etc.)
        )
        self.input_states_r = None  # input states right
        self.Tlr = None  # transmission left to right
        self.Trl = None  # transmission right to left
        self.Rll = None  # reflection left to left
        self.Rrr = None  # transmission right to right

    def calc_input_left(self):
        raise NotImplementedError

    def calc_input_right(self):
        raise NotImplementedError

    def calc_input_both(self):
        raise NotImplementedError

    def calc_from_left(self):
        raise NotImplementedError

    def calc_from_right(self):
        raise NotImplementedError

    def calc_both(self):
        raise NotImplementedError

    def get_matrix(self, inc, out):
        if inc == self.left and out == self.left:
            return self.Rll
        elif inc == self.left and out == self.right:
            return self.Tlr
        elif inc == self.right and out == self.left:
            return self.Trl
        elif inc == self.right and out == self.right:
            return self.Rrr

        raise ValueError("Values inc={inc} and out={out} are not valid.")

    def get_matrices(self, inc=left, out=left):
        matrices = []
        if inc & self.left and out & self.left and not self.Rll is None:
            matrices.append(self.Rll)
        elif inc & self.left and out & self.right and not self.Tlr is None:
            matrices.append(self.Tlr)
        elif inc == self.right and out & self.left and not self.Trl is None:
            matrices.append(self.Trl)
        elif inc & self.right and out & self.right and not self.Rrr is None:
            matrices.append(self.Rrr)

        return matrices

    def set_zero(self, inc=both, out=both):
        if inc & self.left and out & self.left:
            self.Rll = np.zeros((self.N, self.N), dtype=complex)
        elif inc & self.left and out & self.right:
            self.Tlr = np.zeros((self.N, self.N), dtype=complex)
        elif inc == self.right and out & self.left:
            self.Trl = np.zeros((self.N, self.N), dtype=complex)
        elif inc & self.right and out & self.right:
            self.Rrr = np.zeros((self.N, self.N), dtype=complex)

    def show(self, inc=both, out=both, log=False):
        for matrix in self.get_matrices(inc, out):
            # fig, ax = plt.subplots()
            if log:
                plt.imshow(np.abs(matrix), norm=LogNorm(vmin=0.001, vmax=1.0))
            else:
                plt.imshow(np.abs(matrix))
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.colorbar()
            plt.show()


class CoupledDipoleSMatrix(SMatrixCalculator):
    def __init__(
        self,
        N_states=11,
        theta_max_deg=None,
        theta_max=None,
        lambda_0=1.0,
        Lz=16.0,
        Lx=8.0,
        Ns=0,
        alpha=4j / (2 * np.pi / 1.0) ** 2,
        exclusion_radius=0.637,  # Derived from sigma_s ≈ 0.637λ₀
        grid_points_x=400,
        seed=0,
    ):
        super().__init__(N_states)
        self.N_states = N_states
        self.lambda_0 = lambda_0
        self.k = 2 * np.pi / lambda_0

        self.set_zero(SMatrixCalculator.left, SMatrixCalculator.right)

        if not theta_max is None:

            self.theta_max = theta_max

        elif not theta_max_deg is None:

            self.theta_max_deg = theta_max_deg

        assert -90 <= self.theta_max_deg <= 90

        self._kxmax = self.k * np.sin(self.theta_max)

        dk_sample = 2 * self._kxmax / (self.N_states - 0)

        xmax = 1 * 2 * np.pi / dk_sample
        dx = self.lambda_0 / 8.0
        n = np.ceil(xmax / dx).astype(int)
        # dx = xmax / n
        # print('_kxmax', self._kxmax)
        # print('dk_sample', dk_sample)
        # print('xmax', xmax)
        # print('dx', dx)
        # print('n', n)
        # print('dk`:\t\t', 2*np.pi/xmax)

        self.cds = CoupledDipoleSimulator(
            lambda_0=lambda_0,
            Lz=Lz,
            Lx=xmax,
            Ns=Ns,
            alpha=alpha,
            exclusion_radius=exclusion_radius,  # Derived from sigma_s ≈ 0.637λ₀
            grid_points_x=n + 1,
            theta_inc=0,
            seed=0,
        )

    def calc_input_both(self):
        self._kxmax = self.k * np.sin(self.theta_max)
        self.k_range = np.linspace(
            -self._kxmax,
            self._kxmax,
            num=self.N_states,
            endpoint=False,  # [ -kxmax, kxmax )
        )

        # dk_sample = 2*self._kxmax / (self.N_states - 0)
        # xmax = 1*2*np.pi/dk_sample
        # dx = self.lambda_0 / 4.
        # n = np.ceil(xmax/dx).astype(int)
        # dx = xmax / n
        # dx = self.cds.Lx / (self.cds.grid_points_x - 1)
        self.k_range = np.fft.fftfreq(self.N_states, d=1 / (2 * self._kxmax))
        # dk = 2*np.pi/xmax
        # print(dk, 1/(2*self._kxmax))

        # The k-values range from:
        # [0, ..., kxmax, -kxmax, ..., - dk] (odd) or
        # [dk, ..., kxmax, -kxmax, ..., dk]   (even)
        # self.k_range = np.fft.ifftshift(self.k_range)

        self.input_states_l = self.k_range
        self.input_states_r = self.k_range.copy()

    def for_each_input_left(self, func):
        for i, kx in enumerate(self.input_states_l):
            self.cds.theta_inc = self.k_x2theta(kx)

            self.cds.reset()
            func(self.cds, kx)

    def calc_from_left(self):

        k_x_range = self.input_states_l

        # debugging purposes; leave here for now.
        dkk = k_x_range[1] - k_x_range[0]
        # print(f'dk input:\t {dkk} ')
        # dx1 = self.cds.Lx / (self.cds.grid_points_x - 1)
        # k__ = np.fft.fftfreq(self.cds.grid_points_x - 1, dx1)*2*np.pi
        # dkk = k__[1] - k__[0]
        # print('dk np:\t\t', dkk)
        # dk2 = 2 * self._kxmax / (self.N_states - 0)
        # print('dk formula:\t', dk2)
        # print('dx calc:\t', dx1)
        # print('dx actual:\t', self.cds.x[1] - self.cds.x[0])
        # print('n1:\t\t\t', self.cds.x.__len__())

        k_x_samples = k_x_range
        k_x_samples_index = np.round(k_x_samples / dkk).astype(int)

        for i, kx in enumerate(self.input_states_l):
            # if i > self.N_states - 5:
            #     break

            self.cds.theta_inc = self.k_x2theta(kx)

            self.cds.reset()
            E_output = self.cds.compute_zL_fields_direct()

            # I need to exclude the endpoint in order for the 'k' to correspond with
            # the correct physical dimensions.
            E_output = E_output[:-1]

            E_k = np.fft.fft(E_output, norm="forward")
            # E_k = np.fft.fftshift(E_k)

            plt.plot(np.real(E_k), label="Re(E_k)")
            plt.plot(np.abs(E_k), label="abs(E_k)")

            # plt.plot(np.real(E_output), label='Re(E_k)')
            # plt.plot(np.abs(E_output), label='abs(E_k)')

            plt.legend()
            plt.show()

            self.Tlr[i] = E_k[k_x_samples_index]

    def theta2k_x(self, theta):
        k_x = self.k * np.sin(theta)
        return k_x

    def k_x2theta(self, k_x):
        theta = np.arcsin(k_x / self.k)
        return theta

    @property
    def theta_max_deg(self):
        return np.rad2deg(self.theta_max)

    @theta_max_deg.setter
    def theta_max_deg(self, deg):
        self.theta_max = np.deg2rad(deg)


if __name__ == "__main__":
    cdsm = CoupledDipoleSMatrix(theta_max_deg=80, N_states=51, Ns=1)
    cdsm.calc_input_both()

    def show_total_field(cds, kx):
        # cds.compute_total_field()
        cds.visualize_abs()

    def show_output_field(cds, kx):
        cds.compute_zL_fields_direct()
        E_output = cds.E_output

        plt.plot(np.abs(E_output))
        plt.show()

        E_k = np.fft.fft(E_output)

        # plt.plot(np.abs(E_k))
        # plt.show()

    # cdsm.for_each_input_left(show_total_field)

    cdsm.calc_from_left()

    cdsm.show(log=True)
