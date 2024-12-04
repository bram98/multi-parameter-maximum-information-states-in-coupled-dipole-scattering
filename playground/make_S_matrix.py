#%%
import sys

import fourierfun as ff
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../libraries")

from fourierfun import cplot
from libraries.ScatteringSimulation import CoupledDipoleSimulator

#%%

class SMatrixCalculator:
    left: 1
    right: 2
    
    def __init__(self, N, input_states_l=None, input_states_r=None):
        self.N = N
        self.input_states_l = None      # input states left (can be angles or vectors etc.)
        self.input_states_r = None      # input states right
        self.Tlr = None                 # transmission left to right
        self.Trl = None                 # transmission right to left   
        self.Rll = None                 # reflection left to left
        self.Rrr = None                 # transmission right to right
    
    def calc_input_left(self):
        pass
    
    def calc_input_right(self):
        pass
    
    def calc_input_both(self):
        pass
    
    def calc_from_left(self):
        pass
    
    def calc_from_right(self):
        pass
    
    def calc_both(self):
        pass
      
        
def CoupledDipoleSMatrix(SMatrixCalculator):
    
    def __init__(self, 
                 N_input=11,
                 theta_max_deg=None,
                 theta_max=None,
                 lambda_0=1.0,
                 Lz=16.0,
                 Lx=8.0,
                 Ns=0,
                 alpha=4j / (2 * np.pi / 1.0)**2,
                 exclusion_radius=0.637,  # Derived from sigma_s ≈ 0.637λ₀
                 grid_points_x=400,
                 seed=0):
        
        self.lambda_0 = lambda_0
        self.k = 2 * np.pi / lambda_0
        
        if not theta_max is None:
            
            self.theta_max = theta_max
            
        elif not theta_max_deg is None:
            
            self.theta_max_deg = self.theta_max_deg
        
        assert -90 <= self.theta_max_deg <= 90
            
        self.cds = CoupledDipoleSimulator(lambda_0=lambda_0,
                                          Lz=Lz,
                                          Lx=Lx,
                                          Ns=Ns,
                                          alpha=alpha,
                                          exclusion_radius=exclusion_radius,  # Derived from sigma_s ≈ 0.637λ₀
                                          grid_points_x=400,
                                          theta_inc=0,
                                          seed=0)
        
        def calc_input_both(self):
            kxmax = self.k * np.sin(self.theta_max)
            self.k_range = np.linspace( - kxmax, kxmax,     # [ - kxmax, kxmax]
                                       num=self.N, 
                                       endpoint=True)
            
            # [0, ..., kxmax, - kxmax, ..., - dk] (odd) or
            # [dk, ..., kxmax, -kxmax, ..., dk]   (even)
            self.k_range = np.fft.ifftshift(self.k_range)
            
            self.input_states_l = self.k_x2theta(self.k_range)
            
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

Ntheta = 20

cds = CoupledDipoleSimulator(
    lambda_0=1.0,
    Lz=10,
    Lx=100,
    Ns=0,
    alpha=4j / (2 * np.pi / 1.0) ** 2,
    exclusion_radius=0.637,  # Derived from sigma_s ≈ 0.637λ₀
    grid_points_x=400,
    theta_inc_deg=0.0,
    seed=0,
)

fr = ff.FourierRange2d(
    nx=cds.grid_points_z,
    ny=cds.grid_points_x,
    xmax=cds.Lz,
    ymax=cds.Lx,
    k_in_radians=True,
)

theta_degs = np.linspace(0, 60, num=Ntheta)
theta_rads = theta_degs / 180 * np.pi
S = np.zeros((Ntheta, Ntheta), dtype=complex)
S_hr = np.zeros((Ntheta, Ntheta), dtype=complex)

k_x_range = fr.kyrange()
k_x_samples = cds.theta2k_x(theta_rads)
k_x_samples_index = np.round(k_x_samples / fr.dky).astype(int)

# pad with zeros in real space to perform Lanczos upsampling in Fourier space
n_highres = 800
fr_highres = ff.FourierRange.from_kw(n=n_highres, kmax=fr.kymax)
k_x_highres = fr_highres.krange()
k_x_samples_index_highres = np.round(k_x_samples / fr_highres.dk).astype(int)

for i, theta_deg in enumerate(theta_degs):
    cds.theta_inc_deg = theta_deg
    cds.reset()
    # cds.compute_total_field()
    # cds.visualize_real()
    # cds.visualize_abs()
    # cds.visualize_output_field()
    E_output = cds.compute_zL_fields_direct()

    k_x = cds.k * np.sin(cds.theta_inc)
    # print(k_x)
    x_points = fr.yrange()

    E_incident = np.exp(1j * (0 + k_x * x_points))
    # plt.plot(E_incident)
    # plt.show()
    E_inc_k = np.fft.fft(E_incident, norm="forward")
    plt.plot(k_x_range, np.abs(E_inc_k) + 10, label="fft input")
    plt.xlabel("$k_x$")
    plt.ylabel(r"$|fft(E)|$")
    # print(np.argmax(np.abs(E_inc_k)))
    plt.axvline(k_x_range[np.argmax(np.abs(E_inc_k))], c="r", label="maximum")
    # print(k_x_range[np.argmax(np.abs(E_inc_k))])
    # plt.show()

    # upsampling
    n_high_res = 800
    # fr2 = ff.FourierRange2d.from_kw(**fr.to_dict())

    E_output_highres = np.pad(
        E_output,
        pad_width=(n_high_res - cds.grid_points_x) // 2,
        mode="constant",
        constant_values=0.0,
    )
    E_output_highres = np.fft.ifftshift(E_output_highres)
    E_k = np.fft.fft(E_output, norm="forward")
    E_k_highres = np.fft.fft(E_output_highres, norm="forward")

    E_k_samples = E_k[k_x_samples_index]
    E_k_samples_highres = E_k_highres[k_x_samples_index_highres]
    # k_output = np.fft.fftfreq(Ntheta)
    # k_z = fr.kyrange()
    plt.plot(k_x_highres, np.abs(E_k_highres), label="fft output")
    plt.axvline(k_x_samples[i], c="g", label="predicted maximum")
    plt.legend()
    # print(k_x_samples[i])
    plt.show()
    S[i] = E_k_samples
    S_hr[i] = E_k_samples_highres

plt.imshow(np.abs(S))
plt.colorbar()
plt.show()
plt.imshow(np.abs(S_hr))
plt.colorbar()
plt.show()

#%%

plt.title("high res")
plt.plot(k_x_highres[:100], np.abs(E_k_highres)[:100])
plt.xlabel("$k_x$")
plt.ylabel("$abs(E)$")
plt.show()

plt.title("Low res")
plt.plot(k_x_range[:25], np.abs(E_k)[:25])
plt.xlabel("$k_x$")
plt.ylabel("$abs(E)$")
plt.show()


#%%
def sample_frequency(E, x, k, L):
    return np.trapezoid(np.exp(-1j * x * k) * E, x=x) / L


Ntheta = 20

cds = CoupledDipoleSimulator(
    lambda_0=1.0,
    Lz=4 * np.pi,
    Lx=100,
    Ns=0,
    alpha=4j / (2 * np.pi / 1.0) ** 2,
    exclusion_radius=0.637,  # Derived from sigma_s ≈ 0.637λ₀
    grid_points_x=400,
    theta_inc_deg=0.0,
    seed=0,
)

fr = ff.FourierRange2d(
    nx=cds.grid_points_z,
    ny=cds.grid_points_x,
    xmax=cds.Lz,
    ymax=cds.Lx,
    k_in_radians=True,
)

theta_degs = np.linspace(0, 60, num=Ntheta)
theta_rads = theta_degs / 180 * np.pi
S = np.zeros((Ntheta, Ntheta), dtype=complex)
S_hr = np.zeros((Ntheta, Ntheta), dtype=complex)

x = fr.yrange()
k_x_range = fr.kyrange()
k_x_samples = cds.theta2k_x(theta_rads)
k_x_samples_index = np.round(k_x_samples / fr.dky).astype(int)

# pad with zeros in real space to perform Lanczos upsampling in Fourier space
n_highres = 800
fr_highres = ff.FourierRange.from_kw(n=n_highres, kmax=fr.kymax)
k_x_highres = fr_highres.krange()
k_x_samples_index_highres = np.round(k_x_samples / fr_highres.dk).astype(int)

for i, theta_deg in enumerate(theta_degs):
    cds.theta_inc_deg = theta_deg
    cds.reset()
    # cds.compute_total_field()
    # cds.visualize_real()
    # cds.visualize_abs()
    # cds.visualize_output_field()
    E_output = cds.compute_zL_fields_direct()

    E_samples = [sample_frequency(E_output, x, k, fr.ymax) for k in k_x_samples]

    # k_x = cds.k * np.sin(cds.theta_inc)
    # # print(k_x)
    # x_points = fr.yrange()

    # E_incident = np.exp(1j * (0 + k_x * x_points))
    # # plt.plot(E_incident)
    # # plt.show()
    # E_inc_k = np.fft.fft(E_incident)
    # plt.plot(k_x_range, np.abs(E_inc_k) + 10, label='fft input')
    # plt.xlabel('$k_x$')
    # plt.ylabel(r'$|fft(E)|$')
    # print(np.argmax(np.abs(E_inc_k)))
    # plt.axvline(k_x_range[np.argmax(np.abs(E_inc_k))], c='r', label='maximum')
    # print(k_x_range[np.argmax(np.abs(E_inc_k))])
    # # plt.show()

    # S[i] = E_k_samples
    S_hr[i] = E_samples

plt.imshow(np.abs(S))
plt.show()
plt.imshow(np.abs(S_hr))
plt.show()
