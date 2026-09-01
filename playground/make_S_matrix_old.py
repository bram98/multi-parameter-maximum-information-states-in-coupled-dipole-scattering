# %%
import sys

import fourierfun as ff
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../libraries")

from fourierfun import cplot

# from fourierfun import cplot
from libraries.ScatteringSimulation import CoupledDipoleSimulator

# %%

Ntheta = 20

cds = CoupledDipoleSimulator(
    lambda_0=1.0,
    Lz=10,
    Lx=2,
    Ns=0,
    alpha=4j / (2 * np.pi / 1.0) ** 2,
    exclusion_radius=0.637,  # Derived from sigma_s ≈ 0.637λ₀
    grid_points_x=32,
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

# %%

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


# %%
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

    E_samples = [
        sample_frequency(E_output, x, k, fr.ymax) for k in k_x_samples
    ]

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
