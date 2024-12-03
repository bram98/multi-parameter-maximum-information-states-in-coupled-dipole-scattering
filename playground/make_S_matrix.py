#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
import fourierfun as ff
sys.path.append('../libraries')
from ScatteringSimulation import CoupledDipoleSimulator

#%%

Ntheta = 20

cds = CoupledDipoleSimulator(
        lambda_0=1.,
        Lz=4*np.pi,
        Lx=2*np.pi,
        Ns=0,
        alpha=4j / (2 * np.pi / 1.0)**2,
        exclusion_radius=0.637,  # Derived from sigma_s ≈ 0.637λ₀
        grid_points_x=400,
        theta_inc_deg=0.0,
        seed=0
    )

fr = ff.FourierRange2d(nx=cds.grid_points_z, ny=cds.grid_points_x,
                       xmax=cds.Lz, ymax=cds.Lx,
                       k_in_radians=True)

theta_degs = np.linspace(0, 80, num=Ntheta)
theta_rads = theta_degs / 180 * np.pi
S = np.zeros((Ntheta, Ntheta), dtype=complex)

k_x_range = fr.kyrange()
k_x_samples = cds.theta2k_x(theta_rads)
k_x_index = np.round(k_x_samples / fr.dky).astype(int)

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
    E_inc_k = np.fft.fft(E_incident)
    plt.plot(k_x_range, np.abs(E_inc_k) + 10, label='fft input')
    plt.xlabel('$k_x$')
    plt.ylabel(r'$|fft(E)|$')
    print(np.argmax(np.abs(E_inc_k)))
    plt.axvline(k_x_range[np.argmax(np.abs(E_inc_k))], c='r', label='maximum')
    print(k_x_range[np.argmax(np.abs(E_inc_k))])
    # plt.show()
    
    E_k = np.fft.fft(E_output)
    E_k_samples = E_k[k_x_index]
    # k_output = np.fft.fftfreq(Ntheta)
    # k_z = fr.kyrange()
    plt.plot(k_x_range, np.abs(E_k), label='fft output')
    plt.axvline(k_x_samples[i], c='g', label='predicted maximum')
    plt.legend()
    # print(k_x_samples[i])
    plt.show()
    S[i] = E_k_samples
    
plt.imshow(np.abs(S))
plt.show()
    


