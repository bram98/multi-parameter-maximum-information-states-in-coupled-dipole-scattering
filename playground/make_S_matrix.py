#%%
import numpy as np
import sys
sys.path.append('../libraries')
from ScatteringSimulation import CoupledDipoleSimulator

#%%
cds = CoupledDipoleSimulator(
        lambda_0=1.,
        Lz=16.0,
        Lx=8.0,
        Ns=0,
        alpha=4j / (2 * np.pi / 1.0)**2,
        exclusion_radius=0.637,  # Derived from sigma_s ≈ 0.637λ₀
        grid_points_x=400,
        theta_inc_deg=0.0,
        seed=0
    )

for theta in np.linspace(0, np.pi/2, num=10):
    cds.theta_inc = theta
    cds.reset()
    cds.compute_total_field()
    cds.visualize_intensity()



    # def __init__(self,
    #              lambda_0=1.0,
    #              Lz=16.0,
    #              Lx=8.0,
    #              Ns=10,
    #              alpha=4j / (2 * np.pi / 1.0)**2,
    #              exclusion_radius=0.637,  # Derived from sigma_s ≈ 0.637λ₀
    #              grid_points_x=400,
    #              theta_inc_deg=0.0,
    #              seed=0):