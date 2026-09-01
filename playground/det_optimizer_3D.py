# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# %%
nparams = 3  # number of parameters
nhilbert = 3  # number of dimensions for states
r = 2.0  # 'radius' of eigenvalues: eigenvalues of dS will tend towards semicircular distribution with radius r

seed = 3  # seed for np random number generator

np.random.seed(seed)

filename = f"dS_np-{nparams}_nh-{nhilbert}_r-{r:.4f}_seed-{seed}.npy"
filepath = r"../data/numeric_scatter_deriv/" + filename
if os.path.isfile(filepath):
    print("Found existing data for current parameters. Loading...")
    dS = np.load(filepath)
else:
    sigma = r / np.sqrt(2 * nhilbert)
    dS = np.random.normal(
        loc=0.0, scale=sigma, size=(nparams, nhilbert, nhilbert)
    )
    np.save(filepath, dS)

F = [
    [dS_dmu.conj().T @ dS_dnu + dS_dnu.conj().T @ dS_dmu for dS_dmu in dS]
    for dS_dnu in dS
]
F = np.array(F)

G = np.random.normal(
    loc=0, scale=r / np.sqrt(2 * nhilbert), size=(nhilbert, nhilbert)
)


# %%
def J_from_F(F, v, engine=torch):
    return engine.einsum("mnij, ...i, ...j -> ...mn", F, v, v)


def det_F(F, v, engine=torch):
    return -engine.linalg.det(J_from_F(F, v, engine=engine))


def rayleigh(G, v, engine=torch):
    return engine.einsum("ij, ...i, ...j -> ...", G, v, v)


def forward(F, v):
    v = torch.nn.functional.normalize(v, dim=0)
    return det_F(F, v)


def forward_rayleigh(G, v):
    v = torch.nn.functional.normalize(v, dim=0)
    return rayleigh(G, v)


def snapshot(v):
    return v.detach().cpu().numpy().copy()


# %%

# debugging purposes

F = np.zeros((2, 2, 3, 3), dtype=int)
for i in range(2):
    for j in range(2):
        for k in range(3):
            for l in range(3):
                F[i, j, k, l] = 1000 * i + 100 * j + 10 * k + l


# np.einsum('mnij,i,j', F, u, v)
# %%
def cartesian_from_spherical(phi, theta):
    return np.array(
        [
            np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta),
        ]
    )


def spherical_from_cartesian(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.array([phi, theta])


def visualize_over_angles(func, F, ntheta=50, nphi=50):
    # if not F.shape[2:] == (3, 3):
    #     raise ValueError(f'Error Hilbert space needs to be 3D for visualization. F has shape {F.shape}')

    theta = np.linspace(0, np.pi, num=ntheta)
    phi = np.linspace(0, 2 * np.pi, num=nphi)
    Phi, Theta = np.meshgrid(phi, theta)

    v = cartesian_from_spherical(Phi, Theta)
    v = np.transpose(v, axes=(1, 2, 0))

    fig, ax = plt.subplots(layout="constrained")
    img = func(F, v, engine=np)
    angle_plot = ax.imshow(
        img, extent=(0, 2 * np.pi, 0, np.pi), origin="lower"
    )

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\theta$")
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="3%", pad="2%")
    fig.colorbar(angle_plot, cax=cax)

    return fig, ax


visualize_over_angles(det_F, F, ntheta=200, nphi=200)
# visualize_over_angles(rayleigh, G, ntheta=200, nphi=200)
plt.show()

# %%
v = torch.empty(3, requires_grad=True).cpu()
torch.nn.init.normal_(v)
Ft = torch.from_numpy(F.astype(np.float32)).cpu()
Ft.requires_grad = False

optimizer = torch.optim.Adam([v], lr=0.1)

v_history = []
phi_history = []
theta_history = []
loss_history = []

for i in range(100):
    optimizer.zero_grad()
    # loss = forward_rayleigh(Gt, v)
    loss = forward(Ft, v)
    loss.backward()
    optimizer.step()
    if i % 100 == 0 or True:
        v_npy = v.detach().cpu().numpy().copy()
        v_history.append(v_npy)
        phi, theta = spherical_from_cartesian(*v_npy)
        phi %= 2 * np.pi
        phi_history.append(phi)
        theta_history.append(theta)
        loss_history.append(loss.item())

fig, ax = visualize_over_angles(det_F, F, ntheta=200, nphi=200)
ax.plot(phi_history, theta_history, "-r", label="Path")
ax.plot(
    phi_history[-1],
    theta_history[-1],
    "*",
    c="tab:pink",
    label="Final solution",
)

ax.set_title("Det(J) using Adam")

plt.show()

# %%

from torchswarm.swarmoptimizer import SwarmOptimizer

v = torch.empty(3, requires_grad=True).cuda()
torch.nn.init.normal_(v)

# v_ = torch.nn.functional.normalize(v, dim=0)

# G = np.array([[1, 0, 0],
#              [0, 2, 0],
#              [0, 0, 3]])

# x = cartesian_from_spherical(2, 1)
# G = - np.outer(x, x)

# Gt = torch.from_numpy(G.astype(np.float32))
# Gt.requires_grad = False

Ft = torch.from_numpy(F.astype(np.float32)).cuda()
Ft.requires_grad = False


class DetFisher:
    def __init__(self, F):
        self.F = F

    def evaluate(self, v):
        return forward(self.F, v[:, 0])


# optimizer = torch.optim.Adam([v], lr=.1)
optimizer = SwarmOptimizer(
    3, 10, swarm_optimizer_type="standard", max_iterations=1, bounds=[-1, 1]
)


optimizer.optimize(DetFisher(Ft))


v_history = []
loss_history = []
swarm_history = []
swarm_phi = []
swarm_theta = []

for i in range(20):
    if i > 0:
        results = optimizer.run(verbosity=False)
        v_history.append(snapshot(optimizer.gbest_position[:, 0]))
        loss_history.append(results.gbest_value)
    swarm_positions = []
    swarm_phi_positions = []
    swarm_theta_positions = []
    for particle in optimizer.swarm:
        pos = snapshot(particle.position)[:, 0]
        swarm_positions.append(pos)
        phi, theta = spherical_from_cartesian(*pos)
        phi = phi % (2 * np.pi)
        swarm_phi_positions.append(phi)
        swarm_theta_positions.append(theta)
    swarm_history.append(swarm_positions)
    swarm_phi.append(swarm_phi_positions)
    swarm_theta.append(swarm_theta_positions)
# for i in range(100):
#     optimizer.run()
# for i in range(100):
#     optimizer.zero_grad()
#     # loss = forward_rayleigh(Gt, v)
#     loss = forward(Ft, v)
#     loss.backward()
#     optimizer.step()
#     if i%100 == 0 or True:
#         v_history.append(v.detach().cpu().numpy().copy())
#         loss_history.append(loss.item())


v_history = np.array(v_history)
swarm_history = np.array(swarm_history)
swarm_phi = np.array(swarm_phi).T
swarm_theta = np.array(swarm_theta).T

# phi, theta = spherical_from_cartesian(*v_history.T)
# phi = phi % (2*np.pi)


# fig, ax = visualize_over_angles(rayleigh, G, ntheta=200, nphi=200)
fig, ax = visualize_over_angles(det_F, F, ntheta=200, nphi=200)

for phi, theta in zip(swarm_phi, swarm_theta):
    ax.plot(phi, theta, "-r")


for phi, theta in zip(swarm_phi, swarm_theta):
    ax.plot(phi[-1], theta[-1], "*", c="tab:pink")


# ax.plot(phi, theta, '-r')
# ax.plot(phi[-1], theta[-1], '*', c='tab:pink')
# plt.show()

plt.figure()
plt.plot(loss_history)
plt.show()
