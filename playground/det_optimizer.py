#%%
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

#%%
nparams = 3             # number of parameters
nhilbert = 3         # number of dimensions for states
r = 2.                  # 'radius' of eigenvalues: eigenvalues of dS will tend towards semicircular distribution with radius r

seed = 20               # seed for np random number generator

np.random.seed(seed)

filename = f'dS_np-{nparams}_nh-{nhilbert}_r-{r:.4f}_seed-{seed}.npy'
filepath = r'../data/numeric_scatter_deriv/' + filename
if os.path.isfile(filepath):
    print('Found existing data for current parameters. Loading...')
    dS = np.load(filepath)
else:
    sigma = r / np.sqrt(2 * nhilbert)
    dS = np.random.normal(loc=0.,
                              scale=sigma,
                              size=(nparams, nhilbert, nhilbert))
    np.save(filepath, dS)

F = [[dS_dmu.conj().T@dS_dnu + dS_dnu.conj().T@dS_dmu for dS_dmu in dS] for dS_dnu in dS]
F = np.array(F)



#%%
def J_from_F(F, v, engine=torch):
    return engine.einsum('mnij, ...i, ...j -> ...mn', F, v, v)

def det_F(F, v, engine=torch):
    return - engine.linalg.det(J_from_F(F, v, engine=engine))

def forward(F, v):
    v = torch.nn.functional.normalize(v, dim=0)
    return det_F(F, v) + (v.dot(v) - 1)**2

#%%

# debugging purposes

F = np.zeros((2,2,3,3), dtype=int)
for i in range(2):
    for j in range(2):
        for k in range(3):
            for l in range(3):
                F[i,j,k,l] = 1000 * i + 100*j + 10 * k + l

# np.einsum('mnij,i,j', F, u, v)
#%%
def cartesian_from_spherical(phi, theta):
    return np.array([np.cos(phi)*np.sin(theta),
                     np.sin(phi)*np.sin(theta),
                     np.cos(theta)])
def spherical_from_cartesian(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return np.array([phi, theta])


def visualize_over_angles(func, F, ntheta=50, nphi=50):
    if not F.shape[2:] == (3, 3):
        raise ValueError(f'Error Hilbert space needs to be 3D for visualization. F has shape {F.shape}')

    theta = np.linspace(0, np.pi, num=ntheta)
    phi = np.linspace(0, 2 * np.pi, num=nphi)
    Phi, Theta = np.meshgrid(phi, theta)

    v = cartesian_from_spherical(Phi, Theta)
    v = np.transpose(v, axes=(1, 2, 0))

    fig, ax = plt.subplots(layout='constrained')
    img = func(F, v, engine=np)
    angle_plot = ax.imshow(img, extent=(0, 2*np.pi, 0, np.pi))
    # plt.xlim(0, 2*np.pi)
    # plt.ylim(0, np.pi)
    # ax.set_xlabel(r'$\phi$')
    # ax.set_ylabel(r'$\theta$')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('right', size='3%', pad='2%')
    fig.colorbar(angle_plot, cax=cax)
    # plt.show()
    # img = [[func(F, [np.cos(p)*np.sin(t), np.sin(p)*np.sin(t), np.cos(t)], engine=np)for t in theta] for p in phi]

    # plt.imshow(img)
    # plt.show()
    return fig, ax
# theta = np.linspace(0, np.pi, num=200)
# phi = np.linspace(0, 2 * np.pi, num=300)
# Phi, Theta = np.meshgrid(phi, theta)
# v = np.array([np.cos(Phi)*np.sin(Theta),
#               np.sin(Phi)*np.sin(Theta),
#               np.cos(Theta)])
# v = np.transpose(v, axes=(1, 2, 0))
visualize_over_angles(det_F, F, ntheta=200, nphi=200)
plt.show()

#%%

v = torch.empty(3, requires_grad=True)
Ft = torch.from_numpy(F.astype(np.float32))
Ft.requires_grad = False
torch.nn.init.normal_(v)

# v_ = torch.nn.functional.normalize(v, dim=0)

optimizer = torch.optim.Adam([v], lr=0.1)

v_history = []
v_history.append(v.detach().numpy())
loss_history = []
for i in range(10000):
    optimizer.zero_grad()
    loss = forward(Ft, v)
    loss.backward()
    optimizer.step()
    if i%100 == 0:
        v_history.append(v.detach().cpu().numpy())
        loss_history.append(loss.item())

v_history = np.array(v_history)

phi, theta = spherical_from_cartesian(*v_history.T)
phi = phi % (2*np.pi)

fig, ax = visualize_over_angles(det_F, F, ntheta=200, nphi=200)


ax.plot(phi, theta, 'or')
plt.show()

plt.figure()
plt.plot(loss_history)
plt.show()

# # Define the function to optimize
# def custom_function(x):
#     return (x - 3) ** 2 + torch.sin(5 * x)  # Example function

# # Initialize the parameter
# x = torch.tensor([0.0], requires_grad=True)

# # Define the optimizer
# optimizer = torch.optim.SGD([x], lr=0.1)

# # Optimization loop
# for i in range(100):
#     optimizer.zero_grad()  # Zero gradients
#     loss = custom_function(x)  # Compute loss
#     loss.backward()  # Compute gradients
#     optimizer.step()  # Update x

#     if i % 10 == 0:
#         print(f"Step {i}: x = {x.item()}, loss = {loss.item()}")

# print(f"Optimized x: {x.item()}")