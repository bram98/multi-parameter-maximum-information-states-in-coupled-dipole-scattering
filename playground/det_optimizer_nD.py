#%%
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

#%%
nparams = 3             # number of parameters
nhilbert = 100         # number of dimensions for states
r = 2.                  # 'radius' of eigenvalues: eigenvalues of dS will tend towards semicircular distribution with radius r

seed = 3               # seed for np random number generator

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

G = np.random.normal(loc=0,
                     scale=r / np.sqrt(2 * nhilbert),
                     size=(nhilbert, nhilbert))


#%%
def J_from_F(F, v, engine=torch):
    return engine.einsum('mnij, ...i, ...j -> ...mn', F, v, v)

def det_F(F, v, engine=torch):
    return - engine.linalg.det(J_from_F(F, v, engine=engine))

def rayleigh(G, v, engine=torch):
    return engine.einsum('ij, ...i, ...j -> ...', G, v, v)

def forward(F, v):
    # with torch.no_grad():
    #     v /= torch.norm(v)
    vnorm = v / torch.norm(v)
    return det_F(F, vnorm)

def forward_curry(F):
    def _forward(v):
        return forward(F, v)
    return _forward

def forward_rayleigh(G, v):
    vnorm = v / torch.norm(v)
    return rayleigh(G, vnorm)

def snapshot(v):
    return v.detach().cpu().numpy().copy()

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

v = torch.empty(nhilbert, requires_grad=True, device='cuda')
torch.nn.init.normal_(v)
with torch.no_grad():
    v /= torch.norm(v)

Ft = torch.from_numpy(F.astype(np.float32)).cuda()
Ft.requires_grad = False

optimizer = torch.optim.Adam([v], lr=.1)

loss_history = []
best_loss_adam = np.inf
best_loss_history_adam = []
for i in range(1000):
    with torch.no_grad():
        v /= torch.norm(v)
    optimizer.zero_grad()
    # loss = forward_rayleigh(Gt, v)
    loss = forward(Ft, v)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    best_loss_adam = min(loss.item(), best_loss_adam)
    best_loss_history_adam.append(best_loss_adam)


loss_history_ADAM = np.array(loss_history)
best_loss_history_adam = np.array(best_loss_history_adam)

skip = 200
plt.plot(loss_history_ADAM[-skip:])
plt.plot(np.ones(skip)*best_loss_adam, 'r--')
plt.show()

#%%
from torchswarm.swarmoptimizer import SwarmOptimizer

v = torch.empty(nhilbert, requires_grad=True).cuda()
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
nswarm = 100
optimizer = SwarmOptimizer(nhilbert, nswarm, swarm_optimizer_type="standard", max_iterations=1, bounds=[-1,1])


optimizer.optimize(DetFisher(Ft))
# results = optimizer.run(verbosity=False)
# print(f'Best value: {results.gbest_value}')



loss_history = []
prev_loss = np.inf
curr_loss = np.inf
avg_loss_improve = None
# results = optimizer.run(verbosity=True)
for i in range(50):
    results = optimizer.run(verbosity=False)
    curr_loss = results.gbest_value
    loss_history.append(curr_loss)
    if i >= 1:
        loss_improve = np.abs(curr_loss - prev_loss)
        if avg_loss_improve is None:
            avg_loss_improve = loss_improve
        else:
            avg_loss_improve = 0.4*loss_improve + 0.6*avg_loss_improve

        print(f'{curr_loss:.6f}, {np.abs(curr_loss - prev_loss):.6f}, {avg_loss_improve:.6f}')
    prev_loss = curr_loss


plt.figure()
plt.plot(np.abs(loss_history))
plt.yscale('log')
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from torchswarm.swarmoptimizer import SwarmOptimizer

# v = torch.empty(nhilbert, requires_grad=True).cuda()
# torch.nn.init.normal_(v)

# Define a range of nswarm values using geometric spacing
nswarm_values = np.geomspace(10, 1000, num=20, dtype=int)  # 20 values logarithmically spaced
best_values = []

Ft = torch.from_numpy(F.astype(np.float32)).cuda()
Ft.requires_grad = False

class DetFisher:
    def __init__(self, F):
        self.F = F

    def evaluate(self, v):
        return forward(self.F, v[:, 0])


for nswarm in nswarm_values:
    print(f'Starting nswarm={nswarm}')
    optimizer = SwarmOptimizer(nhilbert, nswarm,
                               swarm_optimizer_type="standard",
                               max_iterations=50,
                               bounds=[-1,1])
    optimizer.optimize(DetFisher(Ft))

    prev_loss = np.inf
    curr_loss = np.inf
    avg_loss_improve = None
    results = optimizer.run(verbosity=False)
    # for i in range(50):
    #     results = optimizer.run(verbosity=False)
    #     curr_loss = results.gbest_value
    #     loss_history.append(curr_loss)
    #     if i >= 1:
    #         loss_improve = np.abs(curr_loss - prev_loss)
    #         if avg_loss_improve is None:
    #             avg_loss_improve = loss_improve
    #         else:
    #             avg_loss_improve = 0.4*loss_improve + 0.6*avg_loss_improve
    #             if avg_loss_improve < 0.005:
    #                 break

            # print(f'{curr_loss:.6f}, {np.abs(curr_loss - prev_loss):.6f}, {avg_loss_improve:.6f}')
        # prev_loss = curr_loss

    best_values.append(results.gbest_value)
    print(f'nswarm: {nswarm}, Best value: {results.gbest_value}')

# Plot results
plt.figure(figsize=(8,5))
plt.plot(nswarm_values, best_values, marker='o', linestyle='-')
plt.xscale("log")  # Log scale for better visualization
plt.xlabel("Number of Particles (nswarm)")
plt.ylabel("Best Value Found")
plt.title("Swarm Optimization Performance vs. nswarm")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

#%%
for nswarm in nswarm_values:
    print(f'Starting nswarm={nswarm}')
    optimizer = SwarmOptimizer(nhilbert, nswarm,
                               swarm_optimizer_type="standard",
                               max_iterations=50,
                               bounds=[-1,1])
    optimizer.optimize(DetFisher(Ft))

    prev_loss = np.inf
    curr_loss = np.inf
    avg_loss_improve = None
    results = optimizer.run(verbosity=False)
    # for i in range(50):
    #     results = optimizer.run(verbosity=False)
    #     curr_loss = results.gbest_value
    #     loss_history.append(curr_loss)
    #     if i >= 1:
    #         loss_improve = np.abs(curr_loss - prev_loss)
    #         if avg_loss_improve is None:
    #             avg_loss_improve = loss_improve
    #         else:
    #             avg_loss_improve = 0.4*loss_improve + 0.6*avg_loss_improve
    #             if avg_loss_improve < 0.005:
    #                 break

            # print(f'{curr_loss:.6f}, {np.abs(curr_loss - prev_loss):.6f}, {avg_loss_improve:.6f}')
        # prev_loss = curr_loss

    best_values.append(results.gbest_value)
    print(f'nswarm: {nswarm}, Best value: {results.gbest_value}')

# Plot results
plt.figure(figsize=(8,5))
plt.plot(nswarm_values, best_values, marker='o', linestyle='-')
plt.xscale("log")  # Log scale for better visualization
plt.xlabel("Number of Particles (nswarm)")
plt.ylabel("Best Value Found")
plt.title("Swarm Optimization Performance vs. nswarm")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()


#%%
from evotorch import Problem
from evotorch.algorithms import SNES, CMAES
from evotorch.logging import StdOutLogger, PandasLogger
import torch


# Define a Problem instance wrapping the function
class NormalizedProblem(Problem):
    def _transform_solution(self, solution: torch.Tensor) -> torch.Tensor:
        """Normalize solutions before evaluation."""
        return solution / torch.norm(solution, dim=-1, keepdim=True)

# Solutions have length 10
problem = NormalizedProblem(
    "min",
    forward_curry(Ft.cpu()),
    solution_length=nhilbert,
    initial_bounds=(-1, 1),
)

# Instantiate a searcher
searcher_SNES = SNES(problem, stdev_init=5)
searcher_CMAES = CMAES(problem.clone(), stdev_init=5)

# Create a logger
logger_SNES = PandasLogger(searcher_SNES, interval=1, after_first_step=False)
logger_CMAES = PandasLogger(searcher_CMAES, interval=1, after_first_step=False)

# Evolve!
print('Running SNES')
searcher_SNES.run(1000)
print('Running CMA-ES')
searcher_CMAES.run(1000)
loss_history_SNES = logger_SNES.to_dataframe()['best_eval'].to_numpy()
loss_history_CMA = logger_CMAES.to_dataframe()['best_eval'].to_numpy()

#%%
skip = -100
plt.plot(loss_history_ADAM[skip:], label='Adam')
plt.plot(best_loss_history_adam[skip:], label='Adam (best)', c='tab:blue', linestyle='--')
plt.plot(loss_history_SNES[skip:], label='SNES')
plt.plot(loss_history_CMA[skip:], label='CMA-ES')
# plt.yscale('log')
plt.legend()
plt.show()


#%%
bar_labels = ['Adam', 'SNES', 'CMA-ES']
bar_values = [-loss_history_ADAM[-1], -loss_history_SNES[-1], -best_loss_history_adam[-1]]
plt.bar(bar_labels, bar_values)
plt.ylim(np.min(bar_values) - .5, np.max(bar_values))
plt.show()

