#%%
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from timeit import default_timer as timer

#%%
from evotorch.decorators import vectorized

def make_F(dS):
    F = [[dS_dmu.conj().T@dS_dnu + dS_dnu.conj().T@dS_dmu for dS_dmu in dS] for dS_dnu in dS]
    F = np.array(F)
    return F

def J_from_F(F, v, engine=torch):
    return engine.einsum('mnij, ...i, ...j -> ...mn', F, v.conj(), v).real

def det_F(F, v, engine=torch):
    return - engine.linalg.det(J_from_F(F, v, engine=engine))

def rayleigh(G, v, engine=torch):
    return engine.einsum('ij, ...i, ...j -> ...', G, v, v)

def forward(F, v):
    # with torch.no_grad():
    #     v /= torch.norm(v)
    vnorm = v / torch.norm(v)
    return det_F(F, vnorm)

def forward_unnorm(F, v):
    return det_F(F, v)

@vectorized
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
nparams = 3             # number of parameters
nhilbert = 100         # number of dimensions for states
r = 2.                  # 'radius' of eigenvalues: eigenvalues of dS will tend towards semicircular distribution with radius r

seed = 3               # seed for np random number generator
complex_F = True

np.random.seed(seed)

filename = f'dS_np-{nparams}_nh-{nhilbert}_r-{r:.4f}_seed-{seed}_{"c" if complex_F else "r"}.npy'
filepath = r'../data/numeric_scatter_deriv/' + filename
if os.path.isfile(filepath):
    print('Found existing data for current parameters. Loading...')
    dS = np.load(filepath)
else:
    sigma = r / np.sqrt(2 * nhilbert)
    if complex_F:
        dS = ( np.random.normal(scale=sigma, size=(nparams, nhilbert, nhilbert)) +
            1j*np.random.normal(scale=sigma, size=(nparams, nhilbert, nhilbert)) )
    else:
        dS = np.random.normal(loc=0.,
                              scale=sigma,
                              size=(nparams, nhilbert, nhilbert))
    np.save(filepath, dS)


G = np.random.normal(loc=0,
                     scale=r / np.sqrt(2 * nhilbert),
                     size=(nhilbert, nhilbert))

F = make_F(dS)

#%%
dS = np.load("../data/simulation_scatter_deriv/Quad_shiftX_shiftZ_rotation.npy")

F = make_F(dS)
nparams, _, nhilbert, _ = F.shape

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
v = torch.normal(0, 1, (nhilbert,),
                 dtype=torch.complex128,
                 requires_grad=True,
                 device='cuda')
# torch.nn.init.normal_(v)
with torch.no_grad():
    v /= torch.norm(v)

Ft = torch.from_numpy(F.astype(np.complex128)).cuda()
Ft.requires_grad = False

optimizer = torch.optim.Adam([v], lr=.001)

loss_history = []
best_loss_adam = np.inf
best_loss_history_adam = []

steps = 5000
t1 = timer()
for i in range(steps):
    with torch.no_grad():
        v /= torch.norm(v)
    optimizer.zero_grad()
    loss = forward(Ft, v)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    best_loss_adam = min(loss.item(), best_loss_adam)
    best_loss_history_adam.append(best_loss_adam)



t2 = timer()
print('Adam time:', t2 - t1)
print('Learning rate:', optimizer.param_groups[0]['lr'])

loss_history_ADAM_norm = np.array(loss_history)
best_loss_history_adam_norm = np.array(best_loss_history_adam)

#####

v = torch.normal(0, 1, (nhilbert,),
                 dtype=torch.complex128,
                 requires_grad=True,
                 device='cuda')
# torch.nn.init.normal_(v)
with torch.no_grad():
    v /= torch.norm(v)

optimizer = torch.optim.Adam([v], lr=.001)

loss_history = []
best_loss_adam = np.inf
best_loss_history_adam = []

t1 = timer()
for i in range(steps):
    with torch.no_grad():
        v /= torch.norm(v)
    optimizer.zero_grad()
    loss = forward_unnorm(Ft, v)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    best_loss_adam = min(loss.item(), best_loss_adam)
    best_loss_history_adam.append(best_loss_adam)


loss_history_ADAM= np.array(loss_history)
best_loss_history_adam = np.array(best_loss_history_adam)

#%%

skip = 20
plt.figure()
plt.plot(loss_history_ADAM[skip:], c='tab:blue', label='Adam')
plt.plot(best_loss_history_adam[skip:], c='tab:blue', label='Adam (best)', linestyle='--')
plt.plot(loss_history_ADAM_norm[skip:], c='tab:orange', label='Adam normalized')
plt.plot(best_loss_history_adam_norm[skip:], c='tab:orange', label='Adam normalized (best)', linestyle='--')
plt.ticklabel_format(style='plain', useOffset=False)
plt.legend()
plt.show()


skip = -100
plt.figure()
plt.plot(loss_history_ADAM[skip:], c='tab:blue', label='Adam')
plt.plot(best_loss_history_adam[skip:], c='tab:blue', label='Adam (best)', linestyle='--')
plt.plot(loss_history_ADAM_norm[skip:], c='tab:orange', label='Adam normalized')
plt.plot(best_loss_history_adam_norm[skip:], c='tab:orange', label='Adam normalized (best)', linestyle='--')
plt.ticklabel_format(style='plain', useOffset=False)
plt.legend()
plt.show()

#%%
import evotorch
from evotorch import Problem
from evotorch.algorithms import SNES, CMAES
from evotorch.logging import StdOutLogger, PandasLogger
import torch


# Define a Problem instance wrapping the function
class NormalizedProblem(Problem):
    def _transform_solution(self, solution: torch.Tensor) -> torch.Tensor:
        """Normalize solutions before evaluation."""
        return solution / torch.norm(solution, dim=-1, keepdim=True)

sigma_list = []

# Solutions have length 10
problem = NormalizedProblem(
    "min",
    forward_curry(Ft.cpu()),
    solution_length=nhilbert,
    initial_bounds=(-1, 1),
    device='cpu'
)

# Instantiate a searcher
searcher_SNES = SNES(problem.clone(), stdev_init=1)
searcher_CMAES = CMAES(problem.clone(), stdev_init=1, separable=False)
searcher_CMAES_sep = CMAES(problem.clone(), stdev_init=1, separable=True)

# Create a logger
logger_SNES = PandasLogger(searcher_SNES, interval=1, after_first_step=False)
logger_CMAES = PandasLogger(searcher_CMAES, interval=1, after_first_step=False)
logger_CMAES_sep = PandasLogger(searcher_CMAES_sep, interval=1, after_first_step=False)

# steps = 2000

# Evolve!

t1 = timer()
print('Running SNES')
searcher_SNES.run(steps)
t2 = timer()
print('Running CMA-ES')
searcher_CMAES.run(steps)
t3 = timer()
print('Running CMA-ES sep.')
searcher_CMAES_sep.run(steps)
t4 = timer()
print('SNES time: ', t2 - t1)
print('CMA-ES time: ', t3 - t2)
print('CMA-ES sep. time: ', t4 - t3)

loss_history_SNES = logger_SNES.to_dataframe()['best_eval'].to_numpy()
loss_history_CMA = logger_CMAES.to_dataframe()['best_eval'].to_numpy()
loss_history_CMA_sep = logger_CMAES_sep.to_dataframe()['best_eval'].to_numpy()


#%%
skip = -100
if skip >= 0:
    step_range = np.arange(len(loss_history_SNES))
else:
    step_range = np.arange(skip, 0)
plt.plot(step_range, loss_history_ADAM[skip:], label='Adam', c='tab:red')
plt.plot(step_range, best_loss_history_adam[skip:], label='Adam (best)', c='tab:red', linestyle='--')
# plt.plot(step_range, loss_history_SNES[skip:], label='SNES')
plt.plot(step_range, loss_history_CMA[skip:], label='CMA-ES')
plt.plot(step_range, loss_history_CMA_sep[skip:], label='CMA-ES sep')
plt.xlabel('Iterations')
plt.ticklabel_format(style='plain', useOffset=False)
plt.legend()
plt.show()


 #%%
bar_labels = ['Adam', 'SNES', 'CMA-ES', 'CMAS-ES sep.']
bar_values = [-best_loss_history_adam[-1],
              -loss_history_SNES[-1],
              -loss_history_CMA[-1],
              -loss_history_CMA_sep[-1],
              ]
plt.bar(bar_labels, bar_values)
plt.ylim(np.min(bar_values) - .1, np.max(bar_values))
plt.show()

