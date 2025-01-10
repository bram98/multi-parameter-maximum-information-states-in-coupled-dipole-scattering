# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:11:11 2024

@author: 5914787
"""

import fourierfun as ff
import matplotlib.pyplot as plt
import numpy as np
import scipy
# %%
from scipy import stats

# %%


def make_cross(X, Y, r=0.1, d=0.3, x0=0.0, y0=0.0):
    legV = (np.abs(X - x0) < r) & (np.abs(Y - y0) < d)
    legH = (np.abs(Y - y0) < r) & (np.abs(X - x0) < d)
    return legV | legH


def t_(X, Y, theta):
    return 1 - 0.5 * make_cross(X - theta, Y)


def measure(S, E_in, scale=1.0, nsamples=1):
    E_out = scale * S @ E_in
    I_out = np.abs(E_out) ** 2
    return stats.poisson.rvs(I_out, size=(nsamples, len(I_out))) / scale ** 2


def measure_once(S, E_in, scale=1.0):
    E_out = measure(S, E_in, scale=scale)
    E_out = np.squeeze(E_out)
    # E_out = E_out / np.sqrt(np.sum(np.abs(E_out)**2))
    return E_out


def plotvec(vec):
    plt.imshow(np.abs(vec).reshape((ny, nx)))


# %%
nx = 20
ny = 20
n = nx * ny

fr = ff.fourierrange.FourierRange2d(nx, ny, xmax=2, ymax=2)
X, Y = fr.XY_c()

cross = make_cross(X, Y)
dtheta = fr.dx * 1.5
# dtheta = 1
t0 = t_(X, Y, 0)

# t0 = np.arange(n).reshape((ny, nx))
# t1 = t0.copy()
# t1[:, 0] = 0

t1 = t_(X, Y, dtheta)
U_before = stats.unitary_group(dim=n).rvs()
U_after = stats.unitary_group(dim=n).rvs()

U_before = np.eye(n)
U_after = np.eye(n)

t0_diag = np.diag(t0.flatten())
t1_diag = np.diag(t1.flatten())
S0 = U_after @ t0_diag @ U_before
S1 = U_after @ t1_diag @ U_before
dS_dtheta = np.matrix((S1 - S0) / dtheta)
F = dS_dtheta.H @ dS_dtheta

eig, v = scipy.linalg.eigh(F)

sorted_ = np.argsort(-np.abs(eig))
eig = eig[sorted_]
v = v[:, sorted_]
# %%
plt.plot(np.abs(eig))
# %%

plt.plot(eig)
plt.show()
v_img = np.zeros((ny, nx))
for i in np.nonzero(eig)[0]:
    v_img += np.reshape(np.abs(v[:, i]), (ny, nx))
plt.imshow(v_img)
plt.title(f"$\\lambda:$ {eig[i]: .2f}")
plt.show()
# %%

E_in = np.random.rand(n)

E_in = np.array(v[:, 0]).squeeze()
E_out = measure(S0, E_in, scale=100.0)
E_out = E_out / np.sum(np.abs(E_out) ** 2)
E_out = E_out.reshape((ny, nx))

plt.imshow(E_out)

# %%
scale = 10000


def estimate(a, a0, da):
    return (a - a0) @ da / (da @ da)


def estimateI(I, a, b):
    return np.sum(I * a + b)


# %%
"""
Estimator from random
"""

est_a = np.zeros(n)
est_b = 0
for i in range(1):
    E_in = np.random.rand(n)

    I0_out = measure_once(S0, E_in, scale=scale)
    I1_out = measure_once(S1, E_in, scale=scale)

    # I0_out = np.abs(E0_out)**2
    # I1_out = np.abs(E1_out)**2

    dI_dtheta = (I1_out - I0_out) / dtheta
    a = np.zeros(len(I1_out))

    zeros = dI_dtheta == 0
    a[zeros] = 0
    a[~zeros] = dI_dtheta[~zeros] / I0_out[~zeros]
    est_a += a
    est_b += dI_dtheta


# %%
"""
Test random estimator
"""
ref_results = []
shift_results = []
for i in range(10):
    E_in = np.random.rand(n)

    I0_out = measure_once(S0, E_in, scale=scale)
    I1_out = measure_once(S1, E_in, scale=scale)

    # I0_out = np.abs(E0_out)**2
    # I1_out = np.abs(E1_out)**2

    ref_results.append(estimateI(I0_out, est_a, est_b))
    shift_results.append(estimateI(I1_out, est_a, est_b))

plt.plot(ref_results, label="ref")
plt.plot(shift_results, label="shift")
plt.legend()
# %%
"""
Estimator from Fisher
"""

est_a = np.zeros(n)
est_b = np.zeros(n)
# for i in np.nonzero(eig)[0]:
for i in range(1):
    E_in = np.array(v[:, i]).squeeze()

    I0_out = measure_once(S0, E_in, scale=scale)
    I1_out = measure_once(S1, E_in, scale=scale)

    # I0_out = np.abs(E0_out)**2
    # I1_out = np.abs(E1_out)**2

    dI_dtheta = (I1_out - I0_out) / dtheta

    J = (E_in @ F @ E_in)[0, 0]
    a = np.zeros(len(I1_out))
    nonzero = dI_dtheta != 0
    a[nonzero] = dI_dtheta[nonzero] / I0_out[nonzero] / J
    est_a += a
    est_b += dI_dtheta / J

# %%
"""
Estimator from Fisher MY VERSION
"""

est_a = np.zeros(n)
est_b = np.zeros(n)
# for i in np.nonzero(eig)[0]:
count = 0

for i in range(20):
    E_in = np.array(v[:, i]).squeeze()

    I0_out = measure_once(S0, E_in, scale=scale)
    I1_out = measure_once(S1, E_in, scale=scale)

    # I0_out = np.abs(E0_out)**2
    # I1_out = np.abs(E1_out)**2

    dI_dtheta = (I1_out - I0_out) / dtheta

    a = np.zeros(len(I0_out))
    b = np.zeros(len(I0_out))
    nonzero = dI_dtheta != 0
    a[nonzero] = 1 / dI_dtheta[nonzero]
    b[nonzero] = -I0_out[nonzero] / dI_dtheta[nonzero]
    est_a += a
    est_b += b
    count += 1
est_a /= count
est_b /= count
# %%

"""
Test Fisher estimator
"""

ref_results = []
shift_results = []
for i in range(10):
    E_in = np.random.rand(n)
    E_in = np.array(v[:, i]).squeeze()

    I0_out = measure_once(S0, E_in, scale=scale)
    I1_out = measure_once(S1, E_in, scale=scale)

    # I0_out = np.abs(E0_out)**2
    # I1_out = np.abs(E1_out)**2

    ref_results.append(estimateI(I0_out, est_a, est_b))
    shift_results.append(estimateI(I1_out, est_a, est_b))
    # ref_results.append(estimate(E0_out, est_E0_f, est_dE_f))
    # shift_results.append(estimate(E1_out, est_E0_f, est_dE_f))

ref_results = np.array(ref_results)
shift_results = np.array(shift_results)

plt.plot(ref_results, label="ref")
plt.plot(shift_results, label="shift")
plt.legend()


# %%
from numpy.linalg import norm as normy

nn = 1000
a = np.random.rand(nn)
b = a + np.random.rand(nn) * 0.01
a = E0_out.copy()
b = E1_out.copy()

a /= normy(a)
b /= normy(b)

T = np.linspace(0, 1)
da = b - a
estimations = [estimate(a + t * da, est_E0_f.copy(), da) for t in T]
plt.plot(T, estimations)
