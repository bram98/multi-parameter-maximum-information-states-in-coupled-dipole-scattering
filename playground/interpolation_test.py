# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:28:51 2024

@author: 5914787
"""

import fourierfun as ff
import matplotlib.pyplot as plt

# %%
import numpy as np
import scipy
from scipy.interpolate import (
    CubicHermiteSpline,
    barycentric_interpolate,
    krogh_interpolate,
    pchip_interpolate,
)

# %%
"""'
We want to sample at a specific frequency, which may lie halfway between two 
bins. If we take the fft of a sinewave that is 'incommensurate', we get spectral
leakage: the amplitude is not concentrated in a single bin, but leaks towards
other bins. The amplitude in the right bin is also lower. Can we sample in such
a way that this doesn't happen?

I define y as 
    y = exp(1j * (k + dx) * x)
I start with a k such that y is totally incommensurate, i.e. the left and right
hand side of the domain are not continuous. I then sweep dk such that y becomes
commensurate and finally non commensurate again. The result is then Fourier
transformed and ideally this should look like a delta function.

In this section I play with interpolators. 
"""
n = 50
xmax = 2

n_interp = 500

fr = ff.FourierRange(n=n, xmax=xmax)


k = 2 * np.pi + np.pi / 2
x = fr.xrange()

k_range = fr.krange()
k_interp = np.linspace(0, fr.kmax, num=n_interp, endpoint=False)

for dk in np.linspace(0, np.pi, num=50):
    y = np.exp(1j * k * x)
    y *= np.exp(1j * dk * x)

    # y *= scipy.stats.norm.pdf(x, loc=xmax/2, scale=1.)

    # y = np.fft.ifftshift(y)

    # plt.plot(x, y.real)
    # plt.show()

    y_k = np.fft.fft(y, norm="forward")
    sub = 40
    plt.plot(k_range[:sub], np.abs(y_k)[:sub], label="original")
    # plt.show()
    plt.xlim(0, 10)
    plt.ylim(0, 1.1)
    for name, interp in zip(
        ["bary", "krogh", "pchip"],
        [barycentric_interpolate, krogh_interpolate, pchip_interpolate],
    ):
        y_k_interp = interp(k_range, np.abs(y_k), k_interp)
        plt.plot(k_interp, y_k_interp, label=name)
    plt.legend()
    plt.show()


# %%
"""
In this section I play with window function. The amplitude is way off and I do
not pursue this for very long.
"""


def gauss(x, sigma=0.5):
    return np.exp(-0.5 * ((x - 0.5) / (sigma * 0.5)) ** 2)


def flat_top(x):
    a0 = 0.215
    a1 = 0.416
    a2 = 0.277
    a3 = 0.083
    a4 = 0.0069
    return (
        a0
        - a1 * np.cos(2 * np.pi * x)
        + a2 * np.cos(4 * np.pi * x)
        - a3 * np.cos(6 * np.pi * x)
        + a4 * np.cos(8 * np.pi * x)
    )


n = 200
xmax = 1

n_interp = 500

fr = ff.FourierRange(n=n, xmax=xmax)


k = 2 * np.pi + np.pi / 2
x = fr.xrange()

k_range = fr.krange()
k_interp = np.linspace(0, fr.kmax, num=n_interp, endpoint=False)

for dk in np.linspace(0, np.pi, num=50):
    y = np.exp(1j * k * x)
    y *= np.exp(1j * dk * x)

    # y *= scipy.stats.norm.pdf(x, loc=xmax/2, scale=1.)

    # y = np.fft.ifftshift(y)

    # plt.plot(x, y.real)
    # plt.show()

    y_k = np.fft.fft(y, norm="forward")
    sub = 40
    plt.plot(k_range[:sub], np.abs(y_k)[:sub], label="original")
    # plt.show()
    plt.xlim(0, 10)
    plt.ylim(0, 1.1)

    for name, window in zip(
        ["gauss", "flat-top"], [gauss(x, 1.0), flat_top(x)]
    ):
        yw = y * window
        yw = np.fft.ifftshift(yw)
        y_kw = np.fft.fft(yw, norm="forward")
        plt.plot(k_range[:sub], np.abs(y_kw)[:sub], label=name)

    plt.legend()
    plt.show()

# %%
"""
Finally I considered some 'exact' frequency approaches. We can always integrate
 with an exponential with some exact frequency. This is potentially
much slower than fft, but for a small number of samples this might actually be
faster.
"""

n = 200
xmax = 5

n_interp = 500

fr = ff.FourierRange(n=n, xmax=xmax)


k = (
    2 * np.pi + np.pi / 2
)  # wavelength of source signal, start as worst case scenario
x = fr.xrange()  # range of x-values

k_range = fr.krange()  # range of k-values
k_interp = np.linspace(0, fr.kmax, num=n_interp, endpoint=False)

phase = 0.1  # start with arbitrary phase

for dk in np.linspace(-np.pi / 4, np.pi / 4, num=50):
    y = np.exp(1j * k * x)
    y *= np.exp(1j * dk * x)

    oscillator = np.exp(-1j * k * x)
    oscillator_90 = -1j * oscillator

    I = np.trapz(oscillator * y, x=x) / xmax
    Q = np.trapz(oscillator_90 * y, x=x) * 0
    magnitude = np.sqrt(np.abs(I) ** 2 + np.abs(Q) ** 2)
    print(f"{magnitude:.4f} {dk:.2f}")

    y_k = np.fft.fft(y, norm="forward")
    # plt.plot(k_range[:sub], np.abs(y_k)[:sub], label='original')
    # plt.show()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.scatter([np.real(Q), np.real(I)], [np.imag(Q), np.imag(I)])

    plt.legend()
    plt.show()
