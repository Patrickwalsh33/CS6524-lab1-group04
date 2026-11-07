import numpy as np


def descriptors(x, y):
    """Treat the closed trajectory as one period of z[n] = x[n] + i y[n]."""
    z = np.asarray(x) + 1j * np.asarray(y)
    return np.fft.fft(z)


def reconstruct(coefficients, count):
    """Inverse DFT keeping only the `count` coefficients with the largest amplitude."""
    order = np.argsort(np.abs(coefficients))[::-1]
    kept = np.zeros_like(coefficients)
    kept[order[:count]] = coefficients[order[:count]]
    z = np.fft.ifft(kept)
    return z.real, z.imag
