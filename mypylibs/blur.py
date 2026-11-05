import numpy as np
from scipy.ndimage import convolve


def gaussian_kernel(sigma):
    radius = int(np.ceil(3 * sigma))
    ax = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(ax, ax)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(img, sigma):
    kernel = gaussian_kernel(sigma)
    return convolve(img, kernel, mode='reflect')
