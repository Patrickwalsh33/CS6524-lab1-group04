import numpy as np
from scipy.ndimage import convolve


def disk(radius):
    ax = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(ax, ax)
    return (x**2 + y**2 <= radius**2).astype(float)


def vertical_line(length):
    return np.ones((length, 1))


def build_kernel(config):
    """Create a kernel from a YAML entry such as {shape: disk, size: 5}."""
    shapes = {'disk': disk, 'vertical': vertical_line}
    return shapes[config['shape']](config['size'])


def erode(mask, kernel):
    """A pixel stays on only if the kernel around it is entirely foreground."""
    # Outside counts as foreground so objects are not eaten at the image edge.
    response = convolve(np.asarray(mask, dtype=float), kernel,
                        mode='constant', cval=1.0)
    return response > kernel.sum() - 0.5


def dilate(mask, kernel):
    """A pixel turns on if the kernel around it touches any foreground."""
    response = convolve(np.asarray(mask, dtype=float), kernel,
                        mode='constant', cval=0.0)
    return response > 0.5


def opening(mask, kernel):
    """Remove any part of the mask that the kernel cannot fit inside."""
    return dilate(erode(mask, kernel), kernel)
