import numpy as np
from scipy.ndimage import convolve


def calculate(mask, threshold=0.1):
    """Return Sobel magnitude and a thin border on the foreground side."""
    foreground = np.asarray(mask, dtype=bool)
    image = foreground.astype(float)
    # Explicit Sobel kernels; divide by 8 to keep the response scale small.
    gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float) / 8
    gy_kernel = gx_kernel.T
    gx = convolve(image, gx_kernel, mode='constant', cval=0.0)
    gy = convolve(image, gy_kernel, mode='constant', cval=0.0)
    magnitude = np.sqrt(gx**2 + gy**2)

    # Sobel responds on BOTH sides of a transition. Keep only foreground
    # pixels touching the background to obtain a one-pixel inner border.
    neighbours_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    neighbours = convolve(image, neighbours_kernel, mode='constant', cval=0.0)
    border = (magnitude > threshold) & foreground & (neighbours < 4)
    return magnitude, border
