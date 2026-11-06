import numpy as np


def estimate(img):
    """Median colour of the outer edge of the image."""
    edge = np.concatenate((img[0], img[-1], img[:, 0], img[:, -1]))
    return np.median(edge, axis=0)


def distance(img):
    """Colour distance from the background, scaled to the range 0-1."""
    return np.sqrt(np.sum((img - estimate(img))**2, axis=-1) / 3)
