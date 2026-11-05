import numpy as np

def to_grayscale(img):
    weights = np.array([0.333, 0.333, 0.333])
    return img @ weights
