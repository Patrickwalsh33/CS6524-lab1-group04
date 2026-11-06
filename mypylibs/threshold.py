from collections import deque

import numpy as np


def keep_largest(mask):
    """Keep the largest group of foreground pixels (8-neighbour connectivity)."""
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask)
    largest = []
    height, width = mask.shape
    for y, x in zip(*np.where(mask)):
        if visited[y, x]:
            continue
        visited[y, x] = True
        queue = deque([(y, x)])
        component = []
        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        if len(component) > len(largest):
            largest = component
    result = np.zeros_like(mask)
    if largest:
        ys, xs = np.array(largest).T
        result[ys, xs] = True
    return result


def fill_holes(mask):
    """Fill background regions that cannot reach the image border."""
    # Padding provides one connected exterior, including at image corners.
    foreground = np.pad(np.asarray(mask, dtype=bool), 1)
    exterior = np.zeros_like(foreground)
    exterior[0, 0] = True
    queue = deque([(0, 0)])
    height, width = foreground.shape
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (0 <= ny < height and 0 <= nx < width
                    and not foreground[ny, nx] and not exterior[ny, nx]):
                exterior[ny, nx] = True
                queue.append((ny, nx))
    return ~exterior[1:-1, 1:-1]


def calculate(blurred, threshold=0.5, fill=True):
    """Select dark pixels, remove smaller blobs, and optionally fill holes."""
    mask = keep_largest(np.asarray(blurred) < threshold)
    return fill_holes(mask) if fill else mask
