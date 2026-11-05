import numpy as np

from mypylibs.threshold import keep_largest


def _order_points(img):
    """Walk the outer border clockwise using Moore's 8-neighbour tracing."""
    border = keep_largest(np.asarray(img) > 0.5)
    if np.count_nonzero(border) < 2:
        raise ValueError('Need at least two connected edge pixels to trace.')
    padded = np.pad(border, 1)
    start = tuple(np.argwhere(padded)[0])
    current = start
    backtrack = (start[0], start[1] - 1)
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, 1),
                  (1, 1), (1, 0), (1, -1), (0, -1))
    points = [start]
    first_next = None
    # A thin branch can be visited from both sides. Stop only when the
    # initial directed step repeats, rather than at the first start visit.
    for _ in range(8 * np.count_nonzero(border)):
        offset = (backtrack[0] - current[0], backtrack[1] - current[1])
        begin = directions.index(offset)
        for step in range(8):
            index = (begin + step) % 8
            dy, dx = directions[index]
            neighbour = (current[0] + dy, current[1] + dx)
            if padded[neighbour]:
                break
        if current == start and neighbour == first_next:
            # Coordinates were (row, column), including the padding.
            return np.asarray(points, dtype=float)[:, ::-1] - 1
        if first_next is None:
            first_next = neighbour
        dy, dx = directions[(index - 1) % 8]
        backtrack = (current[0] + dy, current[1] + dx)
        current = neighbour
        points.append(current)
    raise ValueError('Could not close the edge trace; check the boundary mask.')


def _resample(points, num_samples):
    """Sample equal arc-length intervals, including the closing segment."""
    if not np.array_equal(points[-1], points[0]):
        points = np.vstack((points, points[0]))
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_dist[-1]
    if total_length == 0:
        raise ValueError('The boundary must have a non-zero length.')
    # Do not duplicate the start: these samples will form a periodic signal.
    target_dists = np.linspace(0, total_length, num_samples, endpoint=False)
    x_resampled = np.interp(target_dists, cumulative_dist, points[:, 0])
    y_resampled = np.interp(target_dists, cumulative_dist, points[:, 1])
    return x_resampled, y_resampled


def calculate(img, num_samples=500):
    """Trace the largest edge component and return floating-point x and y."""
    if not isinstance(num_samples, (int, np.integer)) or num_samples < 2:
        raise ValueError('num_samples must be an integer of at least 2.')
    return _resample(_order_points(img), num_samples)
