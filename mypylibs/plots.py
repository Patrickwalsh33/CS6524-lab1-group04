import numpy as np
import matplotlib.pyplot as plt

def save_mosaic(stages, out_path):
    columns = min(3, len(stages))
    rows = (len(stages) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (title, im) in zip(axes.flat, stages):
        ax.imshow(im, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_trajectory(x, y, out_path):
    """Show ten stages from the starting point to the closed contour."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    margin = max(float(np.ptp(x)), float(np.ptp(y)), 1.0) * 0.05
    for ax, progress in zip(axes.flat, np.linspace(0, 1, 10)):
        count = 1 + round(progress * len(x))
        ax.plot(x_closed[:count], y_closed[:count], color='tab:blue', linewidth=1)
        ax.plot(x_closed[count - 1], y_closed[count - 1], 'o',
                color='tab:red', markersize=3)
        ax.set_xlim(x.min() - margin, x.max() + margin)
        # Image coordinates increase downwards, so keep the original orientation.
        ax.set_ylim(y.max() + margin, y.min() - margin)
        ax.set_aspect('equal')
        ax.set_title(f't/T = {progress:.2f}')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
