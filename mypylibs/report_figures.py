"""Regenerate the report's experiment figures: python -m mypylibs.report_figures

Not called by main.py, which only writes the required figures to out/.
"""
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from mypylibs import background, blur, fourier, grayscale, morphology, threshold

SIGMA = 2.0
VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]
CASES = {'Duck': 0, 'Fish': 2, 'Giraffe': 3, 'Tiger': 5}


def threshold_sweep(out_path):
    fig, axes = plt.subplots(len(CASES), len(VALUES), figsize=(7, 5.2))
    for row, (name, case) in zip(axes, CASES.items()):
        img = np.array(Image.open(f'in/{case:02}.png').convert('RGB')) / 255.0
        smooth = blur.gaussian_blur(grayscale.to_grayscale(img), SIGMA)
        for ax, value in zip(row, VALUES):
            mask = threshold.calculate(smooth, value)
            # Fixed range: an all-foreground mask would otherwise render black.
            ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'T={value}: {mask.mean():.0%}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def ground_removal(out_path):
    """Compare no opening, a disk and a vertical line on the two ground images."""
    rows = {'Tiger, T=0.2': (5, 0.2), 'Giraffe, T=0.15': (3, 0.15)}
    kernels = {'No opening': None,
               'Disk, r=5': morphology.disk(5),
               'Vertical line, L=19': morphology.vertical_line(19)}
    fig, axes = plt.subplots(len(rows), len(kernels), figsize=(7, 4.6))
    for row, (name, (case, value)) in zip(axes, rows.items()):
        img = np.array(Image.open(f'in/{case:02}.png').convert('RGB')) / 255.0
        selected = blur.gaussian_blur(background.distance(img), SIGMA) > value
        for ax, (label, kernel) in zip(row, kernels.items()):
            opened = selected if kernel is None else morphology.opening(selected, kernel)
            ax.imshow(threshold.clean(opened), cmap='gray', vmin=0, vmax=1)
            ax.set_title(label, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def pipeline(out_path, case=2):
    """Compact version of XX_transforms sized for one report column."""
    from main import outline
    stages, _, _ = outline(case)
    short = ['Original', 'Distance', 'Blurred', 'Threshold',
             'Cleaned mask', 'Sobel magnitude', 'Border']
    fig, axes = plt.subplots(2, 4, figsize=(5, 2.3))
    for ax in axes.flat:
        ax.axis('off')
    for ax, label, (_, im) in zip(axes.flat, short, stages):
        ax.imshow(im, cmap='gray')
        ax.set_title(label, fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def low_harmonics(out_path, case=3, counts=(2, 3, 5, 10, 25, 50)):
    """The giraffe rebuilt from very few harmonics, below the 5% where XX_final starts."""
    from main import outline
    _, x, y = outline(case)
    coefficients = fourier.descriptors(x, y)
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 3.2), sharex=True, sharey=True)
    for ax, count in zip(axes.flat, counts):
        rx, ry = fourier.reconstruct(coefficients, count)
        error = np.mean(np.hypot(rx - x, ry - y))
        ax.plot(np.append(x, x[0]), np.append(y, y[0]), color='lightgray', linewidth=1.5)
        ax.plot(np.append(rx, rx[0]), np.append(ry, ry[0]), color='tab:blue', linewidth=0.8)
        ax.set_ylim(y.max() + 10, y.min() - 10)
        ax.set_aspect('equal')
        ax.set_title(f'K={count}: {error:.1f} px', fontsize=8)
        ax.axis('off')
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    threshold_sweep('assets/latex/threshold_sweep.png')
    ground_removal('assets/latex/ground_removal.png')
    low_harmonics('assets/latex/low_harmonics.png')
    pipeline('assets/latex/pipeline_fish.png')
