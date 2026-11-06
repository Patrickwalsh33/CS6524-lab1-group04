"""Regenerate the report's threshold figure: python -m mypylibs.report_figures

Not called by main.py, which only writes the required figures to out/.
"""
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from mypylibs import blur, grayscale, threshold

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


if __name__ == '__main__':
    threshold_sweep('assets/latex/threshold_sweep.png')
