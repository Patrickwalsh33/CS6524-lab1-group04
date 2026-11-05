import matplotlib.pyplot as plt

def save_mosaic(stages, out_path):
    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4))
    for ax, (title, im) in zip(axes, stages):
        ax.imshow(im, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
