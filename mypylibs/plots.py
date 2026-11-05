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
