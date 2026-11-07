"""Bonus: an MP4 of the epicycles drawing the outline.

Needs imageio-ffmpeg (requirements.txt), which supplies the ffmpeg binary that
matplotlib writes MP4s with; the course image has no ffmpeg of its own.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection


def save_video(x, y, coefficients, out_path, circles=50, step=2, fps=25):
    """Animate every harmonic, drawing only the `circles` largest to keep rendering cheap.

    One frame per `step` samples, so 500 samples give a 10 s video at 25 fps.
    """
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

    n_samples = len(coefficients)
    frequencies = np.fft.fftfreq(n_samples, d=1 / n_samples)
    # The zero-frequency term is the centroid: it is where the chain starts, not a circle.
    order = 1 + np.argsort(np.abs(coefficients[1:]))[::-1]
    radii = np.abs(coefficients[order]) / n_samples

    frames = np.arange(0, n_samples, step)
    # terms[f, k]: the k-th largest circle's arm at frame f; their running sum is the chain.
    terms = (coefficients[order] / n_samples) * np.exp(
        2j * np.pi * np.outer(frames, frequencies[order]) / n_samples)
    chain = coefficients[0] / n_samples + np.cumsum(terms, axis=1)
    centres = np.concatenate([np.full((len(frames), 1), coefficients[0] / n_samples),
                              chain[:, :circles - 1]], axis=1)
    tips = chain[:, -1]

    unit = np.exp(2j * np.pi * np.linspace(0, 1, 33))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(np.append(x, x[0]), np.append(y, y[0]), color='lightgray', linewidth=2)
    rings = LineCollection([], colors='tab:gray', linewidths=0.5, alpha=0.6)
    ax.add_collection(rings)
    arm, = ax.plot([], [], color='tab:red', linewidth=0.8)
    path, = ax.plot([], [], color='tab:blue', linewidth=1.5)
    # Frame the outline, not the circles: the largest circles reach well past the shape.
    margin = 0.25 * max(np.ptp(x), np.ptp(y))
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.max() + margin, y.min() - margin)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Epicycles: all {n_samples} harmonics, largest {circles} circles drawn')
    fig.tight_layout()

    def draw(f):
        ring_points = centres[f, :, None] + radii[:circles, None] * unit
        rings.set_segments(np.stack([ring_points.real, ring_points.imag], axis=-1))
        links = np.append(centres[f], tips[f])
        arm.set_data(links.real, links.imag)
        path.set_data(tips[:f + 1].real, tips[:f + 1].imag)
        return rings, arm, path

    movie = animation.FuncAnimation(fig, draw, frames=len(frames), blit=True)
    movie.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1500))
    plt.close(fig)
