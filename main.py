
import yaml
import numpy as np

from PIL import Image
from os import path as opath
from mypylibs import trajectory, background, plots, blur, threshold, edges, morphology, fourier

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'

def outline(case: int):
    """Run the segmentation for one image; return the stages and the resampled outline."""
    img = np.array(Image.open(opath.join(ROOT_IN, f'{case:02}.{EXT}')).convert('RGB')) / 255.0
    with open(opath.join(ROOT_IN, f'{case:02}.yaml')) as f:
        params = yaml.safe_load(f)

    distance = background.distance(img)
    blurred = blur.gaussian_blur(distance, params['sigma'])
    selected = blurred > params['threshold']
    stages = [('Original', img),
              ('Distance from background colour', distance),
              (f'Gaussian blur (σ={params["sigma"]})', blurred),
              (f'Threshold > {params["threshold"]}', selected)]

    if 'opening' in params:
        opening = params['opening']
        selected = morphology.opening(selected, morphology.build_kernel(opening))
        stages.append((f'Opening ({opening["shape"]}, size {opening["size"]})', selected))

    mask = threshold.clean(selected, fill=params.get('fill_holes', True))
    magnitude, border = edges.calculate(mask, params.get('edge_threshold', 0.1))
    stages += [('Largest blob, holes filled', mask),
               ('Sobel gradient magnitude', magnitude),
               ('Thin Sobel border', border)]

    x, y = trajectory.calculate(border, params.get('num_samples', 500))
    return stages, x, y


def main(case: int):
    stages, x, y = outline(case)
    plots.save_mosaic(stages, opath.join(ROOT_OUT, f'{case:02}_transforms.{EXT}'))
    plots.save_trajectory(x, y, opath.join(ROOT_OUT, f'{case:02}_trajectory.{EXT}'))

    coefficients = fourier.descriptors(x, y)
    reconstructions = []
    for percent in range(5, 101, 5):
        count = round(percent / 100 * len(coefficients))
        rx, ry = fourier.reconstruct(coefficients, count)
        error = np.mean(np.hypot(rx - x, ry - y))
        reconstructions.append((f'{percent}% (K={count}), error {error:.2f} px', rx, ry))
    plots.save_reconstructions(x, y, reconstructions,
                               opath.join(ROOT_OUT, f'{case:02}_final.{EXT}'))

if __name__ == '__main__':
    case = 0
    while True:
        try:
            main(case)
        except FileNotFoundError:
            break
        case += 1
