
import yaml
import numpy as np

from PIL import Image
from os import path as opath
from mypylibs import trajectory, background, plots, blur, threshold, edges, morphology

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'

def main(case: int):

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

    plots.save_mosaic(stages, opath.join(ROOT_OUT, f'{case:02}_transforms.{EXT}'))

    x, y = trajectory.calculate(border, params.get('num_samples', 500))
    plots.save_trajectory(x, y, opath.join(ROOT_OUT, f'{case:02}_trajectory.{EXT}'))

    # Fourier reconstruction is the next stage.
    # plt.savefig(opath.join(ROOT_OUT, f'{case:02}_final.{EXT}'))

if __name__ == '__main__':
    case = 0
    while True:
        try:
            main(case)
        except FileNotFoundError:
            break
        case += 1
