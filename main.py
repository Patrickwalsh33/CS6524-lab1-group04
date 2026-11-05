
import yaml
import numpy as np

from PIL import Image
from os import path as opath
from mypylibs import trajectory, grayscale, plots, blur

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'

def main(case: int):

    img = np.array(Image.open(opath.join(ROOT_IN, f'{case:02}.{EXT}')).convert('RGB')) / 255.0
    with open(opath.join(ROOT_IN, f'{case:02}.yaml')) as f:
        params = yaml.safe_load(f)

    gray = grayscale.to_grayscale(img)
    blurred = blur.gaussian_blur(gray, params['sigma'])

    plots.save_mosaic([('Original', img),
                       ('Grayscale', gray),
                       (f'Gaussian blur (σ={params["sigma"]})', blurred)],
                      opath.join(ROOT_OUT, f'{case:02}_transforms.{EXT}'))

    # Expected outputs per case
    # plt.savefig(opath.join(ROOT_OUT, f'{case:02}_trajectory.{EXT}'))
    # plt.savefig(opath.join(ROOT_OUT, f'{case:02}_final.{EXT}'))

if __name__ == '__main__':
    case = 0
    while True:
        try:
            main(case)
        except FileNotFoundError:
            break
        case += 1
