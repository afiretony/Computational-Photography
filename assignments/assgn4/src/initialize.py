import numpy as np
from utils import read_image
from tqdm import tqdm

im = read_image("../data/chessboard_lightfield.png")


def lightfield_rendering(im):
    """
    render the lightfield from the image
    Inputs:
        im: the input image (400*16, 700*16, 3)
    Outputs:
        lightfield: the lightfield array (u, v, s, t, c)
    """
    h, w = im.shape[:2]

    lightfield = np.zeros((16, 16, h // 16, w // 16, 3), dtype=np.uint8)

    for u in tqdm(range(16), ncols=80, desc="converting to lightfield"):
        for v in range(16):
            index_u = np.arange(15 - u, h, 16)
            index_v = np.arange(v, w, 16)
            index_map_v, index_map_u = np.meshgrid(index_v, index_u)
            lightfield[u, v, ...] = im[index_map_u, index_map_v, :]
    return lightfield


def mosaic(im, lightfield):
    """
    mosaic the lightfield to the image
    inputs:
        im: the input image (400*16, 700*16, 3)
        lightfield: the lightfield array (u, v, s, t, c)
    outputs:
        masaic: the mosaiced image
    """
    mosaic = np.zeros_like(im)
    for u in tqdm(range(16), ncols=80, desc="converting to mosaic"):
        for v in range(16):
            mosaic[
                (15 - u) * 400 : (16 - u) * 400, v * 700 : (v + 1) * 700, :
            ] = lightfield[u, v, :, :, :]
    return mosaic


lightfield = lightfield_rendering(im)
np.save("../data/lightfield.npy", lightfield)

mosaic = mosaic(im, lightfield)
np.save("../data/mosaic.npy", mosaic)
