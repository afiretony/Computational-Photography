import matplotlib.pyplot as plt
from matplotlib.pyplot import ginput
import time
import glob
import os
from skimage import io
import cv2
import numpy as np
from tqdm import tqdm


def read_image(filename):
    im = io.imread(filename)
    return im


def read_tif(filename, downsample=1):
    im = plt.imread(filename)
    return im[::downsample, ::downsample]


def show_im(im):
    plt.imshow(im)
    plt.show()


def save_im(filename, im):
    """assume im in float, [0, 1]"""

    plt.imsave(filename, np.clip(im, 0, 1))
