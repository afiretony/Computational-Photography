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


def writeHDR(name, data):
    # flip from rgb to bgr for cv2
    cv2.imwrite(name, data[:, :, ::-1].astype(np.float32))


def readHDR(name):
    raw_in = cv2.imread(name, flags=cv2.IMREAD_ANYDEPTH)
    # flip from bgr to rgb
    return raw_in[:, :, ::-1]


def prepare_image_stack(path_to_dir, downsample=200, type="tiff", channel=None):
    """
    list and read all images in a folder, with downsampling, returns image stack
    note that this function uses skimage.io to read the jpg / tiff file and opencv2
    for reading .hdr file

    inputs:
        path_to_dir : path to the image folder
        downsample  : down sampleing scale
        type        : type of image to read, typically jpg, tiff or hdr
        channel     : use perticular channels [R,G,B]
    output:
        Z           : stacked images [HWC x N]
    """

    path_to_images = sorted(glob.glob(os.path.join(path_to_dir, "*.{}".format(type))))
    N = len(path_to_images)
    print(path_to_images)
    if channel:
        channel_map = {"R": 0, "G": 1, "B": 2}
        sample = read_image(path_to_images[0])[
            ::downsample, ::downsample, channel_map[channel]
        ]
        H, W = sample.shape
        C = 1

        Z = np.zeros((H * W, N))
        for i in tqdm(range(N)):
            im = read_image(path_to_images[i])[
                ::downsample, ::downsample, channel_map[channel]
            ].flatten()
            Z[:, i] = im
    else:
        sample = read_image(path_to_images[0])[::downsample, ::downsample]
        H, W, C = sample.shape

        Z = np.zeros((H * W * C, N))
        for i in tqdm(range(N)):
            im = read_image(path_to_images[i])[::downsample, ::downsample].flatten()
            Z[:, i] = im

    if type in ("tiff", "jpg"):
        Z = Z.astype(int)

    print("Stack shape: ", Z.shape)
    return Z, H, W, C


def reconstruct_image(Z, H, W, C):
    """
    reconstruct image from flattened array
    """
    assert H * W * C == Z.shape[0]

    return Z.reshape(H, W, 3)


def plot_g(g, save=False):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(np.arange(256), g, ".")
    plt.xlabel("pixel value")
    plt.ylabel("g(x)")
    plt.title("function g")
    if save:
        plt.savefig("g.jpg")
    plt.show()
