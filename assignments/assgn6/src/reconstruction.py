import numpy as np


import os
import glob
import numpy as np
from utils import read_img, bgr2gray, show_img, visualize_PCD
import matplotlib.pyplot as plt
from tqdm import tqdm
from cp_hw6 import pixel2ray

normals = np.load("../data/frog/normals.npy")
P1_cam = np.load("../data/frog/P1_cam.npy")
intrinsics = np.load("../data/calib/intrinsic_calib.npz")
mtx = intrinsics["mtx"]  # camera matrix
dist = intrinsics["dist"]  # distortion coefficients

# load the images
frog_dir = os.path.join("..", "data", "frog")
frog_image_paths = sorted(glob.glob(os.path.join(frog_dir, "*.jpg")))
h, w = read_img(frog_image_paths[0]).shape[:2]
frog_images = [read_img(path) for path in frog_image_paths]
frog_gray_images = [bgr2gray(img) for img in frog_images]
frog_gray_images = np.stack(frog_gray_images, axis=0).astype(np.float32)


def max_img(img):
    # find the maximum intensity image
    max_img = np.max(img, axis=0)
    return max_img


def min_img(img):
    # find the minimum intensity image
    min_img = np.min(img, axis=0)
    return min_img


def difference_image(I, I_shadow):
    # compute the difference image
    diff_img = I - I_shadow
    return diff_img


def shadow_image(I_max, I_min):
    # compute the shadow image
    shadow_img = (I_max + I_min) / 2
    return shadow_img


def shadow_time():
    """
    compute and display shadow time plot
    """
    # find the maximum intensity image
    I_max = max_img(frog_gray_images)
    # find the minimum intensity image
    I_min = min_img(frog_gray_images)
    # compute the shadow image
    I_shadow = shadow_image(I_max, I_min)
    # compute the difference image
    diff_img = difference_image(frog_gray_images, I_shadow)

    begin = 0
    end = 166

    time_map = np.zeros((diff_img.shape[1], diff_img.shape[2]))
    for i in range(begin, end):
        timestamp = i
        time_map[np.bitwise_and(diff_img[i] < 0, time_map == 0)] = timestamp

    return time_map


def get_shadow_time(pixel, time_map):
    """
    compute the shadow time for a given pixel
    """
    t = time_map[pixel[0], pixel[1]]
    return t


def main():
    time_map = shadow_time()
    ul = (320, 300)
    lr = (640, 820)

    begin = 60
    end = 135

    Points = []
    Colors = []

    for r in range(ul[0], lr[0]):
        for c in range(ul[1], lr[1]):
            t = get_shadow_time((r, c), time_map)

            if t < begin or t >= end:
                # ignore pixels outside the shadow
                continue

            t = int(t - begin)

            # get the shadow plane normal and P1_cam at that shadow time
            n = normals[t].reshape(3, 1)
            P1 = P1_cam[t].reshape(3, 1)

            # get the ray of the pixel
            ray = pixel2ray((c, h - r), mtx, dist).reshape(3, 1)

            # compute the depth of the pixel
            d = np.dot(n.T, P1) / np.dot(n.T, ray)

            # compute the 3D point
            P = d * ray
            Points.append(P)
            Colors.append(frog_images[0][r, c].astype(np.float32).reshape(3, 1) / 255.0)

    Points = np.hstack(Points).T
    Colors = np.hstack(Colors).T

    visualize_PCD(Points, Colors)


main()
