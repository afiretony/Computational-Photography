"""
Per-frame shadow edge estimation
"""

import os
import glob
import numpy as np
from utils import read_img, bgr2gray, show_img
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_images():
    # load the images
    frog_dir = os.path.join("..", "data", "frog")
    frog_image_paths = sorted(glob.glob(os.path.join(frog_dir, "*.jpg")))
    h, w = read_img(frog_image_paths[0]).shape[:2]
    frog_images = [read_img(path) for path in frog_image_paths]
    frog_gray_images = [bgr2gray(img) for img in frog_images]
    frog_gray_images = np.stack(frog_gray_images, axis=0).astype(np.float32)
    return frog_images, frog_gray_images


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


def estimate_per_frame_zero_crossings_vertical(diff_img):
    zero_crossings = []
    for r in range(300):  # for each row
        state = 0
        for c in range(230, 790):  # for each column
            # find the zeros crossing from negative to positive
            if state == 0 and diff_img[r, c] < 0:
                state = 1
            if state == 1 and diff_img[r, c] > 0:
                state = 2
                zero_crossings.append((r, c))
                break
    return zero_crossings


def estimate_per_frame_zero_crossings_horizontal(diff_img):
    zero_crossings = []
    for r in range(660, diff_img.shape[0]):  # for each row
        state = 0
        for c in range(230, 790):  # for each column
            # find the zeros crossing from negative to positive
            if state == 0 and diff_img[r, c] < 0:
                state = 1
            if state == 1 and diff_img[r, c] > 0:
                state = 2
                zero_crossings.append((r, c))
                break
    return zero_crossings


def estimate_per_frame_shadow_edge(zero_crossings):
    # find the shadow edge
    zero_crossings = np.array(zero_crossings)
    # append coloumn of ones
    zero_crossings = np.hstack((zero_crossings, np.ones((zero_crossings.shape[0], 1))))
    u, s, vh = np.linalg.svd(zero_crossings, full_matrices=False, compute_uv=True)
    edge = vh[-1, :]
    return edge


def overlay_shadow_edge(I, edge, filename):
    # overlay the shadow edge on the image
    # find the points on the edge
    points = []
    for r in range(I.shape[0]):  # for each column
        c = int((-edge[2] - edge[0] * r) / edge[1])
        points.append((r, c))
    points = np.array(points)
    # overlay the points on the image
    plt.imshow(I)
    plt.plot(points[:, 1], points[:, 0], "r")
    plt.savefig(filename)
    plt.show()
    return


def display_zero_crossings(I, zero_crossings):
    # display the zero crossings
    plt.imshow(I)
    plt.plot(np.array(zero_crossings)[:, 1], np.array(zero_crossings)[:, 0], "r.")
    plt.show()
    return


def main():
    frog_images, frog_gray_images = load_images()
    # find the maximum intensity image
    I_max = max_img(frog_gray_images)
    # find the minimum intensity image
    I_min = min_img(frog_gray_images)
    # compute the shadow image
    I_shadow = shadow_image(I_max, I_min)
    # compute the difference image
    diff_img = difference_image(frog_gray_images, I_shadow)

    edges_vertical = []
    edges_horizontal = []
    begin = 60
    end = 135

    for i in tqdm(range(begin, end), desc="Processing"):
        # find the zeros crossing from negative to positive
        zero_crossings = estimate_per_frame_zero_crossings_vertical(diff_img[i])
        # find the shadow edge
        edges_vertical.append(estimate_per_frame_shadow_edge(zero_crossings))

        # find the zeros crossing from negative to positive
        zero_crossings = estimate_per_frame_zero_crossings_horizontal(diff_img[i])
        # find the shadow edge
        edges_horizontal.append(estimate_per_frame_shadow_edge(zero_crossings))

    edges_vertical = np.vstack(edges_vertical)
    edges_horizontal = np.vstack(edges_horizontal)

    return edges_vertical, edges_horizontal


def shadow_time():
    """
    compute and display shadow time plot
    """
    frog_images, frog_gray_images = load_images()
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
        timestamp = (i - begin) / (end - begin) * 32
        time_map[np.bitwise_and(diff_img[i] < 0, time_map == 0)] = timestamp

    time_map = time_map.astype(np.uint8)
    plt.imshow(time_map, cmap="jet", interpolation=None)
    plt.show()


if __name__ == "__main__":

    edges_vertical, edges_horizontal = main()
    np.save("edges_vertical.npy", edges_vertical)
    np.save("edges_horizontal.npy", edges_horizontal)

    shadow_time()
