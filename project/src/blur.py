import numpy as np
from utils import display_img
import cv2

NUM_BLURS = 30
OVERLAP = 0.0


def ApplyBlur(im, kernel_size, sigma):
    """
    Apply gaussian blur to an image
    """
    blurred = cv2.GaussianBlur(im, (kernel_size, kernel_size), sigma)
    return blurred


def truncated_tent(d, d_small, d_large, overlap):
    """
    Implementation of truncated tent function
    Used for smooth transition between blurs
    """
    eta = overlap * (d_large - d_small)
    alpha = 1.0 + 1.0 / eta * np.minimum((d - d_small), (d_large - d))
    return np.clip(alpha, 0.0, 1.0)


def blur_image(im, depth_map, d_focus, sigma):

    d_stack = np.linspace(0, 1, NUM_BLURS)
    blurred = np.zeros_like(im)

    for i in range(NUM_BLURS - 1):
        d_small = d_stack[i]
        d_large = d_stack[i + 1]

        # compute mask to apply blur
        alpha = truncated_tent(depth_map, d_small, d_large, OVERLAP)
        # display_img(alpha)

        # compute radius of blur (kernel size)
        radius = max(abs(d_focus - d_small) - 0.1, 0)

        # enforce radius to be odd and reasonable for kernel size
        radius = int(np.ceil(radius * 25 / 2.0) * 2 + 1)
        # print(radius)

        curr_blur = ApplyBlur(im, radius, sigma)
        for i in range(3):
            curr_blur[:, :, i] = curr_blur[:, :, i] * alpha

        blurred = blurred + curr_blur

    return blurred


if "__main__" == __name__:
    x = np.linspace(0, 1, 100)
    a = truncated_tent(x, 0.4, 0.6, 0.5)
    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(10, 8))
    # plt.plot(x, a)
    # plt.grid("on", which="both")
    # plt.title("Truncated Tent Function")
    # plt.show()
