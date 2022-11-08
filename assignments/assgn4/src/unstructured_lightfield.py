import numpy as np
from scipy.signal import correlate2d
import cv2 as cv
from utils import read_image, show_im
from matplotlib import pyplot as plt
import os
import glob
from tqdm import tqdm


def focal_stack(scene):
    """
    creates a focal stack from a scene
    Output:
        images: a list of images
    """
    image_dir = os.path.join("data", scene)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    images = [read_image(path) for path in image_paths]
    return images


def gray_scale(image):
    """
    converts an image to grayscale
    """
    gray_scale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    return gray_scale


def normalized_cross_correlation(image_g, template_g):
    """
    computes the normalized cross correlation between an image and a template
    Inputs:
        image_g: a grayscale image
        image_g: a grayscale template
    Outputs:
        h: the normalized cross correlation
        r: the row index of the maximum correlation
        c: the column index of the maximum correlation
    """
    image = image_g - image_g.mean()
    template = template_g - template_g.mean()

    image_square = image**2
    template_square = template**2

    corr_numerator = correlate2d(image, template, boundary="symm", mode="same")
    corr_denominator = np.sqrt(
        correlate2d(image_square, template_square, boundary="symm", mode="same")
    )
    h = corr_numerator / corr_denominator

    r, c = np.unravel_index(np.argmax(h), h.shape)  # find the match
    return h, r, c


def select_focus_point(image, patch_size):
    """
    A simple GUI to select a template from an image
    """
    plt.imshow(image, cmap="gray")
    while True:
        plt.title("Select the focus point")
        pt = plt.ginput(1)
        plt.plot(pt[0][0], pt[0][1], "ro")
        plt.title("Press enter to confirm")
        if plt.waitforbuttonpress():
            break

    c, r = pt[0]
    print(
        "The template is a patch of size patch_size centered at ({}, {})".format(c, r)
    )

    template = image[
        int(r - patch_size // 2) : int(r + patch_size // 2),
        int(c - patch_size // 2) : int(c + patch_size // 2),
    ]
    # plt.imsave("figs/pumpkin/template.png", template, cmap="gray")
    plt.imshow(template, cmap="gray")
    plt.title("The template")
    plt.waitforbuttonpress(0)
    plt.close()

    return template


def refocus(scene, patch_size=50):
    """
    refocuses a scene
    """
    images = focal_stack(scene)
    template = select_focus_point(gray_scale(images[0]), patch_size)
    num_images = len(images)
    r0, c0 = None, None
    refocused = np.zeros_like(images[0]).astype(np.float64)
    r_ind = np.arange(images[0].shape[0])
    c_ind = np.arange(images[0].shape[1])
    c_ind, r_ind = np.meshgrid(c_ind, r_ind)

    for i in tqdm(range(num_images)):
        corr, r, c = normalized_cross_correlation(gray_scale(images[i]), template)
        if i == 0:
            r0, c0 = r, c

        # shift the image to align with the template
        shift_r = r - r0
        shift_c = c - c0
        r_shift = np.clip(r_ind + shift_r, 0, images[0].shape[0] - 1).astype(np.int16)
        c_shift = np.clip(c_ind + shift_c, 0, images[0].shape[1] - 1).astype(np.int16)
        refocused += (images[i][r_shift, c_shift] / num_images).astype(np.float64)

    refocused = np.clip(refocused, 0, 255).astype(np.uint8)
    return refocused


if __name__ == "__main__":
    refocused = refocus()

    # plt.imsave("figs/pumpkin/refocused.png", refocused, cmap="gray")
    plt.imshow(refocused)
    plt.show()
