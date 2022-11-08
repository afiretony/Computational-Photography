import numpy as np
from utils import read_image, save_im, gamma_decoding, lRGB2XYZ
import cv2 as cv


def compute_all_in_focus(refocused_stack, sigma1, sigma2, kernel_size):
    luminance_stack = np.zeros((10, 400, 700), dtype=np.float32)
    I_low_f = np.zeros((10, 400, 700), dtype=np.float32)
    I_high_f = np.zeros((10, 400, 700), dtype=np.float32)
    w_sharpness = np.zeros((400, 700, 10), dtype=np.float32)

    for i in range(10):
        XYZ = lRGB2XYZ(gamma_decoding(refocused_stack[..., i]))
        luminance_stack[i, ...] = XYZ[..., 1]
    # print(luminance_stack.max(), luminance_stack.min())

    # conpute low frequency luminance and high frequency luminance
    for i in range(10):
        I_low_f[i, ...] = cv.GaussianBlur(
            luminance_stack[i, ...], (kernel_size, kernel_size), sigma1
        )
        I_high_f[i, ...] = luminance_stack[i, ...] - I_low_f[i, ...]

    # compute sharpness weight
    for i in range(10):
        w_sharpness[..., i] = cv.GaussianBlur(
            I_high_f[i, ...] ** 2, (kernel_size, kernel_size), sigma2
        )

    # save the weight map
    # for i in range(10):
    #     save_im("../figs/w_sharpness_{:02d}.png".format(i), w_sharpness[..., i])

    # compute the depth map
    depth = np.arange(10)
    depth_map = np.zeros((400, 700), dtype=np.float32)
    depth_map = np.sum(w_sharpness * depth, axis=-1) / (
        0.01 + np.sum(w_sharpness, axis=-1)
    )
    depth_map = (depth_map - depth_map.min()) / (
        depth_map.max() - depth_map.min()
    )  # normalize to [0, 1]

    # compute the all-in-focus image
    w_sharpness = np.stack([w_sharpness] * 3, axis=2)
    I_all_in_focus = np.zeros((400, 700, 3), dtype=np.float32)
    I_all_in_focus = np.sum(refocused_stack * w_sharpness, axis=-1) / np.sum(
        w_sharpness, axis=-1
    )
    # save_im("../figs/I_all_in_focus.png", I_all_in_focus)
    # save_im(
    #     "../figs/depth_map_sigma1_{}_sigma2_{}.png".format(sigma1, sigma2), depth_map
    # )
    return I_all_in_focus, depth_map


refocused_stack = np.load("../data/refocused_stack.npy")
all_in_focus, depth = compute_all_in_focus(refocused_stack, 0.5, 10, 17)
