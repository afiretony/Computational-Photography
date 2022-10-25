import os
import numpy as np
import cv2
import tqdm
from scipy.signal import convolve2d
from scipy.interpolate import interpn
import gc
from gamma_encoding import gamma_encoding
from gamma_correction import gamma_correction
import logger
import argparse

from utils import read_tif, show_im, save_im, read_image


def bilateral_filtering(I, kernel, sigma_s, sigma_r):
    """
    implementation of "piecewise bilateral" algo by Durand and Dorsey

    inputs:
        kernel: gaussian kernel size
        sigma_s: spacial
        sigma_r: intensity
    """
    logger.info("Computing piecewise bilateral filtering.")

    lamb = 0.01

    min_I = np.min(I) - lamb
    max_I = np.max(I) + lamb
    NB_SEGMENTS = int(np.ceil((max_I - min_I) / sigma_r))
    J = np.zeros((NB_SEGMENTS, *I.shape))

    logger.info("Number of segments: {}".format(NB_SEGMENTS))

    for j in tqdm.tqdm(range(NB_SEGMENTS)):
        i_j = min_I + j * (max_I - min_I) / NB_SEGMENTS
        G_j = np.exp(-((I - i_j) ** 2) / (2 * sigma_r**2))  # use gaussian as g(x)
        K_j = cv2.GaussianBlur(G_j, (kernel, kernel), sigma_s)
        H_j = G_j * I
        H_j = cv2.GaussianBlur(H_j, (kernel, kernel), sigma_s)
        J_j = H_j / K_j
        J[j, :] = J_j

    gc.collect()

    # interpn for each channel
    filtered = []

    for c in range(3):
        x, y = np.arange(I.shape[1]), np.arange(I.shape[0])
        vx, vy = np.meshgrid(x, y)

        # create 3D data
        points = (np.linspace(min_I, max_I, NB_SEGMENTS), y, x)
        values = J[:, :, :, c]
        query = (I[:, :, c].flatten(), vy.flatten(), vx.flatten())

        filtered.append(interpn(points, values, query).reshape(I.shape[0], I.shape[1]))
        gc.collect()

    return np.dstack(filtered)


def joint_bilateral_filtering(ambient, flash, kernel, sigma_s, sigma_r):
    """
    implementation of "piecewise bilateral" algo by Durand and Dorsey
    """

    logger.info("Computing joint bilateral filtering.")

    lamb = 0.01

    min_I = np.min(flash) - lamb
    max_I = np.max(flash) + lamb
    NB_SEGMENTS = int(np.ceil((max_I - min_I) / sigma_r))
    J = np.zeros((NB_SEGMENTS, *ambient.shape))

    logger.info("Number of segments: {}".format(NB_SEGMENTS))

    for j in tqdm.tqdm(range(NB_SEGMENTS)):
        i_j = min_I + j * (max_I - min_I) / NB_SEGMENTS
        G_j = np.exp(-((flash - i_j) ** 2) / (2 * sigma_r**2))  # use gaussian as g(x)
        K_j = cv2.GaussianBlur(G_j, (kernel, kernel), sigma_s)
        H_j = G_j * ambient
        H_j = cv2.GaussianBlur(H_j, (kernel, kernel), sigma_s)
        J_j = H_j / K_j
        J[j, :] = J_j

    gc.collect()

    # interpn for each channel
    filtered = []

    for c in range(3):
        x, y = np.arange(ambient.shape[1]), np.arange(ambient.shape[0])
        vx, vy = np.meshgrid(x, y)

        # create 3D data
        points = (np.linspace(min_I, max_I, NB_SEGMENTS), y, x)
        values = J[:, :, :, c]
        query = (ambient[:, :, c].flatten(), vy.flatten(), vx.flatten())

        filtered.append(
            interpn(points, values, query).reshape(ambient.shape[0], ambient.shape[1])
        )
        gc.collect()

    return np.dstack(filtered)


def computeF_detail(F, F_base, epsilon):

    F_detail = (F + epsilon) / (F_base + epsilon)

    return F_detail


def computeMask(F, A, tau):
    F_lin = gamma_correction(F)
    A_lin = gamma_correction(A)
    M = np.zeros_like(F)
    M = np.where(F_lin - A_lin <= tau, 1, 0)
    return M


def detail_transfer(ambient, flash, epsilon, kernel, sigma_s, sigma_r, tau):
    A_base = bilateral_filtering(ambient, kernel, sigma_s, sigma_r)
    F_base = bilateral_filtering(flash, kernel, sigma_s, sigma_r)
    F_detail = computeF_detail(flash, F_base, epsilon)
    M = computeMask(flash, ambient, tau)
    A_NR = joint_bilateral_filtering(ambient, flash, kernel, sigma_s, sigma_r)
    A_detail = A_NR * F_detail
    A_final = (1 - M) * A_detail + M * A_base
    return A_detail, A_final


def differene(imageA, imageB):
    # difference
    diff = imageA - imageB
    # normalize
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    return diff


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--scene", type=str, default="lamp")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--sigma_r", type=float, default=0.08)
    parser.add_argument("--sigma_s", type=float, default=50)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--kernel", type=int, default=9)

    args = parser.parse_args()

    logger = logger.create_logger()
    logger.info(args)
    data_dir = "../data"

    ambient = (
        read_image(
            os.path.join(data_dir, args.scene, "{}_ambient.jpg".format(args.scene))
        )[:: args.downsample, :: args.downsample]
        / 255.0
    )
    flash = (
        read_image(
            os.path.join(data_dir, args.scene, "{}_flash.jpg".format(args.scene))
        )[:: args.downsample, :: args.downsample]
        / 255.0
    )

    logger.info("Piecewise_bilateral")
    piecewise_bilateral = bilateral_filtering(
        ambient, args.kernel, args.sigma_s, args.sigma_r
    )
    # save_im(
    #     os.path.join(data_dir, args.scene, "kernel", "{}.jpg".format(args.kernel)),
    #     piecewise_bilateral,
    # )
    # save_im(
    #     os.path.join(data_dir, args.scene, "kernel", "D_{}.jpg".format(args.kernel)),
    #     differene(piecewise_bilateral, ambient),
    # )

    save_im(
        os.path.join(data_dir, args.scene, "piecewise.jpg"),
        piecewise_bilateral,
    )
    save_im(
        os.path.join(data_dir, args.scene, "D_piecewise.jpg"),
        differene(piecewise_bilateral, ambient),
    )

    logger.info("Joint bilateral")
    joint_bilateral = joint_bilateral_filtering(
        ambient, flash, args.kernel, args.sigma_s, args.sigma_r
    )
    save_im(
        os.path.join(data_dir, args.scene, "joint.jpg"),
        joint_bilateral,
    )
    save_im(
        os.path.join(data_dir, args.scene, "D_joint.jpg"),
        differene(joint_bilateral, piecewise_bilateral),
    )

    logger.info("Detail transfer and Shadow and specularity masking")
    A_detail, A_final = detail_transfer(
        ambient, flash, args.epsilon, args.kernel, args.sigma_s, args.sigma_r, args.tau
    )
    save_im(
        os.path.join(data_dir, args.scene, "detail.jpg"),
        A_detail,
    )
    save_im(
        os.path.join(data_dir, args.scene, "D_detail.jpg"),
        differene(A_detail, joint_bilateral),
    )
    save_im(
        os.path.join(data_dir, args.scene, "final.jpg"),
        A_final,
    )
    save_im(
        os.path.join(data_dir, args.scene, "D_final.jpg"),
        differene(A_final, A_detail),
    )

    logger.info("Complete")
