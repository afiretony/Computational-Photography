# %%
from utils import read_tif, show_im, save_im, read_image
import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.interpolate import interpn
import gc
from gamma_encoding import gamma_encoding
from gamma_correction import gamma_correction
import logger

logger = logger.create_logger()


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

    for j in range(NB_SEGMENTS):
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


# filtered = bilateral_filtering(ambient, sigma_s, sigma_r)
# save_im("piecewise_bilateral.png", filtered)

#%%
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

    for j in range(NB_SEGMENTS):
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

    A_final = (1 - M) * A_NR * F_detail + M * A_base
    return A_final


def differene(imageA, imageB):
    # difference
    diff = imageA - imageB

    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    # normalize
    # diff = np.transpose(diff, (2, 0, 1))
    # normalized = diff / np.linalg.norm(diff, axis=0)

    # return np.transpose(normalized, (1, 2, 0))
    return diff


# filtered = joint_bilateral_filtering(ambient, flash, sigma_s, sigma_r)
# save_im("joint_bilateral.png", filtered)
ambient = read_tif("../data/lamp/lamp_ambient.tif") / 255.0
flash = read_tif("../data/lamp/lamp_flash.tif") / 255.0
downsample = 1
ambient = read_image("glass_ambient.jpg")[::downsample, ::downsample] / 255.0
flash = read_image("glass_flash.jpg")[::downsample, ::downsample] / 255.0
sigma_r = 0.08
sigma_s = 50
epsilon = 0.02
tau = 0.8
kernel = 5
A_final = detail_transfer(ambient, flash, epsilon, kernel, sigma_s, sigma_r, tau)
diff = differene(A_final, ambient)
show_im(A_final)
# save_im("detail_transfer.png", A_final)
# save_im("difference.png", diff)

# %%

# ambient = read_tif("../data/lamp/lamp_ambient.tif") / 255.0
# cv2.bilateralFilter(ambient, sigmaColor=)


# import numpy as np
# I = np.random.rand(10,20,3)
# # %%
# norm = np.linalg.norm(I, axis=-1)

# %%
