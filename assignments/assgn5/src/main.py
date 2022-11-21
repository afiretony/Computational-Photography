import os
import glob
import numpy as np
from utils import *
import scipy
import matplotlib.pyplot as plt
from scipy import ndimage


def initials(image_dir):
    """
    load all images in the directory, return stacked illuminance channel for each image
    input:
        image_dir: directory of images

    output:
        I: stacked illuminance channel for each image
        h, w: height and width of the image
    """
    downsample = 1
    images = sorted(glob.glob(os.path.join(image_dir, "*.tiff")))
    num_images = len(images)
    h, w = read_tif(images[0], downsample).shape[:2]

    I = np.zeros((num_images, h * w), dtype=np.float32)  # illuminance channel
    for i in range(num_images):
        im_path = images[i]
        RGB = read_tif(im_path, downsample)
        XYZ = lRGB2XYZ(RGB)
        I[i] = XYZ[:, :, 1].flatten()

    return I, h, w


def show_illumination_map(L, h, w):
    """
    show illumination map
    input:
        L: light source direction
        h, w: height and width of the image
    """
    L = L.reshape((h, w))
    plt.imshow(L)
    plt.show()


def uncalibrated_photometric_stereo(I, h, w):
    """
    uncalibrated photometric stereo
    input:
        I: stacked illuminance channel for each image
        h, w: height and width of the image

    output:
        B: pesudo normal map
        N: normal map
        a: albedo map
        L: light source direction
    """
    U, s, Vh = np.linalg.svd(I, full_matrices=False)
    S = np.diag(s)
    L = U[:, :3].T  # 3x7
    B = S[:3, :3] @ Vh[:3, :]  # 3xP
    N = B / np.linalg.norm(B, axis=0, keepdims=True)  # 3xP
    a = np.linalg.norm(B, axis=0, keepdims=True)

    return B, N, a, L


def visualize_normal_map(N, h, w):
    N = N.T.reshape((h, w, 3))
    # normalize to [0, 1]
    N = (N + 1) / 2
    plt.imshow(N)
    plt.show()


def visualize_albedo_map(a, h, w):
    a = a.T.reshape((h, w))
    plt.imshow(a, cmap="gray")
    plt.show()


def enforce_integrability(B, h, w):
    """
    enforece integrability constraint
    input:
        B: pesudo normal map (3xP)
    """
    # form three channel pseudo normal image B_e
    B_e = B.T.reshape((h, w, 3))
    # gaussian filter
    B_e_blur = np.zeros_like(B_e)
    for i in range(3):
        B_e_blur[:, :, i] = ndimage.gaussian_filter(B_e[:, :, i], sigma=50)

    # compute gradient of B_e
    B_ex = np.gradient(B_e_blur, axis=1)
    B_ey = np.gradient(B_e_blur, axis=0)

    A1 = B_e[:, :, 0] * B_ex[:, :, 1] - B_e[:, :, 1] * B_ex[:, :, 0]
    A2 = B_e[:, :, 0] * B_ex[:, :, 2] - B_e[:, :, 2] * B_ex[:, :, 0]
    A3 = B_e[:, :, 1] * B_ex[:, :, 2] - B_e[:, :, 2] * B_ex[:, :, 1]
    A4 = -B_e[:, :, 0] * B_ey[:, :, 1] + B_e[:, :, 1] * B_ey[:, :, 0]
    A5 = -B_e[:, :, 0] * B_ey[:, :, 2] + B_e[:, :, 2] * B_ey[:, :, 0]
    A6 = -B_e[:, :, 1] * B_ey[:, :, 2] + B_e[:, :, 2] * B_ey[:, :, 1]

    A = np.hstack(
        (
            A1.reshape(-1, 1),
            A2.reshape(-1, 1),
            A3.reshape(-1, 1),
            A4.reshape(-1, 1),
            A5.reshape(-1, 1),
            A6.reshape(-1, 1),
        )
    )
    U, s, Vh = scipy.linalg.svd(A)
    x = Vh[-1, :]
    Delta = np.array([[-x[2], x[5], 1], [x[1], -x[4], 0], [-x[0], x[3], 0]])
    B_enforced = np.linalg.inv(Delta) @ B
    B_enforced = np.diag((1, 1, -1)) @ B_enforced
    N_enforced = B_enforced / np.linalg.norm(B_enforced, axis=0, keepdims=True)
    A_enforced = np.linalg.norm(B_enforced, axis=0, keepdims=True)

    return N_enforced, A_enforced


def integrate_normal_map(N, h, w):
    """
    compute derivative and integrate to get depth map
    """
    # compute derivatives
    N = N.T.reshape((h, w, 3))
    N[:, :, 2] += 1e-3
    fx = N[:, :, 0] / N[:, :, 2]
    fy = N[:, :, 1] / N[:, :, 2]
    Z = integrate_poisson(fx, fy)
    # normalize to [0, 1]
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    return Z


def calibrate_photometric_stereo(I, h, w):
    """
    compute normal and albedo map using calibrated photometric stereo
    inputs:
        I: stacked illuminance channel for each image
        h, w: height and width of the image
    outputs:
        N: normal map
        A: albedo map
    """
    S = load_sources()
    S_inv = np.linalg.inv(S.T @ S) @ S.T
    B = S_inv @ I
    # B = np.diag((1, 1, -1)) @ B
    N = B / np.linalg.norm(B, axis=0, keepdims=True)
    A = np.linalg.norm(B, axis=0, keepdims=True)
    return N, A


I, h, w = initials("shoe")

B, N, a, L = uncalibrated_photometric_stereo(I, h, w)
# print(N)
# print(N.shape)
# visualize_normal_map(N, h, w)
# visualize_albedo_map(a, h, w)
N, A = enforce_integrability(B, h, w)
visualize_normal_map(N, h, w)
# visualize_albedo_map(a, h, w)
Z = integrate_normal_map(N, h, w)
plt.imshow(Z, cmap="gray")
plt.show()
# visualize_surface(Z)

# N, A = calibrate_photometric_stereo(I, h, w)
# visualize_normal_map(N, h, w)
