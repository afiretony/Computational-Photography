# %%
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
    images = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    num_images = len(images)
    h, w = read_tif(images[0], downsample).shape[:2]
    mask = np.ones((h, w))

    if os.path.exists(os.path.join(image_dir, "mask.png")):
        mask = read_mask(os.path.join(image_dir, "mask.png"), downsample)
        print("mask shape: ", mask.shape)

    I = np.zeros((num_images, h * w), dtype=np.float32)  # illuminance channel
    for i in range(num_images):
        im_path = images[i]
        RGB = read_tif(im_path, downsample)
        XYZ = lRGB2XYZ(RGB)
        # apply mask
        XYZ = XYZ * mask[:, :, np.newaxis]
        I[i] = XYZ[:, :, 1].flatten()
    I += 0.01
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
    # print(I.shape)
    U, s, Vh = np.linalg.svd(I, full_matrices=False)

    S = np.diag(s)
    L = U[:, :3].T  # 3x7
    B = S[:3, :3] @ Vh[:3, :]  # 3xP
    # Q = np.diag([1, 1, -1])
    # Q = np.random.rand(3, 3)
    # for q in Q:
    #     print(q)
    # B = np.linalg.inv(Q).T @ B

    G = np.diag([1.0, 1.0, 1.0])

    # G[2, 0] = 0.0
    # G[2, 1] = 0.0
    # G[2, 2] = 0.01

    # print(G)
    # old_B = B
    B = np.linalg.inv(G).T @ B
    # B[2, :] *= 10

    N = B / np.linalg.norm(B, axis=0, keepdims=True)  # 3xP

    a = np.linalg.norm(B, axis=0, keepdims=True)

    return B, N, a, L


def visualize_normal_map(N, h, w):
    N = N.T.reshape((h, w, 3))
    # normalize to [0, 1]
    N = (N + 1) / 2
    plt.imshow(N)
    plt.show()


def save_normal_map(N, h, w, save_path):
    """
    save normal map
    input:
        N: normal map
        h, w: height and width of the image
        save_path: path to save normal map
    """
    N = N.T.reshape((h, w, 3))
    N = (N + 1) / 2
    N = (N * 255).astype(np.uint8)
    plt.imsave(save_path, N)


def visualize_albedo_map(a, h, w):
    a = a.T.reshape((h, w))
    plt.imshow(a, cmap="gray")
    plt.show()


def save_albedo_map(a, h, w, save_path):
    """
    save albedo map
    input:
        a: albedo map
        h, w: height and width of the image
        save_path: path to save albedo map
    """
    a = a.T.reshape((h, w))
    # a = (a * 255).astype(np.uint8)
    plt.imsave(save_path, a, cmap="gray")


def enforce_integrability(B, h, w, sigma):
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
        B_e_blur[:, :, i] = ndimage.gaussian_filter(B_e[:, :, i], sigma=sigma)

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
    U, s, Vh = scipy.linalg.svd(A, full_matrices=False)
    x = Vh[-1, :]
    Delta = np.array([[-x[2], x[5], 1], [x[1], -x[4], 0], [-x[0], x[3], 0]])
    print(Delta)
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

    # N[:, :, 2] = np.where((N[:, :, 2] >= 0) * (N[:, :, 2] < 0.1), 1, N[:, :, 2])
    # N[:, :, 2] = np.where((N[:, :, 2] < 0) * (N[:, :, 2] > -0.1), -1, N[:, :, 2])
    # N[:, :, 2][0 < N[:, :, 2] < 0.05] = 1
    # N[:, :, 2][N[:, :, 2] < 0 and N[:, :, 2] > -0.05] = -1

    N[:, :, 2] += 2

    # for i in range(3):
    #     N[:, :, i] = ndimage.gaussian_filter(N[:, :, i], sigma=5)

    fx = N[:, :, 0] / N[:, :, 2]
    fy = N[:, :, 1] / N[:, :, 2]

    N_unnormalized = np.dstack((fx, fy, np.ones_like(fx)))
    # Z = integrate_frankot(fx, fy)
    Z = integrate_poisson(fx, fy)
    # normalize to [0, 1]
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    return np.clip(Z, 0.0, 1)


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


def new_render(B, L, h, w):
    L = L.reshape((1, 3))
    I = L @ B
    return I.reshape((h, w))


# %%
sigma = 1

I, h, w = initials("green")
# N, A = calibrate_photometric_stereo(I, h, w)
B, N, a, L = uncalibrated_photometric_stereo(I, h, w)

# visualize_normal_map(N, h, w)
# visualize_albedo_map(a, h, w)
save_normal_map(N, h, w, "fig/green_normal.png")
save_albedo_map(a, h, w, "fig/green_albedo.png")

# save_normal_map(N, h, w, "fig/calibrated_normal_map.png")
# save_albedo_map(A, h, w, "fig/calibrated_albedo_map.png")

N, A = enforce_integrability(B, h, w, sigma)
# save_normal_map(N, h, w, "fig/enforced_normal_map.png")
# A = (A - A.min()) / (A.max() - A.min())
# a = (a - a.min()) / (a.max() - a.min())

# save_albedo_map(A, h, w, "fig/enforced_albedo_map.png")

# visualize_normal_map(N, h, w)
# visualize_albedo_map(A, h, w)
Z = integrate_normal_map(N, h, w)
plt.imsave("fig/green_depth.png", Z, cmap="gray")

# plt.imsave("fig/calibrated_depth_map.png", Z, cmap="gray")

# plt.imshow(Z, cmap="gray")
# plt.show()
visualize_surface(Z)

# N, A = calibrate_photometric_stereo(I, h, w)
# visualize_normal_map(N, h, w)
# L = np.array([[0.01836523], [0.6925411], [0.32856747]])
# print(L)
# I = new_render(B, L, h, w)
# plt.imsave("fig/shoe_new_render.png", I, cmap="gray")
