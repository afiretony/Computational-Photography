# %%
import cv2
import numpy as np
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from cp_hw2 import readHDR, writeHDR, lRGB2XYZ, XYZ2lRGB, xyY_to_XYZ
from gamma_encoding import gamma_encoding

K = 0.03
B = 10


def tone_mapping(HDR, K, B):
    """
    Implementation of photographic tonemapping

    inputs:
        HDR: [HxWx3] HDR images with 3 channels
        K : Key
        B : Burn

    output:
        tone_mapped: resulting image
    """
    H, W = HDR.shape[:2]
    N = H * W
    EPLISON = 1e-4
    tone_mapped = None
    I_flat = HDR.reshape(-1, 3)

    I_m = np.exp(1 / N * np.log(np.sum(I_flat + EPLISON, 0)))

    I_ = K / I_m * I_flat

    I_white = B * np.max(I_, 0)
    I_TM = I_ * (1 + I_ / I_white**2) / (1 + I_)

    return I_TM.reshape(H, W, 3)


def XYZ2xyY(XYZ):
    X, Y, Z = XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2]
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)

    return np.dstack((x, y, Y))


def luminance(RGB_HDR):
    XYZ = lRGB2XYZ(RGB_HDR)

    xyY = XYZ2xyY(XYZ)
    x, y, Y = xyY[:, :, 0], xyY[:, :, 1], xyY[:, :, 2]
    Y = Y * 0.05
    XYZ_ = xyY_to_XYZ(x, y, Y)

    RGB_ = XYZ2lRGB(np.dstack(XYZ_))

    return RGB_


HDR = readHDR("RAW_linear_optimal.hdr")

tone_mapped = tone_mapping(HDR, K, B).clip(0, 1)
plt.imsave("kitchen_k{}_b{}.png".format(K, B), gamma_encoding(tone_mapped))
plt.imshow(gamma_encoding(tone_mapped))
# HDR = np.clip(HDR, 0, 10)
# luminance = luminance(HDR)
# plt.imsave("luminance.png", gamma_encoding(luminance))

# %%
# plt.imshow(gamma_encoding(luminance))
# %%
