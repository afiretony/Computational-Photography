import cv2
import numpy as np
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from cp_hw2 import readHDR, writeHDR


def white_balancing_manual(HDR, loc):
    """
    manually select the white patch and normalize all three channels
    for rggb bayer pattern only!
    """
    patch = HDR[loc[1] - 2 : loc[1] + 2, loc[0] - 2 : loc[0] + 2, :]
    R, G, B = np.mean(patch, axis=(0, 1))

    HDR[:, :, 0] *= G / R
    HDR[:, :, 2] *= G / B
    return HDR


HDR = readHDR("COLOR_CORRECTION.hdr")
corrected = white_balancing_manual(HDR, [767, 291])
writeHDR("white_balanced.hdr", corrected)
