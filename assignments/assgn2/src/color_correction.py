import cv2
import numpy as np
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time
from cp_hw2 import read_colorchecker_gm
from utils import readHDR
from gamma_encoding import gamma_encoding


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


def find_checker_pos(HDR):

    plt.imshow(HDR)
    while True:
        tellme("Select 24 location on color checker with mouse clicks")
        pts = plt.ginput(24, timeout=-1)

        tellme("Happy?")
        print(pts)

        if plt.waitforbuttonpress():
            break
    return pts


def writeHDR(filename, HDR):
    HDR = cv2.cvtColor(np.float32(HDR), cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, HDR)


def color_correction(filename, use_preset=False):
    """
    correct the color with color checker in the image

    inputs:
        HDR: HDR image that needed to be color corrected
        checker_loc_file: saved checker location json, make sure that it matches the images size

    output:
        corrected: HDR image with color corrected
    """
    HDR = readHDR(filename)

    # load ground truth
    GT_R, GT_G, GT_B = read_colorchecker_gm()
    GT_R = GT_R.reshape(-1, 1)
    GT_G = GT_G.reshape(-1, 1)
    GT_B = GT_B.reshape(-1, 1)
    GT_RGB = np.hstack((GT_R, GT_G, GT_B))

    # load checker location
    if not use_preset:
        loc = find_checker_pos(HDR)
    else:
        loc = [
            (671.9399220610522, 291.0436241610737),
            (670.2079454427367, 259.86804503139194),
            (666.7439922061053, 226.9604892833945),
            (666.7439922061053, 197.51688677202844),
            (666.7439922061053, 162.87735440571544),
            (665.0120155877896, 133.43375189434926),
            (704.8474778090497, 287.57967092444244),
            (701.3835245724184, 258.13606841307626),
            (701.3835245724184, 226.9604892833945),
            (701.3835245724184, 200.9808400086597),
            (696.1875947174714, 166.3413076423467),
            (697.919571335787, 136.89770513098063),
            (737.7550335570471, 291.0436241610737),
            (734.2910803204157, 258.13606841307626),
            (732.5591037021002, 230.42444252002588),
            (730.8271270837845, 195.78491015371276),
            (729.0951504654688, 166.3413076423467),
            (729.0951504654688, 138.6296817492962),
            (767.1986360684132, 291.0436241610737),
            (765.4666594500975, 258.13606841307626),
            (760.2707295951506, 228.6924659017102),
            (763.7346828317819, 197.51688677202844),
            (760.2707295951506, 166.3413076423467),
            (758.5387529768349, 135.16572851266494),
        ]

    # read RGB values in checker location
    checker_RGB = np.ones((24, 4))

    for i in range(24):
        patch = HDR[
            int(loc[i][1]) - 2 : int(loc[i][1]) + 2,
            int(loc[i][0]) - 2 : int(loc[i][0]) + 2,
            :,
        ]
        R, G, B = np.mean(patch, axis=(0, 1))
        checker_RGB[i] = [R, G, B, 1]

    print("CHECKER RGB: \n", checker_RGB)

    A = np.zeros((72, 12))

    for i in range(24):
        A[3 * i, 0:4] = checker_RGB[i]
        A[3 * i + 1, 4:8] = checker_RGB[i]
        A[3 * i + 2, 8:12] = checker_RGB[i]
        # b = np.vstack((b, GT_RGB[i].reshape(3,1)))
    b = GT_RGB.reshape(-1, 1)

    # print(A.shape, b.shape)
    sol = np.linalg.lstsq(A, b, rcond=None)
    transformation = sol[0].reshape(3, 4)
    print("TRANSFORMATION MATRIX: \n", transformation)

    transformed_checker = (transformation @ checker_RGB.T).T
    # print("TRANSFORMED CHECKER: \n", )
    # plot_color_checker(transformed_checker)

    h, w = HDR.shape[0], HDR.shape[1]
    Homo_HDR = HDR.reshape(-1, 3)
    Homo_HDR = np.hstack((Homo_HDR, np.ones((h * w, 1))))

    dummy = np.zeros((3, 4))
    dummy[:3, :3] = np.identity(3)
    corrected_ = transformation @ Homo_HDR.T
    # corrected_ = dummy @ Homo_HDR.T
    corrected = corrected_.T.reshape(h, w, 3)
    corrected = np.clip(corrected + 0.05, 0, 10000)

    return corrected


def plot_color_checker(checker_RGB):
    """
    plots color checker

    inputs:
        checkerRGB: [24x3] matrix
    """
    fig = plt.figure()
    ax = fig.gca()
    k = 0
    for i in range(4):
        for j in range(6):
            coord = [[j, -i], [j + 1, -i], [j + 1, -i - 1], [j, -i - 1]]
            print(coord)
            ax.add_patch(patches.Polygon(coord, facecolor=checker_RGB[k], linewidth=6))
            k += 1
    plt.xlim([0, 6])
    plt.ylim([0, -4])
    plt.gca().invert_yaxis()
    plt.show()


corrected = color_correction("../fig/JPG_linear_tent.hdr", True)
writeHDR("COLOR_CORRECTION.hdr", corrected)
