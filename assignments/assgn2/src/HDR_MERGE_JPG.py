# Chenhao Yang @ 2022.10.04
# To merge JPG images for HDR images, consider the following steps:
# 1. Compute NON-LINEAR function g that maps non linear pixel value of JPG pixel values to linear scale
# 2. Linearize the image
# 3. Merge exposure stack to HDR image

import argparse
import numpy as np
from linearization import compute_g, linearize
from weight import weight

from utils import prepare_image_stack, writeHDR, reconstruct_image
from gamma_encoding import gamma_encoding


def HDR_LINEAR_MERGE(Z_LDR, Z_linear, Z_MIN, Z_MAX, B, weight_type="uniform"):
    """
    inputs:
        Z_LDR: [i x j] i if the pixel values of pixel location number i in flattened image j
        Z_linear: linearized pixel value
        B: [i x j] is the delta t, or log shutter speed

    outputs:
        M : merged result

    """

    M = None
    w = weight(Z_LDR / 255.0, weight_type, Z_MIN, Z_MAX, B)
    I = np.divide(Z_linear * w, B)
    M = np.sum(I, -1) / np.sum(w, -1)

    return np.nan_to_num(M)


def HDR_LOG_MERGE(Z_LDR, Z_linear, Z_MIN, Z_MAX, B, weight_type="uniform"):
    """
    inputs:
        Z_LDR: [i x j] i if the pixel values of pixel location number i in flattened image j
        Z_linear: linearized pixel value
        B: [i x j] is the delta t, or log shutter speed

    outputs:
        M : merged result

    """
    EPSILON = 0.001
    w = weight(Z_LDR / 255.0, weight_type, Z_MIN, Z_MAX, B)
    M = None
    I = w * (np.log(Z_linear + EPSILON) - np.log(B))
    M = np.exp(np.sum(I, -1) / np.sum(w, -1))

    return np.nan_to_num(M)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HDR merge for JPG files")

    parser.add_argument("--scene", type=str)
    parser.add_argument(
        "--ds_g", type=int, default=200, help="downsample rate for computing g"
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=5,
        help="downsample rate for computing HDR image",
    )
    parser.add_argument("--weight", type=str, default="uniform")
    parser.add_argument("--z_min", type=float, default=0.05)
    parser.add_argument("--z_max", type=float, default=0.95)
    parser.add_argument(
        "--lamb", type=float, default=100, help="smoothness term for computing g"
    )
    parser.add_argument(
        "--merge_mode",
        type=str,
        default="linear",
        help="Merge into image using Linear or logarithmic",
    )

    args = parser.parse_args()

    # compute g and linearize
    Z_JPG, h_, w_, c_ = prepare_image_stack(
        args.scene, downsample=args.ds_g, type="jpg"
    )

    # B = np.ones_like(Z_JPG) * np.arange(0, 16)
    B = np.ones_like(Z_JPG) * np.arange(0, 8)  # for my own
    B = 1 / 4096 * 2**B

    g, logL = compute_g(
        Z_JPG, B, lamb=args.lamb, w_type=args.weight, Z_MIN=args.z_min, Z_MAX=args.z_max
    )

    # prepare data for merging
    Z_LDR, h_, w_, c_ = prepare_image_stack(
        args.scene, downsample=args.downsample, type="jpg"
    )
    Z_linear = linearize(Z_LDR, g)
    # B = np.ones_like(Z_LDR) * np.arange(0, 16)
    B = np.ones_like(Z_LDR) * np.arange(0, 8)  # for my own

    B = 1 / 4096 * 2**B
    if args.merge_mode == "linear":
        HDR_ = HDR_LINEAR_MERGE(Z_LDR, Z_linear, args.z_min, args.z_max, B, args.weight)
    elif args.merge_mode == "log":
        HDR_ = HDR_LOG_MERGE(Z_LDR, Z_linear, args.z_min, args.z_max, B, args.weight)
    HDR = reconstruct_image(HDR_, h_, w_, c_)
    writeHDR("JPG_{}_{}.hdr".format(args.merge_mode, args.weight), HDR)
