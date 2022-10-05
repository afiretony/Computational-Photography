# Chenhao Yang @ 2022.10.04
# To merge RAW images for HDR images, we don't need to linearize the input because raw
# images are already linear, we just need to merge exposure stack to HDR image

import argparse
import numpy as np
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
    w = weight(Z_LDR, weight_type, Z_MIN, Z_MAX, B)
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
    w = weight(Z_LDR, weight_type, Z_MIN, Z_MAX, B)
    M = None
    I = w * (np.log(Z_linear + EPSILON) - np.log(B))
    M = np.exp(np.sum(I, -1) / np.sum(w, -1))

    return np.nan_to_num(M)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDR merge for JPG files")

    parser.add_argument("--scene", type=str, default="../data/door_stack")
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

    # prepare data for merging
    Z_LDR, h_, w_, c_ = prepare_image_stack(
        args.scene, downsample=args.downsample, type="tiff"
    )
    Z_LDR = Z_LDR / 65535.0
    Z_linear = Z_LDR
    # B = np.ones_like(Z_LDR) * np.arange(0, 16)
    # B = 1 / 2048 * 2**B
    B = np.ones_like(Z_LDR) * np.arange(0, 8)  # for my own
    B = 1 / 4096 * 2**B
    if args.merge_mode == "linear":
        HDR_ = HDR_LINEAR_MERGE(Z_LDR, Z_linear, args.z_min, args.z_max, B, args.weight)
    elif args.merge_mode == "log":
        HDR_ = HDR_LOG_MERGE(Z_LDR, Z_linear, args.z_min, args.z_max, B, args.weight)
    HDR = reconstruct_image(HDR_, h_, w_, c_)
    writeHDR("RAW_{}_{}.hdr".format(args.merge_mode, args.weight), HDR)
