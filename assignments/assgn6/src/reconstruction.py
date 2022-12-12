import os
import numpy as np
from utils import visualize_PCD
from cp_hw6 import pixel2ray
import cv2
import math


def get_shadow_time(pixel, time_map):
    """
    compute the shadow time for a given pixel
    """
    t = time_map[pixel[0], pixel[1]]
    return t


def reconstruction(
    h,
    time_map,
    normals,
    P1_cam,
    ul,
    lr,
    begin,
    end,
    mtx,
    dist,
    I_max,
    I_min,
    obj_images,
):

    Points = []
    Colors = []
    cropped_h = -ul[0] + lr[0]
    cropped_w = -ul[1] + lr[1]

    depth_map = np.zeros((cropped_h, cropped_w), dtype=np.float32)
    ray_map = np.zeros((cropped_h, cropped_w, 3), dtype=np.float32)

    for ir, r in enumerate(range(ul[0], lr[0])):
        for ic, c in enumerate(range(ul[1], lr[1])):
            t = get_shadow_time((r, c), time_map)

            if t < begin or t >= end:
                # ignore pixels outside the shadow
                continue

            t = int(t - begin)
            # print(t)

            # get the shadow plane normal and P1_cam at that shadow time
            n = normals[t].reshape(3, 1)
            P1 = P1_cam[t].reshape(3, 1)

            # get the ray of the pixel
            ray = pixel2ray((c, h - r), mtx, dist).reshape(3, 1)

            # compute the depth of the pixel
            d = np.dot(n.T, P1) / np.dot(n.T, ray)

            depth_map[ir, ic] = d
            ray_map[ir, ic] = ray.reshape(3)

    # depth_map = cv2.GaussianBlur(depth_map, (5, 5), 50)

    for ir, r in enumerate(range(ul[0], lr[0])):
        for ic, c in enumerate(range(ul[1], lr[1])):

            d = depth_map[ir, ic]
            ray = ray_map[ir, ic].reshape(3, 1)

            if d > -6000.0 or d < -30000.0 or math.isnan(d):
                continue

            # if I_max[r, c] - I_min[r, c] < 80:
            #     continue

            # compute the 3D point
            P = d * ray
            Points.append(P)
            Colors.append(obj_images[0][r, c].astype(np.float32).reshape(3, 1) / 255.0)
            # print(d)

    Points = np.hstack(Points).T
    Colors = np.hstack(Colors).T

    visualize_PCD(Points, Colors)


if __name__ == "__main__":
    from shadow_edge_time import (
        load_images,
        max_img,
        min_img,
        difference_image,
        shadow_image,
        compute_shadow_time,
    )

    baseDir = "../data"  # data directory
    objName = "frog"  # object name (should correspond to a dir in data)
    seqName = "v1"  # sequence name (subdirectory of object)
    calName = "calib"  # calibration sourse (also a dir in data)
    image_ext = "jpg"  # file extension for images
    obj_dir = os.path.join(baseDir, objName, seqName)

    begin = 60  # begin of shadow index
    end = 135  # end of shadow index

    # load images
    frog_images, frog_gray_images = load_images(baseDir, objName, seqName, image_ext)
    # find the maximum intensity image
    I_max = max_img(frog_gray_images)
    # find the minimum intensity image
    I_min = min_img(frog_gray_images)
    # compute the shadow image
    I_shadow = shadow_image(I_max, I_min)
    # compute the difference image
    diff_img = difference_image(frog_gray_images, I_shadow)

    time_map = compute_shadow_time(diff_img)

    ul = (320, 300)
    lr = (640, 820)
    h = I_max.shape[0]
    normals = np.load(os.path.join(obj_dir, "normals.npy"))
    P1_cam = np.load(os.path.join(obj_dir, "P1_cam.npy"))
    intrinsics = np.load(os.path.join(baseDir, calName, "intrinsic_calib.npz"))
    mtx = intrinsics["mtx"]  # camera matrix
    dist = intrinsics["dist"]  # distortion coefficients

    reconstruction(
        h,
        time_map,
        normals,
        P1_cam,
        ul,
        lr,
        begin,
        end,
        mtx,
        dist,
        I_max,
        I_min,
        frog_images,
    )
