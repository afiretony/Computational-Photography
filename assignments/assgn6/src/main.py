import os
import numpy as np
from shadow_edge_time import *
from utils import *
from reconstruction import reconstruction
from shadow_plane import compute_shadow_plane

if __name__ == "__main__":

    baseDir = "../data"  # data directory
    objName = "green"  # object name (should correspond to a dir in data)
    seqName = "v1"  # sequence name (subdirectory of object)
    calName = "calib-cup"  # calibration sourse (also a dir in data)
    image_ext = "jpg"  # file extension for images
    obj_dir = os.path.join(baseDir, objName, seqName)
    cal_dir = os.path.join(baseDir, calName)

    # begin = 60  # begin of shadow index
    # end = 135  # end of shadow index
    # ul = (320, 300)
    # lr = (640, 820)
    # v_row_range = (0, 300)  # vertical row range
    # v_col_range = (230, 790)  # vertical column range
    # h_row_range = (660, 720)  # horizontal row range
    # h_col_range = (230, 790)  # horizontal column range
    # num_shadow = 32  # number of shadow images

    # begin = 65  # begin of shadow index
    # end = 108  # end of shadow index
    # ul = (147, 430)
    # lr = (550, 828)
    # v_row_range = (50, 130)  # vertical row range
    # v_col_range = (414, 800)  # vertical column range
    # h_row_range = (600, 700)  # horizontal row range
    # h_col_range = (370, 863)  # horizontal column range
    # num_shadow = 32  # number of shadow images

    # begin = 40  # begin of shadow index
    # end = 70  # end of shadow index
    # ul = (250, 616)
    # lr = (901, 1181)
    # v_row_range = (118, 248)  # vertical row range
    # v_col_range = (490, 1250)  # vertical column range
    # h_row_range = (930, 1060)  # horizontal row range
    # h_col_range = (434, 1332)  # horizontal column range
    # num_shadow = 32  # number of shadow images

    begin = 24  # begin of shadow index
    end = 52  # end of shadow index
    ul = (235, 733)
    lr = (874, 1170)
    v_row_range = (61, 216)  # vertical row range
    v_col_range = (600, 1355)  # vertical column range
    h_row_range = (926, 1012)  # horizontal row range
    h_col_range = (564, 1481)  # horizontal column range
    num_shadow = 32  # number of shadow images
    # load images
    obj_images, obj_gray_images = load_images(baseDir, objName, seqName, image_ext)
    # import matplotlib.pyplot as plt

    # fig = plt.figure()
    # plt.imshow(obj_images[0])
    # plt.show()
    # exit()
    # find the maximum intensity image
    I_max = max_img(obj_gray_images)
    # find the minimum intensity image
    I_min = min_img(obj_gray_images)
    # compute the shadow image
    I_shadow = shadow_image(I_max, I_min)
    # compute the difference image
    diff_img = difference_image(obj_gray_images, I_shadow)

    h, w = I_max.shape

    time_map = compute_shadow_time(diff_img)

    edges_vertical, edges_horizontal, mask_v, mask_h = compute_shadow_edge(
        diff_img, begin, end, v_row_range, v_col_range, h_row_range, h_col_range
    )

    # load extrinsic and intrinsic parameters
    extrinsics = np.load(os.path.join(obj_dir, "extrinsic_calib.npz"))
    intrinsics = np.load(os.path.join(cal_dir, "intrinsic_calib.npz"))
    mtx = intrinsics["mtx"]  # camera matrix
    dist = intrinsics["dist"]  # distortion coefficients

    n, P1_cam = compute_shadow_plane(
        extrinsics, mtx, dist, edges_horizontal, edges_vertical, h, w
    )
    reconstruction(
        h, time_map, n, P1_cam, ul, lr, begin, end, mtx, dist, I_max, I_min, obj_images
    )
