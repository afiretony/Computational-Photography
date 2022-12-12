import os
import glob
import numpy as np
from cp_hw6 import pixel2ray
from utils import read_img


def compute_P_cam(edges, row, R, T, mtx, dist, h):
    """
    computes the 3D points in camera frame
    """
    cs = ((-edges[:, 2] - edges[:, 0] * row) / edges[:, 1]).reshape(
        -1, 1
    )  # column coordinates
    p_cam = np.hstack((cs, np.ones_like(cs) * (h - row)))  # images coordinates
    rays = pixel2ray(p_cam, mtx, dist).reshape(-1, 3)  # N x 3 rays in camera frame
    t = T[2] / (rays @ R[:, 2])  # intersection with plane coordinates Z = 0
    P_cam = t.reshape(-1, 1) * rays  # N x 3  points in camera frame
    return P_cam


def compute_shadow_plane(extrinsics, mtx, dist, edges_h, edges_v, h, w):
    """
    compute the shadow plane
    inputs:
        extrinsics: extrinsic calibration parameters
        mtx: camera matrix
        dist: distortion coefficients
        edges_h: horizontal edges
        edges_v: vertical edges
        h: image height
        w: image width
    outputs:
        n: normal vector of the shadow plane
    """
    R_h = extrinsics["rmat_h"]
    T_h = extrinsics["tvec_h"]
    R_v = extrinsics["rmat_v"]
    T_v = extrinsics["tvec_v"]

    h_row_1 = 650
    h_row_2 = 750
    v_row_1 = 100
    v_row_2 = 200

    P1_cam = compute_P_cam(edges_h, h_row_1, R_h, T_h, mtx, dist, h)
    P2_cam = compute_P_cam(edges_h, h_row_2, R_h, T_h, mtx, dist, h)
    P3_cam = compute_P_cam(edges_v, v_row_1, R_v, T_v, mtx, dist, h)
    P4_cam = compute_P_cam(edges_v, v_row_2, R_v, T_v, mtx, dist, h)

    n = np.cross(P1_cam - P2_cam, P3_cam - P4_cam)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)

    return n, P1_cam


def load_calibration(obj_dir, cal_dir, image_ext):
    extrinsics = np.load(os.path.join(obj_dir, "extrinsic_calib.npz"))
    intrinsics = np.load(os.path.join(cal_dir, "intrinsic_calib.npz"))
    edges_h = np.load(os.path.join(obj_dir, "edges_horizontal.npy"))
    edges_v = np.load(os.path.join(obj_dir, "edges_vertical.npy"))
    mtx = intrinsics["mtx"]  # camera matrix
    dist = intrinsics["dist"]  # distortion coefficients
    image_paths = sorted(glob.glob(os.path.join(obj_dir, "*.{}".format(image_ext))))
    h, w = read_img(image_paths[0]).shape[:2]

    return extrinsics, mtx, dist, edges_h, edges_v, h, w


if __name__ == "__main__":
    # Input data locations
    baseDir = "../data"  # data directory
    objName = "frog"  # object name (should correspond to a dir in data)
    seqName = "v1"  # sequence name (subdirectory of object)
    calName = "calib"  # calibration sourse (also a dir in data)
    image_ext = "jpg"  # file extension for images

    obj_dir = os.path.join(baseDir, objName, seqName)
    cal_dir = os.path.join(baseDir, calName)
    extrinsics, mtx, dist, edges_h, edges_v, h, w = load_calibration(
        obj_dir, cal_dir, image_ext
    )
    n, P1_cam = compute_shadow_plane(extrinsics, mtx, dist, edges_h, edges_v, h, w)
    np.save(os.path.join(obj_dir, "normals.npy"), n)
    np.save(os.path.join(obj_dir, "P1_cam.npy"), P1_cam)
