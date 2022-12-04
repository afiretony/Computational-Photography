import os
import glob
import numpy as np
from cp_hw6 import pixel2ray
from utils import show_img, read_img, bgr2gray

extrinsics = np.load("../data/frog/v1/extrinsic_calib.npz")
intrinsics = np.load("../data/calib/intrinsic_calib.npz")
edges_h = np.load("../data/frog/edges_horizontal.npy")
edges_v = np.load("../data/frog/edges_vertical.npy")

mtx = intrinsics["mtx"]  # camera matrix
dist = intrinsics["dist"]  # distortion coefficients

begin = 60
end = 135
frog_dir = os.path.join("..", "data", "frog")
frog_image_paths = sorted(glob.glob(os.path.join(frog_dir, "*.jpg")))
h, w = read_img(frog_image_paths[0]).shape[:2]
frog_images = [read_img(path) for path in frog_image_paths]
frog_gray_images = [bgr2gray(img) for img in frog_images]
# show_img(frog_images[begin])

from matplotlib import pyplot as plt

# 650 750
# 100 200
R_h = extrinsics["rmat_h"]
T_h = extrinsics["tvec_h"]
R_v = extrinsics["rmat_v"]
T_v = extrinsics["tvec_v"]

cs = ((-edges_h[:, 2] - edges_h[:, 0] * 650) / edges_h[:, 1]).reshape(-1, 1)
p1_cam = np.hstack((cs, np.ones_like(cs) * (h - 650)))  # images coordinates
rays = pixel2ray(p1_cam, mtx, dist).reshape(-1, 3)  # N x 3 rays in camera frame
t = T_h[2] / (rays @ R_h[:, 2])  # intersection with plane coordinates Z = 0
P1_cam = t.reshape(-1, 1) * rays  # N x 3  points in camera frame

cs = ((-edges_h[:, 2] - edges_h[:, 0] * 750) / edges_h[:, 1]).reshape(-1, 1)
p2_cam = np.hstack((cs, np.ones_like(cs) * (h - 750)))
rays = pixel2ray(p2_cam, mtx, dist).reshape(-1, 3)  # 3 x N rays in camera frame
t = T_h[2] / (rays @ R_h[:, 2])
P2_cam = t.reshape(-1, 1) * rays  # N x 3  points in camera frame

cs = ((-edges_v[:, 2] - edges_v[:, 0] * 100) / edges_v[:, 1]).reshape(-1, 1)
p3_cam = np.hstack((cs, np.ones_like(cs) * (h - 100)))
rays = pixel2ray(p3_cam, mtx, dist).reshape(-1, 3)  # 3 x N rays in camera frame
t = T_v[2] / (rays @ R_v[:, 2])
P3_cam = t.reshape(-1, 1) * rays  # N x 3  points in camera frame

cs = ((-edges_v[:, 2] - edges_v[:, 0] * 200) / edges_v[:, 1]).reshape(-1, 1)
p4_cam = np.hstack((cs, np.ones_like(cs) * (h - 200)))
rays = pixel2ray(p4_cam, mtx, dist).reshape(-1, 3)  # 3 x N rays in camera frame
t = T_v[2] / (rays @ R_v[:, 2])
P4_cam = t.reshape(-1, 1) * rays  # N x 3  points in camera frame

# plt.imshow(frog_images[begin])
# print(p1_cam[0, 0], h - p1_cam[0, 1])
# plt.plot(p1_cam[0, 0], h - p1_cam[0, 1], "r.", markersize=10)
# plt.plot(p2_cam[0, 0], h - p2_cam[0, 1], "r.", markersize=10)
# plt.plot(p3_cam[0, 0], h - p3_cam[0, 1], "r.", markersize=10)
# plt.plot(p4_cam[0, 0], h - p4_cam[0, 1], "r.", markersize=10)
# plt.show()

# print(extrinsics["tvec_h"])
# print(extrinsics["rmat_h"])
# print(extrinsics["tvec_v"])
# print(extrinsics["rmat_v"])
# print(intrinsics.files)

# fig = plt.figure(figsize=(12, 7))
# ax = fig.add_subplot(projection="3d")
# img = ax.scatter(P1_cam[:, 0], P1_cam[:, 1], P1_cam[:, 2], cmap=plt.hot())
# fig.colorbar(img)
# ax.scatter(P2_cam[:, 0], P2_cam[:, 1], P2_cam[:, 2], cmap=plt.hot())
# ax.scatter(P3_cam[:, 0], P3_cam[:, 1], P3_cam[:, 2], cmap=plt.hot())
# ax.scatter(P4_cam[:, 0], P4_cam[:, 1], P4_cam[:, 2], cmap=plt.hot())
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# plt.show()

# print(rays)

n = np.cross(P1_cam - P2_cam, P3_cam - P4_cam)
n = n / np.linalg.norm(n, axis=1, keepdims=True)

np.save("../data/frog/normals.npy", n)
np.save("../data/frog/P1_cam.npy", P1_cam)
ul = (320, 300)
lr = (640, 820)

# plt.imshow(frog_images[begin])
# plt.imshow(frog_images[begin][ul[0] : lr[0], ul[1] : lr[1]])
# plt.show()
