import os
import glob
import numpy as np
import cv2
from blur import blur_image
from utils import load_img, display_img, visualize_depth, save_depth
from matplotlib import pyplot as plt

DATA_DIR = "../data"
SCENE = "images"
path2img = os.path.join(DATA_DIR, SCENE)
images_path = sorted(glob.glob(os.path.join(path2img, "*")))
images = [load_img(img_path) for img_path in images_path]

depth_map = np.load(os.path.join(DATA_DIR, "results.npy"))

for i in range(depth_map.shape[0]):
    # normalize depth map
    depth_map[i] = (depth_map[i] - np.min(depth_map[i])) / (
        np.max(depth_map[i]) - np.min(depth_map[i])
    )


depth_map = [
    cv2.pyrUp(depth_map[i], dstsize=(640, 480)) for i in range(depth_map.shape[0])
]
i = 1
before = depth_map[i]
# cv2.imshow("before", depth_map[1])

depth_map = [
    cv2.bilateralFilter(depth_map[i], 9, 75, 75) for i in range(len(depth_map))
]
after = depth_map[i]
cv2.imshow("after", depth_map[i])

# def select_focal_point(image, depth_map):
#     """
#     Select the focal point of the image.
#     """
#     image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     cv2.imshow("image", image_copy)

#     def mouse_click(event, x, y, flags, param):

#         # to check if left mouse
#         # button was clicked
#         image, depth_map = param
#         global ix, iy

#         if event == cv2.EVENT_LBUTTONDOWN:

#             x1, y1 = x - 10, y - 10
#             x2, y2 = x + 10, y + 10

#             # compute blurred image
#             ix, iy = x, y
#             d_focus = depth_map[iy, ix]
#             blurred = blur_image(image, depth_map, d_focus, 20)

#             # display
#             cv2.rectangle(blurred, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR)
#             cv2.imshow("image", blurred_bgr)
#             print(d_focus)


def interactive(image, depth_map):

    global ix, iy, f
    ix = 320
    iy = 240
    f = 5

    def draw():
        global ix, iy, f
        # compute blurred image
        x1, y1 = ix - 10, iy - 10
        x2, y2 = ix + 10, iy + 10
        d_focus = depth_map[iy, ix]
        blurred = blur_image(image, depth_map, d_focus, f)

        # display
        # cv2.rectangle(blurred, (x1, y1), (x2, y2), (255, 0, 0), 2)
        blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", blurred_bgr)

    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global ix, iy
            ix, iy = x, y
            draw()

    def on_trackbar_F(val):
        global f
        f = (val + 1) / 15 * 8
        draw()

    draw()
    cv2.setMouseCallback("image", mouse_click)
    cv2.createTrackbar("aperture", "image", 0, 15, on_trackbar_F)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return ix, iy


image, depth = images[i], depth_map[i]

diff = np.abs(before - after)
# normalize
diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
# cv2.imshow("difference", diff)

# cv2.waitKey(0)

interactive(image, depth)
# d_focus = depth_map[8][iy, ix]

# blurred = blur_image(images[8], depth_map[8], d_focus, 20)

# display_img(blurred)
