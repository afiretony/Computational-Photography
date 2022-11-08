import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def visualize_AFI(focal_aperture_stack, i, j):
    """
    visualize the aperture field image
    Inports:
        focal_aperture_stack: the focal aperture stack (aperture, depth, s, t, c)
        i: the index of pixel x
        j: the index of pixel y
    """
    num_aperture, num_depth, s, t, c = focal_aperture_stack.shape
    AFI = np.zeros((num_aperture, num_depth, c), dtype=np.uint8)
    for k in range(num_aperture):
        for l in range(num_depth):
            AFI[k, l, :] = focal_aperture_stack[k, l, i, j, :]
    plt.imsave("../figs/AFI_{}_{}.jpg".format(i, j), AFI)


focal_aperture_stack = np.load("../data/focal_aperture_stack_circle.npy")


def confocal_stereo(focal_aperture_stack):
    """
    create the depth map and all in focus image using confocal stereo
    Inputs:
        focal_aperture_stack: the focal aperture stack (aperture, depth, s, t, c)
    Outputs:
        depth_map: the depth map (s, t)
        all_in_focus: the all in focus image (s, t, c)
    """
    num_aperture, num_depth, s, t, c = focal_aperture_stack.shape
    depth_map = np.zeros((s, t), dtype=np.float32)
    all_in_focus = np.zeros((s, t, c), dtype=np.uint8)

    for i in tqdm(range(s), ncols=80, desc="creating depth map"):
        for j in range(t):
            AFI = focal_aperture_stack[:, :, i, j, :]
            AFI = AFI.astype(np.float32)
            AFI = np.mean(AFI, axis=2)
            AFI_var = np.var(AFI, axis=0)
            depth_index = np.argmin(AFI_var)
            depth_map[i, j] = depth_index
            all_in_focus[i, j, :] = focal_aperture_stack[0, depth_index, i, j, :]

    # normalize depth map
    depth_map = depth_map / num_depth
    return depth_map, all_in_focus


depth_map, all_in_focus = confocal_stereo(focal_aperture_stack)
