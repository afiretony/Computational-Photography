import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def refocus(lightfield, depth, A):
    """
    refocus the lightfield
    Inputs:
        lightfield: 5D lightfield array (u, v, s, t, c)
        depth: the desired focus depth
        A: the aperture set
    """
    u, v, s, t = lightfield.shape[:4]
    refocused = np.zeros((s, t, 3), dtype=np.float16)
    ss, ts = np.arange(s), np.arange(t)
    t_index, s_index = np.meshgrid(ts, ss)

    for uu in tqdm(range(u), ncols=80, desc="refocusing"):
        for vv in range(v):
            if (uu, vv) in A:
                s_shifted = s_index + depth * (uu - u // 2)
                t_shifted = t_index + depth * (vv - v // 2)
                s_shifted = np.clip(s_shifted, 0, s - 1).astype(np.int16)
                t_shifted = np.clip(t_shifted, 0, t - 1).astype(np.int16)
                refocused += lightfield[
                    uu,
                    vv,
                    s_shifted,
                    t_shifted,
                    :,
                ] / len(A)
    return refocused


def composeA(shape, size):
    """
    compose the aperture set
    """
    A = set()
    if shape == "square":
        for i in range(16 // 2 - size // 2, 16 // 2 - size // 2 + size):
            for j in range(16 // 2 - size // 2, 16 // 2 - size // 2 + size):
                A.add((i, j))

    if shape == "circle":
        for i in range(16):
            for j in range(16):
                if (i - 16 // 2) ** 2 + (j - 16 // 2) ** 2 <= (size // 2) ** 2:
                    A.add((i, j))

    return A


def plot_A(A):
    """
    plot aperture set, for diagnostic purpose
    Inputs:
        A: the aperture set
    Outputs:
        plot: the plot of aperture set
    """
    A_map = np.zeros((16, 16), dtype=np.uint8)
    for i, j in A:
        A_map[i, j] = 255
    plt.imshow(A_map, cmap="gray")
    plt.show()


def create_focal_aperture_stack(lightfield, shape, num_aperture, num_depth):
    """
    create the focal aperture stack, here we use 10 circle aperture sets
    and 20 depths
    Inputs:
        lightfield: the lightfield array (u, v, s, t, c)
        shape: the shape of aperture set (square or circle)
        num_aperture: the number of aperture sets
        num_depth: the number of depths
    Ouputs:
        focal_aperture_stack: the focal aperture stack (aperture, depth, s, t, c)
    """
    u, v, s, t = lightfield.shape[:4]
    focal_aperture_stack = np.zeros((num_aperture, num_depth, s, t, 3), dtype=np.uint8)

    depth = np.linspace(-1.5, 0.2, num_depth)
    size = np.linspace(1, 16, num_aperture)

    for i in tqdm(range(num_aperture), ncols=80, desc="creating focal aperture stack"):
        for j in range(num_depth):
            if shape == "circle":
                A = composeA("circle", size[i])
            elif shape == "square":
                A = composeA("square", size[i])
            focal_aperture_stack[i, j, ...] = refocus(lightfield, depth[j], A)
    focal_aperture_stack = focal_aperture_stack.astype(np.uint8)
    return focal_aperture_stack


def visualize_focal_aperture_stack(focal_aperture_stack):
    """
    visualize the focal aperture stack
    """
    num_aperture, num_depth, s, t, c = focal_aperture_stack.shape
    focal_aperture_stack2d = np.zeros(
        (num_aperture * s, num_depth * t, c), dtype=np.uint8
    )

    for i in range(num_aperture):
        for j in range(num_depth):
            focal_aperture_stack2d[
                i * s : (i + 1) * s, j * t : (j + 1) * t, :
            ] = focal_aperture_stack[i, j, ...]
    plt.imsave("../figs/focal_aperture_stack.jpg", focal_aperture_stack2d)


lightfield = np.load("../data/lightfield.npy")
create_focal_aperture_stack(lightfield, "circle", 10, 20)
