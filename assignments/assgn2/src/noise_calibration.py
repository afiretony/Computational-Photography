import os
import numpy as np
import glob
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import read_image
import numpy as np
from sklearn.linear_model import LinearRegression


def create_ramp_pattern():

    PATTERN = np.tile(np.linspace(0, 1, 255), (255, 1))
    PATTERN = np.stack((PATTERN, PATTERN, PATTERN)).transpose(1, 2, 0)
    print(PATTERN.shape)
    plt.figure()
    plt.imsave("ramp.png", PATTERN)


def capture_test(scene="test", exposure=8):
    """
    capture frame with exposure time 1 / (2**exposure)
    """
    subprocess.run(["gphoto2", "--auto-detect"])
    subprocess.run(
        [
            "gphoto2",
            "--set-config-value",
            "/main/capturesettings/shutterspeed=1/{}".format(int(2**exposure)),
        ]
    )
    subprocess.run(
        [
            "gphoto2",
            "--capture-image-and-download",
            "--filename",
            "{}/test_{}.%C".format(scene, exposure),
        ]
    )


def capture_hdr(scene="test", begin=12, end=5):
    """
    capture image arrays with various exposure time
    from 1 / 2 ** begin to 1 / 2 ** end
    increase x 2
    """
    subprocess.run(["gphoto2", "--auto-detect"])
    k = 1
    for i in range(begin, end - 1, -1):
        subprocess.run(
            [
                "gphoto2",
                "--set-config-value",
                "/main/capturesettings/shutterspeed=1/{}".format(2**i),
            ]
        )
        subprocess.run(
            [
                "gphoto2",
                "--capture-image-and-download",
                "--filename",
                "{}/exposure{:02d}.%C".format(scene, k),
            ]
        )
        k += 1


def capture_darkframe(scene="darkframe", exposure=12):
    """
    capture darkframe
    """
    subprocess.run(["gphoto2", "--auto-detect"])
    subprocess.run(
        [
            "gphoto2",
            "--set-config-value",
            "/main/capturesettings/shutterspeed=1/{}".format(2**exposure),
        ]
    )
    for i in range(50):
        subprocess.run(
            [
                "gphoto2",
                "--capture-image-and-download",
                "--filename",
                "{}/{}_{:02d}.%C".format(scene, scene, i + 1),
            ]
        )


def compute_darkframe(darkframe_stack):

    return np.mean(darkframe_stack, 1)


def prepare_image_stack(path_to_dir, downsample=200, type="tiff", average=False):

    path_to_images = sorted(glob.glob(os.path.join(path_to_dir, "*.{}".format(type))))
    N = len(path_to_images)

    sample = read_image(path_to_images[0])[::downsample, ::downsample, 0]
    if average:
        sample = sample.mean((0, 1))

    H, W = sample.shape
    C = 1
    Z = np.zeros((H * W * C, N))

    for i in tqdm(range(N)):
        im = read_image(path_to_images[i])[::downsample, ::downsample, 0].flatten()
        Z[:, i] = im
    print("Stack shape: ", Z.shape)
    return Z


def clean_darkframe(image_stack, darkframe):
    """
    subtracts darkframe from image

    inputs:
        image_stack: [HW3xN] flattened images
        darkframe: [HW3x1] flattened darkframe
    output:
        out : [HW3xN] image with cleaned darkframe
    """
    out = image_stack.T - darkframe.T
    return out.T


def mean(Z):
    """
    mean value of image stack

    input:
        Z [HW3 x N]

    output:
        Z_mean: [HW3 x 1]
    """
    return np.mean(Z, -1)


def variance(Z):
    """
    variance of image stack

    input:
        Z [HW3 x N]

    output:
        Z_var: [HW3 x 1]
    N = Z.shape[0]
    """
    N = Z.shape[0]

    Z_mean = mean(Z)
    Z_var = 1 / (N - 1) * np.sum(((Z.T - Z_mean.T).T) ** 2, -1)
    return Z_var


def plot_pixel_histogarm(Z, loc, save=False):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(Z[loc])
    plt.title("Histogram of pixel #{}'s value across various images".format(loc))
    plt.xlabel("Value")
    plt.ylabel("Occurances")
    if save:
        plt.savefig("histogram_{}.jpg".format(loc))

    plt.show()


def plot_mean_variance(MV_map, save=False):
    Z_mean, Z_var = MV_map.keys(), MV_map.values()
    plt.figure(figsize=(12, 7))
    plt.plot(Z_mean, Z_var, ".")

    a, b = regression(MV_map)
    reg_X = np.linspace(min(Z_mean), max(Z_mean), 100)
    reg_Y = reg_X * a + b
    plt.plot(reg_X, reg_Y, "-")

    plt.title("Mean v.s. Variance")
    plt.xlabel("mean")
    plt.ylabel("variance")

    if save:
        plt.savefig("mean_vs_variance.jpg")

    plt.show()


def construct_mean_variance_map(start, cutoff, Z_mean, Z_var):

    MV_map = {}
    Z_mean_ = Z_mean.astype(int)
    Z_mean_ = np.clip(Z_mean_, start, cutoff)

    N = len(Z_mean_)

    for i in range(N):
        u, v = Z_mean_[i], Z_var[i]

        if start < u < cutoff:
            if u not in MV_map:
                MV_map[u] = [v]
            else:
                MV_map[u].append(v)

    keys = MV_map.keys()

    for i in keys:
        MV_map[i] = mean(MV_map[i])

    return MV_map


def regression(MV_map):
    model = LinearRegression()
    x = np.array(list(MV_map.keys())).reshape(-1, 1)
    y = np.array(list(MV_map.values()))
    model.fit(x, y)

    b, a = model.intercept_, model.coef_
    print(f"intercept: {b}")
    print(f"slope: {a}")
    return a, b


ramp = prepare_image_stack("../data/ramp")
darkframe_stack = prepare_image_stack("../data/darkframe")

darkframe = compute_darkframe(darkframe_stack)
cleaned = clean_darkframe(ramp, darkframe)

Z_mean = mean(cleaned)
Z_var = variance(cleaned)

MV_map = construct_mean_variance_map(50, 1400, Z_mean, Z_var)
plot_mean_variance(MV_map, True)
