from utils import read_tif, show_im, save_im, read_image
import numpy as np
import os
import cv2
from scipy.signal import convolve2d
from scipy.interpolate import interpn
import gc
from gamma_encoding import gamma_encoding
from gamma_correction import gamma_correction
import logger
import argparse


def gradient(I):
    I_x = -np.diff(I, 1, 1, prepend=0)
    I_y = -np.diff(I, 1, 0, prepend=0)
    return I_x, I_y


def divergence(I_x, I_y):

    I_xx = -np.diff(I_x, 1, 1, append=0)
    I_yy = -np.diff(I_y, 1, 0, append=0)

    return I_xx + I_yy


def laplacian(I):
    filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # laplacian filter
    lap = []
    for i in range(3):
        lap.append(
            convolve2d(I[..., i], filter, mode="full", boundary="fill", fillvalue=0)
        )

    lap = np.dstack(lap)
    return lap[1:-1, 1:-1, :]


def display(I):
    """
    normalize the data before display
    """
    res = np.clip((I - np.min(I)) / (np.max(I) - np.min(I)), 0, 1)
    return res


def createBoundaryMask(I, thickness):
    """
    creates mask for boundary where 1 is boundary and 0 not
    """
    B = np.zeros_like(I)
    B[:thickness, ...] = 1
    B[-thickness:, ...] = 1
    B[:, :thickness, :] = 1
    B[:, -thickness:, :] = 1
    return 1 - B


def GCD(D, I_init, B, I_bound, epsilon, N):
    """
    gradient field integration with conjugate gradient descent
    inputs:
        D: target divergence
        I_init: initialization
        B: boundary mask
        I_bound: boundary values
        epsilon: convergence parameter
        N: iteration number
    outputs:
        I_res: integrated image
    """
    I_res = B * I_init + (1 - B) * I_bound
    r = B * (D - laplacian(I_res))
    d = r
    delta_new = np.linalg.norm(r) ** 2
    n = 0
    while np.linalg.norm(r) > epsilon and n < N:
        if n > 100 and n % 100 == 0:
            logger.info("Number of iteration: {}".format(n))
        q = laplacian(d)
        eta = delta_new / np.sum(d * q)
        I_res += B * (eta * d)
        r = B * (r - eta * q)
        delta_old = delta_new
        delta_new = np.linalg.norm(r) ** 2
        beta = delta_new / delta_old
        d = r + beta * d
        n += 1

    return np.clip(I_res, 0, 1)


def compute_GOC_map(ambient, flash):
    """
    compute gradient orientation coherency map M
    """
    a_x, a_y = gradient(ambient)
    f_x, f_y = gradient(flash)

    M = np.abs(a_x * f_x + a_y * f_y)
    magtitude = np.sqrt(a_x**2 + a_y**2) * np.sqrt(f_x**2 + f_y**2) + 0.005
    mask = np.where(magtitude < 0.005, 0, 1)
    M = M / magtitude

    return M, mask


def compute_saturation_weight_map(flash, tau_s, sigma):
    omega_s = np.tanh(sigma * (flash - tau_s))
    # normalize
    omega_s_ = (omega_s - np.min(omega_s)) / (np.max(omega_s) - np.min(omega_s) + 0.001)
    return omega_s_


def compute_new_gradient_field(omega_s, ambient, flash, M):
    a_x, a_y = gradient(ambient)
    f_x, f_y = gradient(flash)

    grad_x = omega_s * a_x + (1 - omega_s) * (M * f_x + (1 - M) * a_x)
    grad_y = omega_s * a_y + (1 - omega_s) * (M * f_y + (1 - M) * a_y)
    return grad_x, grad_y


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Gradient domain processing")

    parser.add_argument("--scene", type=str, default="museum")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--N", type=float, default=1000)
    parser.add_argument("--sigma", type=float, default=40)
    parser.add_argument("--tau_s", type=float, default=0.9)
    parser.add_argument("--initialization", type=str, default="ambient")
    parser.add_argument("--boundary", type=str, default="ambient")

    args = parser.parse_args()
    data_dir = "../data"

    ambient = (
        read_image(
            os.path.join(data_dir, args.scene, "{}_ambient.jpg".format(args.scene))
        )[:: args.downsample, :: args.downsample, :3]
        / 255.0
    )
    flash = (
        read_image(
            os.path.join(data_dir, args.scene, "{}_flash.jpg".format(args.scene))
        )[:: args.downsample, :: args.downsample, :3]
        / 255.0
    )

    logger = logger.create_logger()
    logger.info(args)

    M, mask = compute_GOC_map(ambient, flash)
    omega_s = compute_saturation_weight_map(flash, args.tau_s, args.sigma)
    grad_x, grad_y = compute_new_gradient_field(omega_s, ambient, flash, M)

    amb_x, amb_y = gradient(ambient)
    fls_x, fls_y = gradient(flash)
    save_im(
        os.path.join(data_dir, args.scene, "amb_x.jpg"),
        display(amb_x),
    )
    save_im(
        os.path.join(data_dir, args.scene, "amb_y.jpg"),
        display(amb_y),
    )

    save_im(
        os.path.join(data_dir, args.scene, "fls_x.jpg"),
        display(fls_x),
    )
    save_im(
        os.path.join(data_dir, args.scene, "fls_y.jpg"),
        display(fls_y),
    )

    save_im(
        os.path.join(data_dir, args.scene, "grad_x.jpg"),
        display(grad_x),
    )
    save_im(
        os.path.join(data_dir, args.scene, "grad_y.jpg"),
        display(grad_y),
    )

    div = divergence(grad_x, grad_y)

    B = createBoundaryMask(ambient, 2)

    if args.initialization == "ambient":
        I_init = ambient
    elif args.initialization == "flash":
        I_init = flash
    elif args.initialization == "average":
        I_init = (ambient + flash) / 2
    elif args.initialization == "zeros":
        I_init = np.zeros_like(ambient)
    else:
        logger.error("Unable to create initialization image to integrate")

    if args.boundary == "ambient":
        I_bound = ambient
    elif args.boundary == "flash":
        I_bound = flash
    elif args.boundary == "average":
        I_bound = (ambient + flash) / 2.0
    else:
        logger.error("Unable to create boundary condition for image to integrate")

    res = GCD(
        D=div, I_init=I_init, B=B, I_bound=I_bound, epsilon=args.epsilon, N=args.N
    )

    save_im(
        os.path.join(
            data_dir,
            args.scene,
            "i_{}_b_{}.jpg".format(args.initialization, args.boundary),
        ),
        res,
    )

# %%
