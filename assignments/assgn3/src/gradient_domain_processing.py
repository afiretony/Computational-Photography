# %%
from torch import save
from utils import read_tif, show_im, save_im, read_image
import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.interpolate import interpn
import gc
from gamma_encoding import gamma_encoding
from gamma_correction import gamma_correction
import logger

logger = logger.create_logger()


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
        I_res = I_res + B * (eta * d)
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
    normalized = (omega_s - np.min(omega_s)) / (np.max(omega_s) - np.min(omega_s))
    return normalized


def compute_new_gradient_field(omega_s, ambient, flash, M):
    a_x, a_y = gradient(ambient)
    f_x, f_y = gradient(flash)

    grad_x = omega_s * a_x + (1 - omega_s) * (M * f_x + (1 - M) * a_x)
    grad_y = omega_s * a_y + (1 - omega_s) * (M * f_y + (1 - M) * a_y)
    return grad_x, grad_y


# %%
ambient = read_image("../data/museum/museum_ambient.png")[:, :, :3] / 255.0
flash = read_image("../data/museum/museum_flash.png")[:, :, :3] / 255.0
ambient = read_image("glass_ambient.jpg")[::10, ::10] / 255.0
flash = read_image("glass_flash.jpg")[::10, ::10] / 255.0
# I_x, I_y = gradient(ambient)
# div = divergence(I_x, I_y)
lap = laplacian(ambient)

# res = display(lap)
# show_im(res)

# GCD(D=lap, I_init=ambient, B=)

# %%
# B = createBoundaryMask(ambient, 2)
# res = GCD(
#     D=lap, I_init=np.zeros_like(ambient), B=B, I_bound=ambient, epsilon=0.01, N=1000
# )
# show_im(res)


# %%
tau_s = 0.1
sigma = 500
M, mask = compute_GOC_map(ambient, flash)
omega_s = compute_saturation_weight_map(flash, tau_s, sigma)
grad_x, grad_y = compute_new_gradient_field(omega_s, ambient, flash, M)
div = divergence(grad_x, grad_y)

# %%
B = createBoundaryMask(ambient, 2)
res = GCD(
    D=div, I_init=np.zeros_like(ambient), B=B, I_bound=ambient, epsilon=0.001, N=1000
)
# show_im(res)
save_im("merged.jpg", res)

# %%
