import numpy as np
from utils import prepare_image_stack, plot_g
from weight import weight

LAMBDA = 100
Z_MIN = 0.05
Z_MAX = 0.95


def compute_g(Z, B, lamb=100, w_type="uniform", Z_MIN=0.05, Z_MAX=0.95):
    """
    inputs:
        Z: [i x j] i if the pixel values of pixel location number i in image j
        B: [i x j] is the log delta t, or log shutter speed, for image j
        l: is lambda
        w_type: weight type

    outputs:
        g(z) is the log exposure corresponding to pixel value z
        lE(i) is the log film irradiance at pixel location i

    """
    n = 256
    A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0]))
    b = np.zeros((A.shape[0], 1))

    print("Computing g")

    # include the data-fitting term
    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = weight(Z[i, j] / 255.0, w_type, Z_MIN, Z_MAX, B[i, j])
            A[k, Z[i, j]] = wij
            A[k, n + i] = -wij
            b[k, 0] = wij * np.log(B[i, j])
            k += 1

    # fix the curve by setting its middle value to 0
    A[k, 128] = 1
    k += 1

    # include the smoothness equations
    # t_k = np.ones_like(Z)
    for i in range(n - 1):
        w = weight(np.array([i + 1]) / 255.0, w_type, Z_MIN, Z_MAX, np.array([1]))
        A[k, i] = lamb * w
        A[k, i + 1] = -2 * lamb * w
        A[k, i + 2] = lamb * w
        k += 1

    x = np.linalg.lstsq(A, b, rcond=None)
    g = x[0][:n]
    logL = x[0][n:]

    print("g is computated.")

    return g, logL


def linearize(im, g):
    """
    Linearize the non-linear image
    inputs:
        im : image [nxmx3] needed to be liearized, format: int
        g  : [256] vector that converts the map
    outputs:
        I  : Linear image
    """
    I = np.exp(g[im]).reshape(im.shape)
    return I


# Z_JPG, h_, w_, c_ = prepare_image_stack(
#     "../data/door_stack", downsample=200, type="jpg"
# )
# B = np.ones_like(Z_JPG) * np.arange(0, 16)
# B = 1 / 2048 * 2**B
# g, logL = compute_g(Z_JPG, B, LAMBDA, "tent", 0.05, 0.95)
# plot_g(g, save=True)
