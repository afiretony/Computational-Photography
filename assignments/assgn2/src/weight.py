import numpy as np


def w_uniform(z, Z_MIN, Z_MAX):
    ans = np.zeros_like(z)
    ans[np.logical_and(Z_MIN <= z, z <= Z_MAX)] = 1.0
    return ans


def w_tent(z, Z_MIN, Z_MAX):
    ans = np.zeros_like(z)
    ans[np.logical_and(Z_MIN <= z, z <= Z_MAX)] = np.where(z < 1 - z, z, 1 - z)[
        np.logical_and(Z_MIN <= z, z <= Z_MAX)
    ]
    return ans


def w_gaussian(z, Z_MIN, Z_MAX):
    ans = np.zeros_like(z)
    ans[np.logical_and(Z_MIN <= z, z <= Z_MAX)] = np.exp(
        -4 * (z - 0.5) ** 2 / 0.5**2
    )[np.logical_and(Z_MIN <= z, z <= Z_MAX)]
    return ans


def w_photon(z, t_k, Z_MIN, Z_MAX):
    ans = np.zeros_like(z)
    ans[np.logical_and(Z_MIN <= z, z <= Z_MAX)] = t_k[
        np.logical_and(Z_MIN <= z, z <= Z_MAX)
    ]
    return ans


def w_optimal(z, t_k, gain, additive, Z_MIN, Z_MAX):
    ans = np.zeros_like(z)
    ans = t_k**2 / (gain * z + additive)
    ans[np.logical_or(z > Z_MAX, z < Z_MIN)] = 0
    return ans


def weight(
    z, type, Z_MIN, Z_MAX, t_k=None, gain=1.3291 / 65535, additive=295.8 / 65535**2
):
    if type == "uniform":
        return w_uniform(z, Z_MIN, Z_MAX)

    elif type == "tent":
        return w_tent(z, Z_MIN, Z_MAX)

    elif type == "gaussian":
        return w_gaussian(z, Z_MIN, Z_MAX)

    elif type == "photon":
        return w_photon(z, t_k, Z_MIN, Z_MAX)

    elif type == "optimal":
        """
        optimal is for 16 bit RAW image only, the gain and additive for noise is estimated
        in 16 bit scale, so we don't need to linearize z
        """
        return w_optimal(z, t_k, gain, additive, Z_MIN, Z_MAX)
