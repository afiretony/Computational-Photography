import numpy as np


def gamma_correction(im):
    """
    apply gamma correction to input images
    input: non-linear image
    output: linear image
    """
    encoded = np.where(im <= 0.0404482, im / 12.92, ((im + 0.055) / 1.055) ** 2.4)
    return encoded
