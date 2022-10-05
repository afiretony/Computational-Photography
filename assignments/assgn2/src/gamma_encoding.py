import numpy as np

def gamma_encoding(im):
    """
    apply gamma encoding to input images
    """
    encoded = np.where(im <= 0.0031308, im*12.92, (1+0.055) * im ** (1/2.4) - 0.055)
    return encoded


