# %%
from importlib.resources import path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage
from skimage.color import rgb2gray
from scipy.interpolate import interp2d, RectBivariateSpline

def read_RAW(img_path):
    """
    read RAW images using skimage
    """
    im = io.imread(img_path)
    return im

def img_info(im):
    """
    input: 
        numpy array
    output:
        image info contains [bits per pixel, width, height]
    """
    if len(im.shape) == 2:
        im_info = {"bpp": im.dtype,
                    "width": im.shape[1],
                    "height": im.shape[0]
                    }

    elif len(im.shape) == 3:
        im_info = {"bpp": im.dtype,
                    "width": im.shape[2],
                    "height": im.shape[1]
                    }

    else:
        raise ValueError("Check image dimension")

    return im_info

def convert2double(im):
    """
    convert the image into a double-precision array
    """
    return im.astype(np.double)

def linearize(im, black, white):
    """
    Linearize the image using method descripted in 1.1
    """
    linearized = np.zeros_like(im)
    linearized = (im - black) / (white - black)
    linearized = np.clip(linearized, 0., 1.)
    return linearized

def identify_bayer_dummy(im, pattern):
    """
    based on pattern of choice, return 3-channel(RGB) image 
    """

    im_info = img_info(im)
    h, w = im_info["height"], im_info["width"]

    RGB = np.zeros((h // 2, w // 2, 3))
    
    for i in range(h//2):
        for j in range(w//2):
            # print(j)
            RGB[i][j][0] = im[i*2, j*2]
            RGB[i][j][1] = im[i*2, j*2+1]
            RGB[i][j][2] = im[i*2+1, j*2+1]
    
    return RGB

def demosaicing(im):
    """
    demosaicing using pattern rggb
    """
    h, w = im.shape[0], im.shape[1]

    # Red Channel
    x = np.arange(0, h, 2)
    y = np.arange(0, w, 2)
    xv, yv = np.meshgrid(x, y)
    z = im[xv.T, yv.T]
    f = RectBivariateSpline(x, y, z)
    R_channel = f(np.arange(0,h), np.arange(0,w))

    
    # Green Channel
    x = np.arange(1, h, 2)
    y = np.arange(0, w, 2)
    xv, yv = np.meshgrid(x, y)
    z = im[xv.T, yv.T]
    f = RectBivariateSpline(x, y, z)
    G_channel = f(np.arange(0,h), np.arange(0,w))

    # Blue Channel
    x = np.arange(1, h, 2)
    y = np.arange(1, w, 2)
    xv, yv = np.meshgrid(x, y)
    z = im[xv.T, yv.T]
    f = RectBivariateSpline(x, y, z)
    B_channel = f(np.arange(0,h), np.arange(0,w))

    demosaic = np.stack((R_channel, G_channel, B_channel), 2)
    demosaic = np.clip(demosaic, 0, 1)
    return demosaic

    

def identify_bayer(im, pattern):
    """
    based on pattern of choice, return 3-channel(RGB) image 
    """
    assert pattern in ['grbg', 'rggb', 'bggr', 'gbrg']

    im_info = img_info(im)
    h, w = im_info["height"], im_info["width"]
    RGB = np.zeros((h //2, w // 2, 3))
    
    # index map according to bayer position
    # upper left
    ulxv, ulyv = np.meshgrid(np.arange(0, h, 2), np.arange(0, w, 2))

    # upper right
    urxv, uryv = np.meshgrid(np.arange(0, h, 2), np.arange(1, w, 2))

    # lower left
    llxv, llyv = np.meshgrid(np.arange(1, h, 2), np.arange(0, w, 2))

    # lower right
    lrxv, lryv = np.meshgrid(np.arange(1, h, 2), np.arange(1, w, 2))

    if pattern == 'grbg':
        RGB[:,:,0] = im[urxv.T, uryv.T]
        RGB[:,:,1] = im[ulxv.T, ulyv.T]
        RGB[:,:,2] = im[llxv.T, llyv.T]
    elif pattern == 'rggb':
        RGB[:,:,0] = im[ulxv.T, ulyv.T]
        RGB[:,:,1] = im[urxv.T, uryv.T]
        RGB[:,:,2] = im[lrxv.T, lryv.T]
    elif pattern == "bggr":
        RGB[:,:,0] = im[lrxv.T, lryv.T]
        RGB[:,:,1] = im[urxv.T, uryv.T]
        RGB[:,:,2] = im[ulxv.T, ulyv.T]
    elif pattern == "gbrg":
        RGB[:,:,0] = im[llxv.T, llyv.T]
        RGB[:,:,1] = im[ulxv.T, ulyv.T]
        RGB[:,:,2] = im[urxv.T, uryv.T]

    return RGB

def show_image(im):
    plt.figure()
    plt.imshow(im)
    plt.show()

def save_image(im, filename):
    plt.figure()
    plt.imsave(filename, im)


def get_mask(img, pattern):
    """
    inputs:
        img: RAW image to get size of
        pattern: bayer pattern combination
    outputs:
        R_mask, G_mask, B_mask: boolean-mask indicates existance of the color pixel
    """
    assert pattern in ['grbg', 'rggb', 'bggr', 'gbrg']
    h, w = img.shape[0], img.shape[1]
    R_mask, G_mask, B_mask = np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool)

    # index map according to bayer position
    # upper left
    ulxv, ulyv = np.meshgrid(np.arange(0, h, 2), np.arange(0, w, 2))
    # upper right
    urxv, uryv = np.meshgrid(np.arange(0, h, 2), np.arange(1, w, 2))
    # lower left
    llxv, llyv = np.meshgrid(np.arange(1, h, 2), np.arange(0, w, 2))
    # lower right
    lrxv, lryv = np.meshgrid(np.arange(1, h, 2), np.arange(1, w, 2))

    if pattern == 'grbg':
        R_mask[urxv.T, uryv.T] = 1
        G_mask[ulxv.T, ulyv.T] = 1
        G_mask[lrxv.T, lryv.T] = 1
        B_mask[llxv.T, llyv.T] = 1

    elif pattern == 'rggb':
        R_mask[ulxv.T, ulyv.T] = 1
        G_mask[urxv.T, uryv.T] = 1
        G_mask[llxv.T, llyv.T] = 1
        B_mask[lrxv.T, lryv.T] = 1

    elif pattern == "bggr":
        R_mask[lrxv.T, lryv.T] = 1
        G_mask[urxv.T, uryv.T] = 1
        G_mask[llxv.T, llyv.T] = 1
        B_mask[ulxv.T, ulyv.T] = 1

    elif pattern == "gbrg":
        R_mask[llxv.T, llyv.T] = 1
        G_mask[ulxv.T, ulyv.T] = 1
        G_mask[lrxv.T, lryv.T] = 1
        B_mask[urxv.T, uryv.T] = 1

    return R_mask, G_mask, B_mask

def white_balancing_white_world(img, pattern):
    """
    white balancing based on forcing brightest object in scene to be white.
    input: RAW image
    output: balanced image
    """
    h, w = img.shape[0], img.shape[1]

    # # perform average pooling on the 2x2 blocks
    # avg_img = skimage.measure.block_reduce(img, (2,2), np.mean)
    
    # find brighest block
    # ind = np.unravel_index(np.argmax(avg_img, axis=None), avg_img.shape)
    # brightest = img[2*ind[0]:2*ind[0]+2, 2*ind[1]:2*ind[1]+2]

    # add offset to all blocks
    # offset = 1.0 - brightest
    # offset = np.tile(offset, (h//2, w//2))

    R_mask, G_mask, B_mask = get_mask(img, pattern)

    # compute per channel maximum
    R_max = np.max(img[R_mask])
    G_max = np.max(img[G_mask])
    B_max = np.max(img[B_mask])

    img[R_mask] *= G_max / R_max
    img[B_mask] *= G_max / B_max

    return img

def white_balancing_gray_world(img, pattern):
    """
    force average color of scene to be grey
    """
    R_mask, G_mask, B_mask = get_mask(img, pattern)

    R_mean = np.mean(img[R_mask])
    G_mean = np.mean(img[G_mask])
    B_mean = np.mean(img[B_mask])

    img[R_mask] *= G_mean / R_mean
    img[B_mask] *= G_mean / B_mean
    return np.clip(img, 0, 1)
    
def white_balancing_preset(img, pattern):
    """
    wb based on preset
    """
    R_mask, G_mask, B_mask = get_mask(img, pattern)
    img[R_mask] *= R_SCALE 
    img[G_mask] *= G_SCALE
    img[B_mask] *= B_SCALE

    return np.clip(img, 0, 1)

def white_balancing_manual(img, coord):
    """
    manually select the white patch and normalize all three channels 
    for rggb bayer pattern only!
    """
    ul = [coord[0] // 2 * 2, coord[1] // 2 * 2]
    R_mask, G_mask, B_mask = get_mask(img, "rggb")

    R_patch = img[ul[0],   ul[1]]
    G_patch = img[ul[0],   ul[1]+1]
    B_patch = img[ul[0]+1, ul[1]+1]
    
    img[R_mask] *= G_patch / R_patch
    img[B_mask] *= G_patch / B_patch
    return np.clip(img, 0, 1)


def color_space_correction(RGB_cam):
    """
    perform color space correctino
    input:
        RGB_cam: color space determined by the camera's spectral sensitivity functions
    output:
        sRGB: linear sRGB color space used in display functions
    """

    M_SRGB2CAM = np.matmul(np.array(M_XYZ2CAM) / 10000., np.array(M_SRGB2XYZ))
    row_sums = M_SRGB2CAM.sum(axis=1)
    M_SRGB2CAM /= row_sums[:, np.newaxis]

    inv_M = np.linalg.inv(M_SRGB2CAM)
    sRGB = np.matmul(inv_M, np.transpose(RGB_cam, (1,2,0)))
    return np.transpose(sRGB, (2, 0, 1))

def brightness_adjustment(RGB, exp_mean):
    """
    adjust brightness of the RGB image
    input:
        RGB: RGB image to scale for
        exp_mean: expected mean gray scale value, range in [0, 1]
    output:
        adjusted image

    """
    mean_gray = np.mean(rgb2gray(RGB))
    
    return np.clip(RGB * exp_mean / mean_gray, 0, 1)

def gamma_encoding(RGB):
    RGB[RGB <=0.0031308] *= 12.92
    RGB[RGB > 0.0031308] = 1.055 * RGB[RGB > 0.0031308]**(1 / 2.4) - 0.055
    return RGB


# %%
BLACK = 150.
WHITE = 4095.
R_SCALE = 2.394531
G_SCALE = 1.0
B_SCALE = 1.597656

M_XYZ2CAM = [[6988,-1384,-714],[-5631,13410,2447],[-1485,2204,7318]]
M_SRGB2XYZ = [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]

# Read and load image
img = read_RAW("../data/campus.tiff")

# formatting and linearization
img = convert2double(img)
img = linearize(img, BLACK, WHITE)

# white balance
# img = white_balancing_gray_world(img, "rggb")
# img= white_balancing_preset(img, "rggb")
img = white_balancing_white_world(img, "rggb")
# img = white_balancing_manual(img, [3100,3380])


# demosaic
RGB = demosaicing(img)

# apply sRGB curve
sRGB = color_space_correction(RGB)
RGB = brightness_adjustment(sRGB, 0.25)

# Gamma encoding
RGB = gamma_encoding(RGB)

# display image
# show_image(RGB)

# save image
filename = "AWB_white"
save_image(RGB, "../data/{}.jpg".format(filename))
save_image(RGB, "../data/{}.png".format(filename))

# %%
