from re import S
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
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
    assert pattern in ['grbg', 'rggb', 'bggr', 'gbrg']

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
    RGB = np.zeros((h, w, 3))
    grid_x, grid_y = np.meshgrid(np.arange(0, h), np.arange(0, w))

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

    # print(xv.shape, yv.shape, z.shape)
    # print(z)
    # f = interp2d(xv, yv, z)
    # z = f(grid_x, grid_y)
    # print(z)
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


def white_balancing_white_world(RGB):
    """
    white balancing based on forcing brightest object in scene to be white.
    input: RGB image
    output: balanced image
    """
    RGB_sum = np.sum(RGB, 2)
    ind = np.unravel_index(np.argmax(RGB_sum, axis=None), RGB_sum.shape)
    brightest = RGB[ind[0], ind[1]]
    offset = 1.0 - brightest

    return RGB + offset

def white_balancing_gray_world(RGB):
    """
    force average color of scene to be grey
    """
    dim = RGB.shape[0] * RGB.shape[1] 
    RGB_average = np.array([np.sum(RGB[:,:,0]) / dim, np.sum(RGB[:,:,1]) / dim, np.sum(RGB[:,:,2]) / dim])
    gray = np.array([0.5, 0.5, 0.5])
    offset = gray - RGB_average
    balanced = RGB + offset
    # print(np.array([np.sum(balanced[:,:,0]) / dim, np.sum(balanced[:,:,1]) / dim, np.sum(balanced[:,:,2]) / dim]))
    return balanced
    
def white_balancing_preset(RGB):
    """
    wb based on preset
    """
    RGB[:,:,0] *= R_SCALE
    RGB[:,:,1] *= G_SCALE
    RGB[:,:,2] *= B_SCALE
    return RGB


BLACK = 150.
WHITE = 4095.
R_SCALE = 2.394531
G_SCALE = 1.0
B_SCALE = 1.597656


img = read_RAW("../data/campus.tiff")

img = convert2double(img)
img = linearize(img, BLACK, WHITE)

RGB = demosaicing(img)

save_image(RGB, "temp.png")

# RGB = identify_bayer(img, "rggb")



# RGB = identify_bayer_dummy(img,"gbrg")
# show_image(RGB)
# RGB = white_balancing_white_world(RGB)
# RGB = white_balancing_gray_world(RGB)
# RGB = white_balancing_preset(RGB)
# show_image(RGB)

# print(img)
# print(img_info(img))


