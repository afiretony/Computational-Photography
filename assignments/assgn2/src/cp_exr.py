#!/usr/bin/python
""" This is a module for hdr imaging homework (15-463/663/862, Computational Photography, Fall 2020, CMU).

You can import necessary functions into your code as follows:
from cp_exr import writeEXR

Note that you have to install OpenEXR package to use writeEXR function.
Please refer to the following link for the details.
https://github.com/jamesbowman/openexrpython
"""

import numpy as np
import OpenEXR
import Imath

def readEXR(name):
    """ Read OpenEXR image (both 16-bit and 32-bit datatypes are supported)
    """
    
    exrFile = OpenEXR.InputFile(name)
    pt = Imath.PixelType(Imath.PixelType.FLOAT) # convert any datatype to float32 for numpy
    
    strR = exrFile.channel('R', pt)
    strG = exrFile.channel('G', pt)
    strB = exrFile.channel('B', pt)

    R = np.frombuffer(strR, dtype=np.float32)
    G = np.frombuffer(strG, dtype=np.float32)
    B = np.frombuffer(strB, dtype=np.float32)
 
    img = np.dstack((R,G,B))

    dw = exrFile.header()['dataWindow']
    sizeEXR = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1, 3)
    img = np.reshape(img, sizeEXR)
    
    return img

def writeEXR(name, data, pixeltype='HALF'):
    """ Write OpenEXR file from data 
    
    pixeltype
    ---------
    HALF:   16-bit OpenEXR
    FLOAT:  32-bit OpenEXR (not supported in preview app)
    """
    if pixeltype == 'FLOAT':
        pt = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        dt = np.float32
    elif pixeltype == 'HALF':
        pt = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        dt = np.float16
    else:
        raise Exception("Unsupported data type")
        
    HEADER = OpenEXR.Header(data.shape[1], data.shape[0])
    HEADER['channels'] = dict([(c, pt) for c in "RGB"])

    exr = OpenEXR.OutputFile(name, HEADER)

    R = (data[:,:,0]).astype(dt).tobytes()
    G = (data[:,:,1]).astype(dt).tobytes()
    B = (data[:,:,2]).astype(dt).tobytes()

    exr.writePixels({'R' : R, 'G' : G, 'B' : B })
    exr.close()

