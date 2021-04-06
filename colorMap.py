import os
import binascii
import numpy as np
import math

def getFileSize(filepath):
    f_size = os.path.getsize(filepath)
    size = f_size/float(1024)
    return round(size, 2)

def getRectangleGrayImg(filename, width=32):
    
    with open(filename, 'rb') as f:
        content = f.read()

    hexst = binascii.hexlify(content)
    fh = np.array([int(hexst[i:i+2], 16) for i in range(0, len(hexst), 2)])
    
    pad_len = width - len(fh) % width
    pad = np.zeros((1, pad_len))
    img = np.concatenate((fh, pad), axis=None)

    return img.reshape(-1, width)

def getSquareGrayImg(filename):
    
    with open(filename, 'rb') as f:
        content = f.read()

    hexst = binascii.hexlify(content)
    fh = np.array([int(hexst[i:i+2], 16) for i in range(0, len(hexst), 2)])

    sqrt = int(math.sqrt(len(fh)))
    pad_len = 0
    if not sqrt == 0:
        pad_len = ((sqrt+1) ** 2) - len(fh)

    pad  = np.zeros((1, pad_len))
    img = np.concatenate((fh, pad), axis=None)

    return img.reshape(sqrt+1, sqrt+1)