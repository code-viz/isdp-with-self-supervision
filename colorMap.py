import os
import binascii
import numpy as np

def getFileSize(filepath):
    f_size = os.path.getsize(filepath)
    size = f_size/float(1024)
    return round(size, 2)

def getRectangleGrayImg(filename, width=32):
    size = getFileSize(filename)

    with open(filename, 'rb') as f:
        content = f.read()

    hexst = binascii.hexlify(content)
    fh = np.array([int(hexst[i:i+2], 16) for i in range(0, len(hexst), 2)])
    
    pad_len = width - len(fh) % width
    pad = np.zeros((1, pad_len))
    img = np.concatenate((fh, pad), axis=None)

    return img.reshape(-1, width)
