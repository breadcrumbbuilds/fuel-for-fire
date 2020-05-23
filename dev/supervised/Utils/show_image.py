import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
import math

def rescale(arr, two_percent=True):
    arr_min = arr.min()
    arr_max = arr.max()
    scaled = (arr - arr_min) / (arr_max - arr_min)

    if two_percent:
        # 2%-linear stretch transformation for hi-contrast vis
        values = copy.deepcopy(scaled)
        values = values.reshape(np.prod(values.shape))
        values = values.tolist()
        values.sort()
        npx = len(values)  # number of pixels
        if values[-1] < values[0]:
            print('error: failed to sort')
            sys.exit(1)
        v_min = values[int(math.floor(float(npx)*0.02))]
        v_max = values[int(math.floor(float(npx)*0.98))]
        scaled -= v_min
        rng = v_max - v_min
        if rng > 0.:
            scaled /= (v_max - v_min)

    return scaled

def visRGB(X, shape=(0,0,0)):
    b = shape[0]
    l = shape[1]
    s = shape[2]
    data_r = X.reshape(b, s * l)
    rgb = np.zeros((l, s, 3))

    for i in range(0, 3):
        rgb[:, :, i] = data_r[4 - i, :].reshape((l, s))
    for i in range(0,3):
        rgb[:, :, i] = rescale(rgb[:, :, i])
    del X
    return rgb

img = np.load('data//full/prepared/train/full-img.npy')
img = visRGB(img, shape=img.shape)

label = np.load('data/full/prepared/train/full-label.npy')

plt.title("Dataset B")
plt.imshow(img)
plt.tight_layout()
plt.show()
