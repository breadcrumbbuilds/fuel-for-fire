import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
import math

label = np.load('data/full/prepared/train/full-label.npy')
counter = 0
# for l in label:
#     for s in l:
#         print(s)
#     counter += 1
# print(counter)
plt.title("Dataset B Reference")
colormap = plt.imshow(label, cmap='cubehelix')


cbar = plt.colorbar(colormap)


cbar.set_ticks(list())



cbar.set_ticks(range(10))


cbar.set_ticklabels(["unlabelled",
        "conifer",
        "ccut",
        "water",
        "broadleaf",
        "shrub",
        "mixed",
        "herb",
        "exposed",
        "river"])


# for index, label in enumerate(()):

#     x = 1

#     y = index

#     cbar.ax.text(x, y, label)
plt.tight_layout()
plt.show()
