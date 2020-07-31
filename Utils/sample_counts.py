import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
import math
from matplotlib.ticker import MaxNLocator
img = np.load('data//full/prepared/train/full-label.npy')

vals, counts = np.unique(img.ravel(), return_counts=True)


ax = plt.figure().gca()

print(counts)
max_count = np.argmax(counts)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}


ax.bar(["unlabelled",
        "conifer",
        "ccut",
        "water",
        "broadleaf",
        "shrub",
        "mixed",
        "herb",
        "exposed",
        "river"], counts)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.tick_params(axis='both', which='major', labelsize=12)
plt.ylabel('Counts', fontsize=16)
plt.xlabel('Class', fontsize=16)
# ax.tight_layout()
plt.title("Dataset B Sample Counts")
plt.show()
