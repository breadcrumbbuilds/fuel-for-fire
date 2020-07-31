from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score

from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import time
import os
# User imports
from Misc import read_binary
""" Relies on prep_data to read the base image in"""
root = "data/full/"
train_root_path = f"{root}/prepared/train"
reference_data_root = f"data/full/data_bcgw/"
raw_data_root = f"{root}data_img/"

def main():

    ## Two sets of targets, the testing image will have portions
    ## of the classes that we are not training for
    n_est = 250 # the number of estimators to fit per fold
    target_all = {
        "conifer" : "CONIFER.bin",
        "ccut" : "CCUTBL.bin",
        "water": "WATER.bin",
        "broadleaf" : "BROADLEAF.bin",
        "shrub" : "SHRUB.bin",
        "mixed" : "MIXED.bin",
        "herb" : "HERB.bin",
        "exposed" : "EXPOSED.bin",
        "river" : "Rivers.bin",
        # "road" : "ROADS.bin",
        # "vri" : "vri_s3_objid2.tif_proj.bin",
    }

    for target in target_all.keys():
        reference_path = f'{reference_data_root}{target_all[target]}'
        if os.path.exists(reference_path):
            cols, rows, bands, y = read_binary(reference_path, to_string=False)
            f, ax = plt.subplots(1,2, sharex=True)
            f.suptitle(target)
            ax[0].imshow(y.reshape(rows, cols), cmap='gray')
            ax[0].set_title("Raw")
            # encode the values to 0/1
            ones = np.ones((cols * rows))

            vals = np.sort(np.unique(y))

            # create an array populate with the false value
            t = ones * vals[len(vals) - 1]

            if target == 'water':
                y = np.not_equal(y, t)
            else:
                y = np.logical_and(y, t)

            # at this stage we have an array that has
            # ones where the class exists
            # and zeoes where it doesn't
            ax[1].imshow(y.reshape(rows, cols), cmap='gray')
            ax[1].set_title("Encoded")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":

   main()
