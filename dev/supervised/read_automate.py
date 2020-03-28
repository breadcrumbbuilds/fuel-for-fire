import os
from Utils.Misc import *
import numpy as np
import pandas as pd

""" At the root of the project, there should be a directory named
data. Within data, specify the different organizations of your data.

Example
----------
- data
    - zoom => A directory with zoomed in images
    - full => A directory with larger images

The Hierarchy within these direcotries will be the same
- data
    - <your sub directories>
        - data_bcgw => Class Map images
        - data_img => Raw training images

"""
def read_data(subpath, collection='numpy'):

    # Define the paths we are aiming to read from
    root = os.path.join("data", subpath)
    rootimg = os.path.join(root, "data_img")
    rootbcgw = os.path.join(root, "data_bcgw")

    targets = {
        "broadleaf" : "BROADLEAF.bin",
        "ccut" : "CCUTBL.bin",
        "conifer" : "CONIFER.bin",
        "exposed" : "EXPOSED.bin",
        "herb" : "HERB.bin",
        "mixed" : "MIXED.bin",
        "river" : "Rivers.bin",
        # "road" : "Roads.bin",
        "shrub" : "SHRUB.bin",
        # "vri" : "vri.bin",
        "water" : "WATER.bin",
    }
    Sentinel = None
    Landsat = None
    # Start by reading in the training images
    try:
        if os.path.exists(f"{rootimg}/S2A.bin"):
            sentinel_cols, sentinel_rows, sentinel_bands, Sentinel = read_binary(f"{rootimg}/S2A.bin")
        if os.path.exists(f"{rootimg}/L8.bin"):
            landsat_cols, landsat_rows, landsat_bands, Landsat = read_binary(f"{rootimg}/L8.bin")
    except:
        print("error")

    ##      Start building the dataframe or numpy array

    if type(Sentinel) is not None and type(Landsat) is not None:

        # Build the shape of this puppy first then lets make a dataframe to store it
        assert int(sentinel_cols) == int(landsat_cols)
        td_cols = int(sentinel_cols)

        assert int(sentinel_rows) == int(landsat_rows)
        td_rows = int(sentinel_rows)

        td_bands = int(sentinel_bands) + int(landsat_bands)
        shape = td_cols, td_rows, td_bands

        result = buildit((Sentinel, Landsat), shape, collection, targets)

        train_data = np.concatenate(Sentinel, Landsat)
        # Now we have the right shape of this puppy
    elif type(Sentinel) is None and type(Landsat) is None:
        print("There's no damn data to read")
    elif type(Sentinel) is None and type(Landsat) is not None:
        print("No Sentinel")
    elif type(Sentinel) is not None and type(Landsat) is None:
        print("No Landsat")
    else:
        print('what the hell is going on')

        #     # # last index is the targets false value
        #     # vals = np.sort(np.unique(tmp))

        #     # # create an array populate with the false value
            # t = ones * vals[len(vals) - 1]

            # if key == 'water':
            #     arr = np.not_equal(tmp,t)
            # else:
            #     arr = np.logical_and(tmp,t)

            # # How did the caller ask for the data
            # if array:
            #     result[:,idx] = arr
            # else:
            #     result.append((key, arr))

    return result

def buildit(data, shape, collection):

    if collection == 'pandas':

        pass
    else # Assume it's numpy for now



read_data('zoom', collection='pandas')