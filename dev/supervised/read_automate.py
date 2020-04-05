import os
from Utils.Misc import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    # the names of the target maps
    targets = {
        "broadleaf": "BROADLEAF.bin",
        "ccut": "CCUTBL.bin",
        "conifer": "CONIFER.bin",
        "exposed": "EXPOSED.bin",
        "herb": "HERB.bin",
        "mixed": "MIXED.bin",
        "river": "RIVERS.bin",
        # "road" : "Roads.bin",
        "shrub": "SHRUB.bin",
        # "vri" : "vri.bin",
        "water": "WATER.bin",
    }

    # assume neither exists
    Sentinel = None
    Landsat = None

    # Start by reading in the training images
    try:
        if os.path.exists(f"{rootimg}/S2A.bin"):
            sentinel_cols, sentinel_rows, sentinel_bands, Sentinel = read_binary(
                f"{rootimg}/S2A.bin", to_string=False)

        if os.path.exists(f"{rootimg}/L8.bin"):
            landsat_cols, landsat_rows, landsat_bands, Landsat = read_binary(
                f"{rootimg}/L8.bin", to_string=False)

    except:
        print("error reading training images")

    # Check that the data holders exist now
    if type(Sentinel) is not None and type(Landsat) is not None:

        # Assert that the two images are in fact the same dimension
        assert int(sentinel_cols) == int(landsat_cols)
        td_cols = int(sentinel_cols)

        assert int(sentinel_rows) == int(landsat_rows)
        td_rows = int(sentinel_rows)

       # Get the data into a shape we can understand (samples by channels)
        Sentinel = Sentinel.reshape(
            (sentinel_cols * sentinel_rows, sentinel_bands))
        Landsat = Landsat.reshape(
            (landsat_cols * landsat_rows, landsat_bands))

        td_bands = int(sentinel_bands) + \
            int(landsat_bands)  # the combined bands

        # we are happy with our assertions, let's store this stuff together
        shape = td_cols, td_rows, td_bands
        result = buildit((Sentinel, Landsat), shape, collection, targets)

   # Now we have the right shape of this puppy
   elif type(Sentinel) is None and type(Landsat) is None:
        print("There's no damn data to read")
   elif type(Sentinel) is None and type(Landsat) is not None:
        print("No Sentinel")

    # We only found a Sentinel data
    elif type(Sentinel) is not None and type(Landsat) is None:
        print("No Landsat")

    else:
        print('what the hell is going on')
    maps = np.zeros((td_cols * td_rows, len(targets)))
    for idx, t in enumerate(targets.keys()):
        col, row, band, Map = read_binary(
            os.path.join(rootbcgw, targets[t]), to_string=False)
        maps[:, idx] = Map.reshape((col * row))
    print(maps.shape)
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
    result = np.concatenate((result, maps), axis=1)
    return result



def buildit(data, shape, collection, targets):
    print(len(targets))

    if collection == 'pandas':
        for d in data:
            print(d.shape)
    else:
        result = np.concatenate((data[0], data[1]), axis=1)

        print(result.shape)

x = read_data('zoom', collection='pandas')

np.savetxt("small_data.csv", x, delimiter=',')
# for idx in range(x.shape[0]):
#     print(x[idx, :])
# print(x[0, :])
# for idx in range(0, x.shape[1]):
#     stack = np.hstack(x[:, idx])
#     plt.hist(stack, alpha=0.5, bins='auto')
#     plt.show()

# read_data('zoom', collection='pandas')
read_data('zoom')

