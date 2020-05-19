import sys
import os
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import read_binary
from Utils.Helper import rescale
from sklearn.preprocessing import StandardScaler


def mkdir(path):
    if not os.path.exists(path):
        print(f"+w {path}")
        os.mkdir(path)
        return True
    else:
        return False


def create_sub_images(X, y, rows, cols):
    # TODO: Be nice to automate this.. need some type of LCD function ...
    # not sure how to automate this yet but I know that these dims will create 10 sub images
    sub_cols = cols
    sub_rows = rows//5
    # shape of the sub images [sub_cols, sub_rows, bands]

    subimages = []
    sublabels = []
    # this will grab a sub set of the original image beginning with the top left corner, then the right top corner
    # and iteratively move down the image from left to right

    """
    Original image         subimages
    --------                --------
    |      |                [  ][  ]
    |      |                [  ][  ]
    |      |                [  ][  ]
    |      |                [  ][  ]
    |      |                [  ][  ]
    --------                --------
    """
    index = 0  # to index the container above for storing each sub image
    for row in range(5):  # represents the 5 'rows' of this image
        # represents the left and right side of the image split down the middle
        img = X[:,
                sub_rows * row: sub_rows * (row + 1),
                0: sub_cols]

        label = y[sub_rows * row: sub_rows * (row + 1),
                0: sub_cols]

        subimages.append(img)
        sublabels.append(label)
        index += 1


    return subimages, sublabels



def encode_one_hot(reference_data_root, xs, xl, array=True):
    """encodes the provided dict into a dense numpy array
    of class values.

    Caveats: The result of this encoding is dependant and naive, in that
    any conflicts of pixel labels are not intelligently resolved. For our
    purposes, at least until now, we don't care. If an instance belongs
    to multiple classes, that instance will be considered a member of
    the last class it encounters, ie, the target that comes latest
    in the dictionary
    """

    target = {
        "conifer" : "CONIFER.bin",
        "ccut" : "CCUTBL.bin",
        "water": "WATER.bin",
        "broadleaf" : "BROADLEAF.bin",
        "shrub" : "SHRUB.bin",
        "mixed" : "MIXED.bin",
        "exposed" : "EXPOSED.bin",
        "herb" : "HERB.bin",
        "river" : "Rivers.bin",
        # "road" : "ROADS.bin",
        # "vri" : "vri_s3_objid2.tif_proj.bin",
    }

    if array:
        result = np.zeros((xs*xl, len(target)))
    else:
        result = list()

    result = np.zeros((xl * xs))
    reslist = []
    for idx, key in enumerate(target.keys()):
        ones = np.ones((xl * xs))
        s, l, b, tmp = read_binary(f"{reference_data_root}/%s" % target[key])

        # same shape as the raw image
        assert int(s) == int(xs)
        assert int(l) == int(xl)

        # last index is the targets false value
        vals = np.sort(np.unique(tmp))

        # create an array populate with the false value
        t = ones * vals[len(vals) - 1]

        if key == 'water':
            arr = np.not_equal(tmp, t)
        else:
            arr = np.logical_and(tmp, t)
        # at this stage we have an array that has
        # ones where the class exists
        # and zeoes where it doesn't
        _, c = np.unique(arr, return_counts=True)
        reslist.append(c)
        result[arr > 0] = idx+1

    return result


def rawread(raw_data, raw_labels):
    cols, rows, bands, X = read_binary(f'{raw_data}/S2A.bin', to_string=False)
    y = encode_one_hot(raw_labels, cols, rows)


    X = X.reshape(bands, rows, cols)
    y = y.reshape(rows, cols)

    return X, y


def save_np(arr, path):
    np.save(path, arr)
    print(f'+w {path}')


def RGB(data_r):
    b, l, s = data_r.shape
    data_r.reshape(data_r.shape[0], data_r.shape[1] * data_r.shape[2])
    rgb = np.zeros((l, s, 3))

    for i in range(0, 3):
        rgb[:, :, i] = data_r[4 - i, :].reshape((l, s))
    for i in range(0, 3):
        rgb[:, :, i] = rescale(rgb[:, :, i])

    return rgb


def oversample(X, y, n_classes=10, extra_samples=0):
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = y.ravel()
    X_out = np.empty(X.shape)
    y_out = np.empty(y.shape)
    # retrieve the counts of the largest class
    vals, counts = np.unique(y, return_counts=True)


    print(vals)
    print(counts)

    maxval = np.amax(counts) + extra_samples
    print(f"Sampling all classes to {maxval}")

    for idx in range(n_classes):
        if(idx == 0):
            # ignore these, they aren't labeled values
            continue

        # return the true values of a class

         # creates a boolean array where true samples correspond to the index of our target idx

        indices = np.where(y == idx)


        X_ = X[:, indices]
        X_ = X_.reshape(X_.shape[0], X_.shape[2])
        y_ = y[indices]

        while y_.shape[0] < maxval:
            if y_.shape[0] == 0: # there was no label
                print(f'Warning, no values found for class {idx}')
                X_ = np.empty(X_.shape)
                y_ = np.empty(y_.shape)
                break
            if y_.shape[0] < maxval//2: # if we are less than halfway to maxval, exponential
                X_ = np.concatenate([X_, X_], axis=1)
                y_ = np.concatenate([y_, y_], axis=0)
            else: # take a subsample of size (maxval - length of oversample so far)
                X_ = np.concatenate([X_[:,:maxval-y_.shape[0]], X_], axis=1)
                y_ = np.concatenate([y_[:maxval-y_.shape[0]], y_])

        # add the oversamples to the output array
        X_out = np.concatenate([X_out, X_], axis=1)
        y_out = np.concatenate([y_out, y_], axis=0)
    print("X_os shape", X_out.shape)
    print("y_os shape", y_out.shape)

    del X_
    del y_
    return X_out, y_out


def create_paths(root):
    """--------------------------------------------------------------------
    * Initialize directory structure
    --------------------------------------------------------------------"""
    # the assumed dirs
    raw_data = os.path.join(root, "data_img")
    raw_labels = os.path.join(root, "data_bcgw")

    # our new prepared data root
    prep_dir = os.path.join(root, 'prepared') # consider this the root

    # root of the training data
    train_dir = os.path.join(prep_dir, 'train')

    # crop the training data
    crop_dir = os.path.join(train_dir, 'cropped')

    # store the original cropped and oversampled seperately
    orig_dir = os.path.join(crop_dir, 'original')
    osampled_dir = os.path.join(crop_dir, 'oversampled')

    """--------------------------------------------------------------------
    * Create Prepared Directory and Load Original Image
    --------------------------------------------------------------------"""
    if mkdir(prep_dir):
        pass
    else:
        pass

    if mkdir(train_dir):
            # make all the train images
            print("Creating Directories and Loading Data")
            X, y = rawread(raw_data, raw_labels)

            print("Full Images")
            print(f'X shape {X.shape}')
            print(f'y shape {y.shape}')

            save_np(X, f'{train_dir}/full-img.npy' )
            save_np(y, f'{train_dir}/full-label.npy')

    else:
        try:
            X = np.load(f'{train_dir}/full-img.npy')
            y = np.load(f'{train_dir}/full-label.npy')
        except Exception as e:
            print(e)
            print("NP failed to load full images")
            print("Reverting to raw read")
            X, y = rawread(raw_data, raw_labels)
            save_np(X, f'{train_dir}/full-img.npy' )
            save_np(y, f'{train_dir}/full-label.npy')


    print(f'X shape {X.shape}')
    print(f'y shape {y.shape}')

    """--------------------------------------------------------------------
    * Crop the image and labels into 5 images, save them
    --------------------------------------------------------------------"""

    subimages = []
    sublabels = []
    if mkdir(crop_dir):
        # now we have to crop X and y
        subimages, sublabels = create_sub_images(X, y, X.shape[1], X.shape[2])
        for idx, (img, label) in enumerate(zip(subimages, sublabels)):
            save_np(img, f'{crop_dir}/{idx}-data')
            save_np(label, f'{crop_dir}/{idx}-label')
    else:
        try:
            for idx in range(5):
                subimages.append(np.load(f'{crop_dir}/{idx}-data.npy'))
                sublabels.append(np.load(f'{crop_dir}/{idx}-label.npy'))
        except Exception as e:
            print(e)
            print("Failed to load cropped images")
            print("Reverting to cropping from original image")
            # if we got part way in the stashed load, make sure we
            # just reload everything from raw images
            subimages.clear()
            sublabels.clear()
            subimages, sublabels = create_sub_images(X, y, X.shape[1], X.shape[2])
            # we failed to load, let's write the crops to the cropdir so we can read
            # faster next time
            for idx, (img, label) in enumerate(zip(subimages, sublabels)):
                save_np(img, f'{crop_dir}/{idx}-data')
                save_np(label, f'{crop_dir}/{idx}-label')

    """--------------------------------------------------------------------
    * Oversample the true values to balance classese in each image, save them
    --------------------------------------------------------------------"""
    if mkdir(osampled_dir):
        for idx, (img, label) in enumerate(zip(subimages, sublabels)):
            print('Oversampling image', idx)
            X_sub, y_sub = oversample(img, label)
            save_np(X_sub, f'{osampled_dir}/{idx}-data')
            save_np(y_sub, f'{osampled_dir}/{idx}-label')
            del X_sub
            del y_sub
    else:
        try:
            for idx, (img, label) in enumerate(zip(subimages, sublabels)):
                print('Oversampling image', idx)
                X_sub, y_sub = oversample(img, label)
                save_np(X_sub, f'{osampled_dir}/{idx}-data')
                save_np(y_sub, f'{osampled_dir}/{idx}-label')
                del X_sub
                del y_sub
        except Exception as e:
            print(e)


def prepare_data(subpath):
    root_path = f"data/{subpath}/"
    reference_data_root = f"{root_path}data_bcgw/"
    raw_data_root = f"{root_path}data_img/"

    create_paths(root_path)

prepare_data('full')