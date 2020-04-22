import cv2
from keras import utils
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
import time
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import read_binary
from Utils.Helper import rescale, create_batch_generator
# TO VIEW MODEL PERFORMANCE IN BROWSER RUN
#
# FROM ROOT DIR
# globals
root_path = "data/full/"
reference_data_root = f"{root_path}data_bcgw/"
raw_data_root = f"{root_path}data_img/"


def main():

    test = True

    print("Numpy Version: %s" % np.__version__)
    print("Keras Version: %s" % keras.__version__)

    """----------------------------------------------------------------------------------------------------------------------------
    * Configuration
    """
    # Config
    test_size = .20

    if test:
        epochs = 1
        batch_size = 8192
    else:
        epochs = 1000
        batch_size = 1024

    learning_rate = 0.00333
    n_folds = 10
    METRICS = [
        keras.metrics.CategoricalAccuracy(name='cat_acc'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    target = {
        "conifer": "CONIFER.bin",
        "ccut": "CCUTBL.bin",
        "water": "WATER.bin",
        "broadleaf": "BROADLEAF.bin",
        "shrub": "SHRUB.bin",
        "mixed": "MIXED.bin",
        "herb": "HERB.bin",
        "exposed": "EXPOSED.bin",
        "river": "Rivers.bin",
        # "road" : "ROADS.bin",
        # "vri" : "vri_s3_objid2.tif_proj.bin",
    }

# ## Preprocess
    cols, rows, bands, X = read_binary(
        f'{raw_data_root}S2A.bin', to_string=False)
    X = X.reshape(cols * rows, bands)
    X = StandardScaler().fit_transform(X)  # standardize unit variance and 0 mean
    onehot = encode_one_hot(target, cols, rows, array=True)

    """-----------------------------------------------------------
    * Cross Val
    """
    # store the image together
    tmp = np.zeros((X.shape[0], X.shape[1] + 1))
    tmp[:, :X.shape[1]] = X
    tmp[:, X.shape[1]] = onehot
    # make sure the input is spatially sound
    tmp = tmp.reshape((cols, rows, bands + 1))

    checkit(tmp)

    X_subbed = create_sub_images(tmp, cols, rows, bands+1)

    checkit(X_subbed)

    X = X_subbed[:3, :,:,:]
    X_train = X[:,:,:,:11]
    y_train = X[:,:,:,11]

    X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1] * X_train.shape[2], X_train.shape[3])
    y_train = y_train.reshape(y_train.shape[0] * y_train.shape[1] * y_train.shape[2])

    print(np.unique(y_train, return_counts=True))
    # for idx in range(n_folds):
    #     # the sub image that will be considered the training set
    #     X_test = X[idx, :, :, :11].reshape(
    #         X.shape[1] * X.shape[2], X.shape[3] - 1)
    #     y_test = X[idx, :, :, 11].reshape(X.shape[1] * X.shape[2])

    #     checkit(X_test)
    #     checkit(y_test)

    #     checkit(X_train)
        # X_train = X # everything else
    # for i in range(n_folds):

    #     cv2.imshow("Reference", X_subbed[i,:,:,3 ])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     checkit(X_subbed[i,:,:,:])

def checkit(passed):
    print()
    print("CHECKIT")
    print(type(passed))
    print(passed)
    print(passed.shape)
    print()


def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run__%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def one_hot_sanity_check(target, xs, xl):
    oh = encode_one_hot(target, xs, xl)
    for idx, val in enumerate(oh):
        print(val[0])
        plt.title(val[0])
        plt.imshow(val[1].reshape(xl, xs), cmap='gray')
        plt.show()


def create_sub_images(X, cols, rows, bands):
    # TODO: Be nice to automate this.. need some type of LCD function ...
    # not sure how to automate this yet but I know that these dims will create 10 sub images
    sub_cols = cols//2
    sub_rows = rows//5
    # shape of the sub images [sub_cols, sub_rows, bands]
    print("New subimage shape (%s, %s, %s)" % (sub_cols, sub_rows, bands))

    # container for the sub images
    sub_images = np.zeros((10, sub_cols, sub_rows, bands))

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
        for col in range(2):
            checkit(X)
            sub_images[index, :, :, :] = X[sub_cols * col: sub_cols *
                                           (col + 1), sub_rows * row: sub_rows * (row + 1), :]
            index += 1

    print("images, width, height, features", sub_images.shape)
    return sub_images


def oversample(X_train, y_train, n_classes=10, extra_samples=50000):

    tmp = np.zeros((X_train.shape[0], X_train.shape[1] + 1))

    tmp[:, :X_train.shape[1]] = X_train
    tmp[:, X_train.shape[1]] = y_train

    # Let's oversample each class so we don't have class imbalance
    vals, counts = np.unique(tmp[:, X_train.shape[1]], return_counts=True)
    maxval = np.amax(counts) + extra_samples

    # WARNING, your validation data has leakage,
    for idx in range(n_classes):
        if(idx == 0):
            # ignore these, they aren't labeled values
            continue

        # return the true values of a class
        idx_class_vals_outside_while = tmp[tmp[:, X_train.shape[1]] == idx]

        # oversample until we have n samples
        while(tmp[tmp[:, X_train.shape[1]] == idx].shape[0] < maxval):

            # this grows exponentially
            idx_class_vals_inside_while = tmp[tmp[:, X_train.shape[1]] == idx]
            # if we are halfway there, let's ease up and do things slower
            # so our classes have similar amounts of samples
            if idx_class_vals_inside_while.shape[0] > maxval//2:
                tmp = np.concatenate(
                    (tmp, idx_class_vals_outside_while), axis=0)
            else:
                tmp = np.concatenate(
                    (tmp, idx_class_vals_inside_while), axis=0)
    X_train = tmp[:, :X_train.shape[1]]
    y_train = tmp[:, X_train.shape[1]]

    return X_train, y_train


def encode_one_hot(target, xs, xl, array=True):
    """encodes the provided dict into a dense numpy array
    of class values.

    Caveats: The result of this encoding is dependant and naive, in that
    any conflicts of pixel labels are not intelligently resolved. For our
    purposes, at least until now, we don't care. If an instance belongs
    to multiple classes, that instance will be considered a member of
    the last class it encounters, ie, the target that comes latest
    in the dictionary
    """
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


def build_vis(prediction, y, shape):

    visualization = np.zeros((len(y), 3))
    for idx, pixel in enumerate(zip(prediction, y)):

        # compare the prediciton to the original
        if pixel[0] and pixel[1]:
                # True Positive
            visualization[idx, ] = [0, 1, 0]

        elif pixel[0] and not pixel[1]:
            # False Positive
            visualization[idx, ] = [1, 0, 0]

        elif not pixel[0] and pixel[1]:
            # False Negative
            visualization[idx, ] = [1, .5, 0]

        elif not pixel[0] and not pixel[1]:
            # True Negative
            visualization[idx, ] = [0, 0, 1]
            # visualization[idx, ] = rgb

        else:
            raise Exception("There was a problem comparing the pixel", idx)

    return visualization.reshape(shape)



if __name__ == "__main__":

    main()
