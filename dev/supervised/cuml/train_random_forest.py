from cuml.preprocessing.model_selection import train_test_split
from cuml.metrics.accuracy import accuracy_score
from cuml import RandomForestClassifier as cuRF
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
import cuml
# User imports
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import read_binary
from Utils.Helper import *

root_path = "data/zoom/"
reference_data_root = f"{root_path}data_bcgw/"
raw_data_root = f"{root_path}data_img/"

def main():
    print("Numpy Version: %s" % np.__version__)
    print("cuML Version: %s" % cuml.__version__)

    test_size = .3 # float representing the percentage of test samples split
    # the available configuralbe params for cuML
    cu_rf_params = {
        'n_estimators': 10000,
        'max_depth': 3,
        'max_features': 0.1,
        }

    # used to onehot encode all of the classes if need be
    target = {
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

    xs, xl, xb, X = read_binary(f'{raw_data_root}S2A.bin', to_string=False)

    X = X.reshape(xl * xs, xb)
    X = StandardScaler().fit_transform(X)  # standardize unit variance and 0 mean



    # Load the labels for the binary classifier we aim to train
    y = encode_one_hot(target, xs, xl, array=True)

    y = y.reshape(int(xl)*int(xs))
    y = y.astype(np.int32)

    print(f"Onehot shape: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True)
     # Deal with imabalanced classes by oversampleing the training set samples
    # let's store the vals and labels together
    tmp = np.zeros((X_train.shape[0], X_train.shape[1] + 1))
    tmp[:,:X_train.shape[1]] = X_train
    tmp[:,X_train.shape[1]] = y_train

    # Let's oversample each class so we don't have class imbalance
    vals, counts = np.unique(tmp[:,X_train.shape[1]], return_counts=True)
    maxval = np.amax(counts) + 50000
    for idx in range(len(target) + 1):
        if(idx == 0):
            # ignore these, they aren't labeled values
            continue

        idx_class_vals_outside_while = tmp[tmp[:,X_train.shape[1]] == idx] # return the true values of a class

        while(tmp[tmp[:,X_train.shape[1]] == idx].shape[0] < maxval): # oversample until we have n samples

            idx_class_vals_inside_while = tmp[tmp[:,X_train.shape[1]] == idx] # this grows exponentially
            # if we are halfway there, let's ease up and do things slower
            # so our classes have similar amounts of samples
            if idx_class_vals_inside_while.shape[0] > maxval//2:
                tmp = np.concatenate((tmp, idx_class_vals_outside_while), axis=0)
            else:
                tmp = np.concatenate((tmp, idx_class_vals_inside_while), axis=0)

    vals, counts = np.unique(tmp[:,X_train.shape[1]], return_counts=True)
    print(vals)
    print(counts)

    X_train = tmp[:,:X_train.shape[1]]
    y_train = tmp[:,X_train.shape[1]]
    y_train = y_train.astype(np.int32)
    forest = cuRF(**cu_rf_params)

    start_fit = time.time()
    forest.fit(X_train, y_train)
    end_fit = time.time()

    start_predict = time.time()
    pred = forest.predict(X, predict_model='CPU')
    end_predict = time.time()

    fit_time = round(end_fit - start_fit,2)
    predict_time = round(end_predict - start_predict, 2)

    confmatTest = confusion_matrix(y_true=y_test, y_pred=forest.predict(X_test))
    confmatTrain = confusion_matrix(y_true=y_train, y_pred=forest.predict(X_train))

    train_score = forest.score(X_train, y_train)
    test_score = forest.score(X_test, y_test)

    # user made function to produce a visual of True pos, False Pos, False Neg, and True Neg
    visualization = build_vis(pred,y, (xl,xs, 3))


    ## Create the figure
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    ex = Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0) # placeholders for the legend
    fig.legend([ex,ex,ex,ex,ex,ex,ex,ex,ex,ex],
               ("Target: %s" % "Water",
                "Test Acc.: %s" % round(test_score,3),
                "Train Acc.: %s" % round(train_score,3),
                "Test Size: %s" % test_size,
                "Train: %ss" % fit_time,
                "Predict: %ss" % predict_time,
                "Estimators: %s" % cu_rf_params['n_estimators'],
                "Max Depth: %s" % cu_rf_params['max_depth'],
                "Max Features: %s" % cu_rf_params['max_features']),
               loc='lower right',
               ncol=3)

    # Reference image supplied
    axs[0,0].set_title('Reference')
    axs[0,0].imshow(y.reshape(xl, xs), cmap='gray')

    # Predictions on the entire raw image visualized as black/white
    axs[0,1].set_title('Prediction')
    axs[0,1].imshow(pred.reshape(xl, xs), cmap='gray')

    # Visualization of the Confusion Matrix results
    axs[0,2].set_title('Visual ConfMatrix')
    # build the mini legend showing what each pixel color represents
    patches = [mpatches.Patch(color=[0,1,0], label='TP'),
               mpatches.Patch(color=[1,0,0], label='FP'),
               mpatches.Patch(color=[1,.5,0], label='FN'),
               mpatches.Patch(color=[0,0,1], label='TN')]
    axs[0,2].legend(loc='upper right',
                    handles=patches,
                    ncol=2,
                    bbox_to_anchor=(1, -0.15)) # moves the legend outside
    axs[0,2].imshow(visualization)

    # Confusion matrix visualization
    axs[1,0].set_title('Test Data Confusion Matrix')

    axs[1,0].matshow(confmatTest, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(confmatTest.shape[0]):
        for j in range(confmatTest.shape[1]):
            axs[1,0].text(x=j, y=i,
                    s=round(confmatTest[i,j],3))
    axs[1,0].set_xticklabels([0, 'False', 'True'])
    axs[1,0].xaxis.set_ticks_position('bottom')
    axs[1,0].set_yticklabels([0, 'False', 'True'])
    axs[1,0].set_xlabel('predicted label')
    axs[1,0].set_ylabel('reference label')

    axs[1,1].set_title('Train Data Confusion Matrix')

    axs[1,1].matshow(confmatTrain, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(confmatTrain.shape[0]):
        for j in range(confmatTrain.shape[1]):
            axs[1,1].text(x=j, y=i,
                    s=round(confmatTrain[i,j],3))
    axs[1,1].set_xticklabels([0, 'False', 'True'])
    axs[1,1].xaxis.set_ticks_position('bottom')
    axs[1,1].set_yticklabels([0, 'False', 'True'])
    axs[1,1].set_xlabel('predicted label')
    axs[1,1].set_ylabel('reference label')
    axs[1,1].margins(x=10)

    plt.tight_layout()

    if not os.path.exists('outs'):
        print('creating outs directory in root')
        os.mkdir('outs')
    if not os.path.exists('outs/RandForest/'):
        pr
        os.mkdir('outs/RandForest/')
    plt.savefig('outs/RandForest/%s.png' % f"cuML_RF_{cu_rf_params['n_estimators']}trees_{cu_rf_params['max_depth']}maxdepth")
    plt.show()


"""Builds a visualization of the confusion matrix
"""
def build_vis(prediction, y, shape):

    visualization = np.zeros((len(y), 3))
    idx = 0
    try:
        for pixel in zip(prediction, y):

            if int(pixel[0]) and pixel[1]:
                # True Positive
                visualization[idx,] = [0,idx,0]

            elif int(pixel[0]) and not pixel[1]:
                # False Positive
                visualization[idx,] = [idx,0,0]

            elif not int(pixel[0]) and pixel[1]:
                # False Negative
                visualization[idx,] = [idx,idx//2,0]

            elif not int(pixel[0]) and not pixel[1]:
                # True Negative
                visualization[idx, ] = [0,0,idx]
                # visualization[idx, ] = rgb
            else:
                raise Exception("There was a problem predicting the pixel", idx)

            idx += 1
    except:
        # on the final iteration, an error is thrown, but our visualization has
        # been built correctly so let's ignore it for now
        pass

    return visualization.reshape(shape)


"""Automated determination of a binary interpretation of the
pixels within the reference images
"""
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
        s,l,b,tmp = read_binary(f"{reference_data_root}/%s" % target[key])

        # same shape as the raw image
        assert int(s) == int(xs)
        assert int(l) == int(xl)

        # last index is the targets false value
        vals = np.sort(np.unique(tmp))

        # create an array populate with the false value
        t = ones * vals[len(vals) - 1]

        if key == 'water':
            arr = np.not_equal(tmp,t)
        else:
            arr = np.logical_and(tmp,t)
        # at this stage we have an array that has
        # ones where the class exists
        # and zeoes where it doesn't
        _, c = np.unique(arr, return_counts=True)
        reslist.append(c)
        result[arr > 0] = idx+1
        # # How did the caller ask for the data
        # if array:
        #     result[:,idx] = arr
        # else:
        #     result.append((key, arr))

    # vals, counts = np.unique(result, return_counts=True)
    # print(vals)
    # print(counts)
    # print(reslist)
    # print(target.keys())
    return result

"""Use this if you want to show the raw image in the output
"""
def visRGB(s, l, b,X):

    data_r = X.reshape(b, s * l)
    rgb = np.zeros((l, s, 3))

    for i in range(0, 3):
        rgb[:, :, i] = data_r[3 - i, :].reshape((l, s))
    for i in range(0,3):
        rgb[:, :, i] = rescale(rgb[:, :, i])
    del X
    return rgb


if __name__ == "__main__":

   main()
