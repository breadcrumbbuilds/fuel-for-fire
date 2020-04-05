from cuml.preprocessing.model_selection import train_test_split
from cuml.metrics.accuracy import accuracy_score
from cuml import RandomForestClassifier as cuRF
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

    ## Config
    test_size = .3
    # params for the forest
    cu_rf_params = {
        'n_estimators': 100,
        'max_depth': 16,
        'max_features': 'auto',
        'n_bins': 8,
        'split_algo': 1,
        'split_criterion': 0,
        'min_rows_per_node': 2,
        'min_impurity_decrease': 0.0,
        'bootstrap': True,
        'bootstrap_features': False,
        'verbose': False,
        'rows_sample': 1.0,
        'max_leaves': -1,
        'quantile_per_tree': False
        }

    # used to onehot encode all of the classes if need be
    target = {
        # "broadleaf" : "BROADLEAF.bin",
        # "ccut" : "CCUTBL.bin",
        # "conifer" : "CONIFER.bin",
        # "exposed" : "EXPOSED.bin",
        # "herb" : "HERB.bin",
        # "mixed" : "MIXED.bin",
        # "river" : "RIVERS.bin",
        # # "road" : "ROADS.bin",
        # "shrub" : "SHRUB.bin",
        # "vri" : "vri_s3_objid2.tif_proj.bin",
        "water" : "WATER.bin",
    }

    # Hardcoded strings to access files
    # TODO: abstract the read
    xs, xl, xb, X = read_binary(f'{raw_data_root}S2A.bin', to_string=False)

    X = X.reshape(xl * xs, xb)

    # There's harded coded paths in this function as well
    # This portion of the code is a prime candidate for
    # scrutiny
    onehot = encode_one_hot(target, xs, xl, array=True)
    onehot = onehot.astype(np.int32)

    y = onehot[:,0] # we commented out all the other samples, 0th entry is water here

    # sanity check for the encoding
    # expect to see false samples at 0. and
    # true samples at 1.
    print(np.histogram(y,bins=2))

    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_size) # scikit learn helper split function

    print(f'X_train: {X_train.shape[0]} elements')
    print(f'X_test: {X_test.shape[0]} elements')
    print(f'y_train: {y_train.shape[0]} elements')
    print(f'y_test: {y_test.shape[0]} elements')

    cu_rf = cuRF(**cu_rf_params)

    start_fit = time.time()
    cu_rf.fit(X_train, y_train)
    end_fit = time.time()

    start_predict = time.time()
    pred = cu_rf.predict(X)
    end_predict = time.time()

    fit_time = round(end_fit - start_fit,2)
    predict_time = round(end_predict - start_predict, 2)

    confmatTest = confusion_matrix(y_true=y_test, y_pred=cu_rf.predict(X_test))
    confmatTrain = confusion_matrix(y_true=y_train, y_pred=cu_rf.predict(X_train))

    train_score = cu_rf.score(X_train, y_train)
    test_score = cu_rf.score(X_test, y_test)

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
                "Max Depth: %s" % cu_rf_params['max_depth']),
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
                visualization[idx,] = [0,1,0]

            elif int(pixel[0]) and not pixel[1]:
                # False Positive
                visualization[idx,] = [1,0,0]

            elif not int(pixel[0]) and pixel[1]:
                # False Negative
                visualization[idx,] = [1,.5,0]

            elif not int(pixel[0]) and not pixel[1]:
                # True Negative
                visualization[idx, ] = [0,0,1]
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
def encode_one_hot(target, xs, xl, array=False):

    if array:
        result = np.zeros((xs*xl, len(target)))
    else:
        result = list()

    for idx, key in enumerate(target.keys()):

        ones = np.ones((xl * xs))
        s,l,b,raw = read_binary("data/zoom/data_bcgw/%s" % target[key], to_string=False)

        # same shape as the raw image
        assert s == xs
        assert l == xl

        # assume the last index is the reference's false value
        sorted_unique_raw_vals = np.sort(np.unique(raw))
        # create an array populated with the false value
        false_val_arr = ones * sorted_unique_raw_vals[len(sorted_unique_raw_vals) - 1]
        # Can't remember exactly why water is a special case here
        if key == 'water':
            arr = np.not_equal(false_val_arr, raw)
        else:
            arr = np.logical_and(false_val_arr, raw)

        # How did the caller ask for the data
        if array:
            result[:,idx] = arr
        else:
            result.append((key, arr))

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
