from cuml.preprocessing.model_selection import train_test_split
from cuml.metrics.accuracy import accuracy_score
from cuml import RandomForestClassifier as RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
# User imports
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import read_binary
from Utils.Helper import *

root_path = "data/full/"
reference_data_root = f"{root_path}data_bcgw/"
raw_data_root = f"{root_path}data_img/"

def main():
    """----------------------------------------------------------------------------------------------------------------------------
    * Configuration
    """
    # the available configuralbe params for SKlearn
    params = {
        'n_estimators': 10000,
        'max_features': 0.3,
        'max_depth': 5,
        'verbose': 1,

        # 'class_weight': 'balanced_subsample'
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
    classes = ["unlabelled"]
    keys = list(target.keys())
    for key in keys:
        classes.append(key)


    testdir = os.path.join(root_path, "subsampled")
    traindir = os.path.join(root_path, "oversample")



    """----------------------------------------------------------------------------------------------------------------------------
    * Metrics to store for output
    """

    processing_time = {'fit': [],
                       'predict': []}

    """----------------------------------------------------------------------------------------------------------------------------
    * Output Stuff
    """
    if not os.path.exists('outs'):
        os.mkdir('outs')
    if not os.path.exists('outs/RandomForest'):
        os.mkdir('outs/RandomForest')
    outdir = 'outs/RandomForest/KFold'
    if not os.path.exists('models/RandomForest/KFold')
        os.mkdir('models/RandomForest/KFold')
    
    """----------------------------------------------------------------------------------------------------------------------------
    * KFold Training
    """
    for test_idx in range(10):

        # lets make a directory to store the fold information
        
        spatial = np.load(f'{testdir}/{test_idx}.npy')
        X = spatial.reshape(spatial.shape[0] * spatial.shape[1], spatial.shape[2])
        X_test = X[:,:11].astype(np.float32) # some loss here so we can infer on gpu
        y_test = X[:,11].astype(np.int32)

        print("Test Index", test_idx)
        print("X_test shape", X_test.shape)
        print("y_test shape", y_test.shape)

        clf = RandomForestClassifier(**params,)

        # Initialize a new random forest
        print(X.shape)

        """----------------------------------------------------------------------------------------------------------------------------
        * Train on all the training sets iteratively
        """
        for train_idx in range(10):
            if test_idx == train_idx:
                # this is our test set => move on
                continue
            X_train = np.load(f'{traindir}/{train_idx}_data.npy')
            y_train = np.load(f'{traindir}/{train_idx}_label.npy').astype(np.int32)
            print("Training Index", train_idx)
            print("\tX_train shape", X_train.shape)
            print("\ty_train shape", y_train.shape)

            start_fit = time.time()

            clf.fit(X_train, y_train)

            end_fit = time.time()
            fit_time = round(end_fit - start_fit, 2)

            processing_time['fit'].append(fit_time)
            pred = clf.predict(X_train)
            vals, counts = np.unique(pred, return_counts=True)
            score = clf.score(X_test, y_test)

            for p in zip(vals, counts):
                print(classes[int(p[0])], p[1])
            print(score)
        start_pred = time.time()

        clf.predict(X_test, y_test, predict_model='CPU')

        end_pred = time.time()


        predict_time = round(end_pred - start_pred, 2)
        processing_time['predict'].append(predict_time)

    print(processing_time)

    # ## Create the figure
    # fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    # ex = Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0) # placeholders for the legend
    # fig.legend([ex,ex,ex,ex,ex,ex,ex,ex,ex,ex],
    #            ("Target: %s" % "Water",
    #             "Test Acc.: %s" % round(test_score,3),
    #             "Train Acc.: %s" % round(train_score,3),
    #             "Test Size: %s" % test_size,
    #             "Train: %ss" % fit_time,
    #             "Predict: %ss" % predict_time,
    #             "Estimators: %s" % cu_rf_params['n_estimators'],
    #             "Max Depth: %s" % cu_rf_params['max_depth'],
    #             "Max Features: %s" % cu_rf_params['max_features']),
    #            loc='lower right',
    #            ncol=3)

    # # Reference image supplied
    # axs[0,0].set_title('Reference')
    # axs[0,0].imshow(y.reshape(xl, xs), cmap='gray')

    # # Predictions on the entire raw image visualized as black/white
    # axs[0,1].set_title('Prediction')
    # axs[0,1].imshow(pred.reshape(xl, xs), cmap='gray')

    # # Visualization of the Confusion Matrix results
    # axs[0,2].set_title('Visual ConfMatrix')
    # # build the mini legend showing what each pixel color represents
    # patches = [mpatches.Patch(color=[0,1,0], label='TP'),
    #            mpatches.Patch(color=[1,0,0], label='FP'),
    #            mpatches.Patch(color=[1,.5,0], label='FN'),
    #            mpatches.Patch(color=[0,0,1], label='TN')]
    # axs[0,2].legend(loc='upper right',
    #                 handles=patches,
    #                 ncol=2,
    #                 bbox_to_anchor=(1, -0.15)) # moves the legend outside
    # axs[0,2].imshow(visualization)

    # # Confusion matrix visualization
    # axs[1,0].set_title('Test Data Confusion Matrix')

    # axs[1,0].matshow(confmatTest, cmap=plt.cm.Blues, alpha=0.5)
    # for i in range(confmatTest.shape[0]):
    #     for j in range(confmatTest.shape[1]):
    #         axs[1,0].text(x=j, y=i,
    #                 s=round(confmatTest[i,j],3))
    # axs[1,0].set_xticklabels([0, 'False', 'True'])
    # axs[1,0].xaxis.set_ticks_position('bottom')
    # axs[1,0].set_yticklabels([0, 'False', 'True'])
    # axs[1,0].set_xlabel('predicted label')
    # axs[1,0].set_ylabel('reference label')

    # axs[1,1].set_title('Train Data Confusion Matrix')

    # axs[1,1].matshow(confmatTrain, cmap=plt.cm.Blues, alpha=0.5)
    # for i in range(confmatTrain.shape[0]):
    #     for j in range(confmatTrain.shape[1]):
    #         axs[1,1].text(x=j, y=i,
    #                 s=round(confmatTrain[i,j],3))
    # axs[1,1].set_xticklabels([0, 'False', 'True'])
    # axs[1,1].xaxis.set_ticks_position('bottom')
    # axs[1,1].set_yticklabels([0, 'False', 'True'])
    # axs[1,1].set_xlabel('predicted label')
    # axs[1,1].set_ylabel('reference label')
    # axs[1,1].margins(x=10)

    # plt.tight_layout()

    # if not os.path.exists('outs'):
    #     print('creating outs directory in root')
    #     os.mkdir('outs')
    # if not os.path.exists('outs/RandForest/'):
    #     pr
    #     os.mkdir('outs/RandForest/')
    # plt.savefig('outs/RandForest/%s.png' % f"cuML_RF_{cu_rf_params['n_estimators']}trees_{cu_rf_params['max_depth']}maxdepth")
    # plt.show()


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
