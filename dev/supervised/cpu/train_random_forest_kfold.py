from sklearn.ensemble import RandomForestClassifier
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import time
# User imports
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import read_binary
from Utils.Helper import *

"""----------------------------------------------------------------------------------------------------------------------------
* Train KFolded Random Forest and output figures to describe the RF's performance
*
* Dependencies:
*       1. Ensure the root of the project contains the following folder structure
*       data
*           full
*               data_bcgw
*                   <Nine class labels of the large image size - naming convention following the 'target' dictionary in this script>
*               data_img
*                   <Binary of the large image>
*       2. Run helpme.py from the root of the project
*       3. Set your parameters within this script
"""



root_path = "data/full/prepared/train/cropped"
reference_data_root = f"{root_path}data_bcgw/"
raw_data_root = f"{root_path}data_img/"

def main():
    """----------------------------------------------------------------------------------------------------------------------------
    * Configuration
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
    classes = ['unlabelled']
    keys = list(target.keys())
    for key in keys:
        classes.append(key)

    # the available configuralbe params for SKlearn
    params = {
                'n_estimators': 0,
                'max_features': 0.1,
                'max_depth': 5,
                'verbose': 1,
                'n_jobs': -1,
                # 'bootstrap': False,
                'oob_score': True,
                'warm_start': True
                }
    # used to onehot encode all of the classes if need be


    outdir = os.path.join(os.curdir,'outs')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdir = os.path.join(outdir, 'RandomForest')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdir = os.path.join(outdir, 'KFold')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdir = get_run_logdir(outdir)
    os.mkdir(outdir)
    testdir = os.path.join(root_path, "subsampled")
    traindir = os.path.join(root_path, "oversample")


    """----------------------------------------------------------------------------------------------------------------------------
    * KFold Training
    """
    for test_idx in range(5):
        clf = RandomForestClassifier(**params)

        path = os.path.join(outdir, f"fold_{test_idx}")
        if not os.path.exists(path):
            os.mkdir(path)


        processing_time = {'fit': [], 'predict': []}
        X_test = np.load(f'{root_path}/{test_idx}-data.npy')
        y_test = np.load(f'{root_path}/{test_idx}-label.npy')
        print(X_test.shape)

        rows, cols = y_test.shape

        plt.title('Reference')
        plt.imshow(y_test, cmap='gray')
        plt.savefig(f'{path}/{test_idx}-reference')
        plt.close()
        print(f'+w {path}/{test_idx}-reference')

        X_test = X_test.reshape(X_test.shape[1] * X_test.shape[2], X_test.shape[0])
        y_test = y_test.ravel()
        # y_test = keras.utils.to_categorical(y_test, num_classes=len(target)+1)


        print("Test Index", test_idx)
        print("X_test shape", X_test.shape)
        print("y_test shape", y_test.shape)
        vals, counts = np.unique(y_test, return_counts=True)
        for v in zip(vals, counts):
            print(v)

        """----------------------------------------------------------------------------------------------------------------------------
        * Train on all the training sets iteratively, see the data three times per training inex
        """
        for p in range(3):
            for train_idx in range(5):

                if test_idx == train_idx:
                    # this is our test set => move on
                    continue
                clf.n_estimators += 1000

                X_train = np.load(f'{root_path}/{train_idx}-data.npy')
                y_train = np.load(f'{root_path}/{train_idx}-label.npy')
                X_train = X_train.reshape(X_train.shape[1] * X_train.shape[2], X_train.shape[0])
                y_train = y_train.ravel()
                # y_train = keras.utils.to_categorical(y_train, num_classes=len(target) + 1)

                print("Pass #", p)
                print("Training Index", train_idx)
                print("\tX_train shape", X_train.shape)
                print("\ty_train shape", y_train.shape)
                vals, counts = np.unique(y_train, return_counts=True)
                for x in zip(vals, counts):
                    print(x)

                start_fit = time.time()

                clf.fit(X_train, y_train)

                end_fit = time.time()
                fit_time = round(end_fit - start_fit, 2)

                processing_time['fit'].append(fit_time)
                pred = clf.predict(X_test)

                confmatTest = confusion_matrix(
                    y_true=y_test, y_pred=pred)
                score = clf.score(X_test, y_test)

                plt.title('Prediction')
                plt.imshow(pred.reshape(rows, cols) / 255, cmap='gray')
                plt.savefig(f'{path}/{train_idx}-prediction')
                plt.close()
                print(f'+w {path}/{train_idx}-prediction')

                plt.title("Test Confusion Matrix")
                plt.matshow(confmatTest, cmap=plt.cm.Blues, alpha=0.5)
                plt.gcf().subplots_adjust(left=.5)
                for i in range(confmatTest.shape[0]):
                    for j in range(confmatTest.shape[1]):
                        plt.text(x=j, y=i,
                                s=round(confmatTest[i,j],3), fontsize=6, horizontalalignment='center')
                plt.xticks(np.arange(10), labels=classes)
                plt.yticks(np.arange(10), labels=classes)
                plt.tick_params('both', labelsize=8, labelrotation=45)
                plt.xlabel('predicted label')
                plt.ylabel('reference label', rotation=90)
                plt.savefig(f'{path}/{train_idx}-test_confusion_matrix')
                print(f'+w {path}/{train_idx}-test_confusion_matrix')
                plt.close()
                with open(path + "results.txt", "w") as f:
                    f.write("Score: " + str(score))
                    f.write("\nProcessing Times:")
                    f.write(json.dumps(processing_time, indent=4, separators=(',', ': ')))
                    f.write("\nOob Score: " + str(clf.oob_score_))



        start_pred = time.time()

        pred = clf.predict(X_test)

        end_pred = time.time()

        predict_time = round(end_pred - start_pred, 2)
        processing_time['predict'].append(predict_time)

        confmatTest = confusion_matrix(
                 y_true=y_test, y_pred=pred)
        score = clf.score(X_test, y_test)

        plt.title('Prediction')
        plt.imshow(pred.reshape(rows, cols), cmap='gray')
        plt.savefig(f'{path}/final-prediction')
        print(f'+w {path}/f    binal-prediction')
        plt.close()

        plt.title("Test Confusion Matrix")
        plt.matshow(confmatTest, cmap=plt.cm.Blues, alpha=0.5)
        plt.gcf().subplots_adjust(left=.5)
        for i in range(confmatTest.shape[0]):
            for j in range(confmatTest.shape[1]):
                plt.text(x=j, y=i,
                        s=round(confmatTest[i,j],3), fontsize=6, horizontalalignment='center')
        plt.xticks(np.arange(10), labels=classes)
        plt.yticks(np.arange(10), labels=classes)
        plt.tick_params('both', labelsize=8, labelrotation=45)
        plt.xlabel('predicted label')
        plt.ylabel('reference label', rotation=90)
        plt.savefig(f'{path}/final-test_confusion_matrix')
        print(f'+w {path}/final-test_confusion_matrix')
        plt.close()

        with open(path + "/results.txt", "w") as f:
                f.write("Score: " + str(score))
                f.write("\nProcessing Times:")
                f.write(json.dumps(processing_time, indent=4, separators=(',', ': ')))
                f.write("\nOob Score: " + str(clf.oob_score_))
                f.write("\nFeature Importance: " + str(clf.feature_importances_))


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

def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run__%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


if __name__ == "__main__":

   main()
