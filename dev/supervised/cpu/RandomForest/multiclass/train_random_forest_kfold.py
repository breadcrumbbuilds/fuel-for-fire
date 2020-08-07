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
# User imports
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import read_binary
from Utils.Helper import *

root_path = "data/full/prepared/train"
reference_data_root = f"{root_path}data_bcgw/"
raw_data_root = f"{root_path}data_img/"

def main():

    ## Two sets of targets, the testing image will have portions
    ## of the classes that we are not training for
    n_est = 250 # the number of estimators to fit per fold
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
    training_classes = target.keys()
    keys = list(target.keys())

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
    classes = ["unlabelled"]

    keys = list(target_all.keys())
    for key in keys:
        classes.append(key)



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

    datadir = f'{root_path}'

    X = np.load(f'{datadir}/full-img.npy')
    y = np.load(f'{datadir}/full-label.npy')

    sub_img_shape = (4835//5,3402)
    fold_length = X.shape[0] // 5
    """----------------------------------------------------------------------------------------------------------------------------
    * KFold Training
    """
    for test_idx in range(5):

        path = os.path.join(outdir, f"fold_{test_idx}")
        if not os.path.exists(path):
            os.mkdir(path)

        processing_time = {'fit': [], 'predict': []}

        X_test = X[test_idx * fold_length : (test_idx + 1)*fold_length, :]
        y_test = y[test_idx * fold_length : (test_idx + 1)*fold_length]
        preserve_shape = X_test.shape



        # Save the image as output for reference
        plt.title('Test Reference')
        colormap = plt.imshow(y_test.reshape(sub_img_shape), cmap='cubehelix')
        cbar = plt.colorbar(colormap,
                      orientation='vertical',
                     boundaries=range(11), shrink=.95,extend='max', extendrect=True, drawedges=True, spacing='uniform')
        cbar.ax.set_yticklabels(["unlabelled",
            "conifer",
            "ccut",
            "water",
            "broadleaf",
            "shrub",
            "mixed",
            "herb",
            "exposed",
            "river",""])
        plt.savefig(f'{path}/{test_idx}-reference')
        plt.close()
        print(f'+w {path}/{test_idx}-reference')

        print("Test Index", test_idx)
        print("X_test shape", X_test.shape)
        print("y_test shape", y_test.shape)



        params = {
        'n_estimators': n_est,
        'max_depth': 8,
        'verbose': 1,
        'n_jobs': -1,
        # 'bootstrap': False,
        'oob_score': True,
        'warm_start': True
        }
        clf = RandomForestClassifier(**params,)

        """----------------------------------------------------------------------------------------------------------------------------
        * Training
        """
        X_train_subs = list()
        y_train_subs = list()
        for train_idx in range(5):
            if train_idx == test_idx:
                continue
            X_train = X[train_idx * fold_length : (train_idx + 1)*fold_length, :]
            y_train = y[train_idx * fold_length : (train_idx + 1)*fold_length]

            # Save this to make some metrics later
            X_train_subs.append(X_train)
            y_train_subs.append(y_train)
            print("Training Index", train_idx)
            print("\tX_train shape", X_train.shape)
            print("\ty_train shape", y_train.shape)


            start_fit = time.time()
            clf.fit(X_train, y_train)
            end_fit = time.time()
            clf.n_estimators = clf.n_estimators + n_est
            fit_time = round(end_fit - start_fit, 2)
            processing_time['fit'].append(fit_time)

        """----------------------------------------------------------------------------------------------------------------------------
        * Prediction and Metrics
        """
        print("Test Predict")
        start_pred = time.time()
        pred = clf.predict(X_test)
        end_pred = time.time()

        predict_time = round(end_pred - start_pred, 2)
        processing_time['predict'].append(predict_time)

        print("Test Score")
        score = balanced_accuracy_score(y_test, pred)
        print("Test Confusion Matrix")
        confmatTest = confusion_matrix(
                 y_true=y_test, y_pred=pred)
        print("Test MSE")
        mse_test = mean_squared_error(y_test, pred)

        f, ax = plt.subplots(2,1, sharey=True, figsize=(30,15))
        f.suptitle("Test Reference vs Prediction")
        y_test = y_test.reshape(sub_img_shape)
        colormap_y = ax[0].imshow(y_test, cmap='cubehelix', vmin=0, vmax=12)
        ax[0].set_title('Ground Reference')
        ax[1].imshow(pred.reshape(sub_img_shape), cmap='cubehelix', vmin=0, vmax=12)
        ax[1].set_title('Prediction')


        cbar = f.colorbar(colormap_y,
                        ax=ax.ravel().tolist(),
                        orientation='vertical',
                        boundaries=range(11), shrink=.95,extend='max', extendrect=True, drawedges=True, spacing='uniform')
        cbar.ax.set_yticklabels(["unlabelled",
                "conifer",
                "ccut",
                "water",
                "broadleaf",
                "shrub",
                "mixed",
                "herb",
                "exposed",
                "river",""],fontsize=20)
        plt.savefig(f'{path}/test_predvsref')
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
        plt.savefig(f'{path}/test_confusion_matrix')
        print(f'+w {path}/test_confusion_matrix')
        plt.close()



        # We crash if we try to do the whole X_train at once
        # Sln: Chunk it up and iteratively produce results
        train_scores = 0
        mse_train = 0
        for idx, (X_train, y_train) in enumerate(zip(X_train_subs, y_train_subs)):

            print("Train Predict")
            train_pred = clf.predict(X_train)
            print("Train Confusion Matrix")
            confmat_train = confusion_matrix(y_true=y_train, y_pred=train_pred)
            print("Train Score")
            train_scores += balanced_accuracy_score(y_train, train_pred)
            print("Train MSE")
            mse_train += mean_squared_error(y_train, train_pred)


            f, ax = plt.subplots(2,1, sharey=True, figsize=(30,15))
            f.suptitle("Train Reference vs Prediction")
            y_train = y_train.reshape(sub_img_shape)
            colormap_y = ax[0].imshow(y_train, cmap='cubehelix', vmin=0, vmax=12)
            ax[0].set_title('Ground Reference')
            ax[1].imshow(train_pred.reshape(sub_img_shape), cmap='cubehelix', vmin=0, vmax=12)
            ax[1].set_title('Prediction')

            cbar = f.colorbar(colormap_y,
                            ax=ax.ravel().tolist(),
                            orientation='vertical',
                            boundaries=range(11), shrink=.95,extend='max', extendrect=True, drawedges=True, spacing='uniform')
            cbar.ax.set_yticklabels(["unlabelled",
                    "conifer",
                    "ccut",
                    "water",
                    "broadleaf",
                    "shrub",
                    "mixed",
                    "herb",
                    "exposed",
                    "river",""],fontsize=20)
            plt.savefig(f'{path}/{idx}_train_predvsref')
            plt.close()


            plt.title("Train Confusion Matrix")
            plt.matshow(confmat_train, cmap=plt.cm.Blues, alpha=0.5)
            plt.gcf().subplots_adjust(left=.5)
            for i in range(confmat_train.shape[0]):
                for j in range(confmat_train.shape[1]):
                    plt.text(x=j, y=i,
                            s=round(confmat_train[i,j],3), fontsize=6, horizontalalignment='center')
            plt.xticks(np.arange(10), labels=classes)
            plt.yticks(np.arange(10), labels=classes)
            plt.tick_params('both', labelsize=8, labelrotation=45)
            plt.xlabel('predicted label')
            plt.ylabel('reference label', rotation=90)
            plt.savefig(f'{path}/{idx}_train_confusion_matrix')
            print(f'+w {path}/{idx}_train_confusion_matrix')
            plt.close()


        score_train = round(train_scores / 4, 3)
        mse_train = round(mse_train / 4,  3)

        with open(path + "/results.txt", "w") as f:
            f.write("Score Test: " + str(score))
            f.write("\nScore Train: " +str(score_train))
            f.write("\nMSE Test: " + str(mse_test))
            f.write("\nMSE Train: " + str(mse_train))
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