import cuml
from cuml import RandomForestClassifier as cuRF
from cuml.preprocessing.model_selection import train_test_split
from cuml.metrics.accuracy import accuracy_score
import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import read_binary
from sklearn.model_selection import train_test_split
import cudf
import dask_cudf
import time
from Utils.Helper import *
import cuml
from cuml import RandomForestClassifier as cuRF
from cuml.preprocessing.model_selection import train_test_split
from cuml.metrics.accuracy import accuracy_score
import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import read_binary
from sklearn.model_selection import train_test_split
import cudf
import dask_cudf
import time
from Utils.Helper import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import multiprocessing
import pickle
import time
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def main():
    print("Numpy Version: %s" % np.__version__)
    print("cuML Version: %s" % cuml.__version__)

    ## Config
    test_size = .5
    n_estimators = 3
    n_features = .9
    max_depth = 24

    ## Load Data
    targetB = {
        "broadleaf" : "BROADLEAF_SP.tif_proj.bin",
        "ccut" : "CCUTBL_SP.tif_proj.bin",
        "conifer" : "CONIFER_SP.tif_proj.bin",
        "exposed" : "EXPOSED_SP.tif_proj.bin",
        "herb" : "HERB_GRAS_SP.tif_proj.bin",
        "mixed" : "MIXED_SP.tif_proj.bin",
        "river" : "RiversSP.tif_proj.bin",
        # "road" : "RoadsSP.tif_proj.bin",
        "shrub" : "SHRUB_SP.tif_proj.bin",
        # "vri" : "vri_s3_objid2.tif_proj.bin",
        "water" : "WATERSP.tif_proj.bin",
    }
    target = {
        # "broadleaf" : "BROADLEAF_SP.tif_project_4x.bin_sub.bin",
        # "ccut" : "CCUTBL_SP.tif_project_4x.bin_sub.bin",
        # "conifer" : "CONIFER_SP.tif_project_4x.bin_sub.bin",
        # "exposed" : "EXPOSED_SP.tif_project_4x.bin_sub.bin",
        # "herb" : "HERB_GRAS_SP.tif_project_4x.bin_sub.bin",
        # "mixed" : "MIXED_SP.tif_project_4x.bin_sub.bin",
        # "river" : "RiversSP.tif_project_4x.bin_sub.bin",
        # # "road" : "RoadsSP.tif_proj.bin",
        # "shrub" : "SHRUB_SP.tif_project_4x.bin_sub.bin",
        # "vri" : "vri_s3_objid2.tif_proj.bin",
        "water" : "WATERSP.tif_project_4x.bin_sub.bin",
    }
 # output4_selectS2.bin
 # S2A.bin_4x.bin_sub.bin
    xs, xl, xb, X = read_binary('data/elhill/data_img/S2A.bin_4x.bin_sub.bin')
    # xs, xl, xb2, X2 = read_binary('data/elhill/data_img/L8.bin_4x.bin_sub.bin')
    xs = int(xs)

    xl = int(xl)
    xb = int(xb)
    # X = np.concatenate((X,X2), axis=0)
    X = X.reshape(xl * xs, xb)
    # X = X.reshape(xl*xs, xb + int(xb2))
    print(X.shape)

    onehot = encode_one_hot(target, xs, xl, array=True)
    onehot = onehot.astype(np.int32)
    y = onehot[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size)

    # mean =  np.mean(X_train)
    # std = np.std(X_train)
    # X_train_c = (X_train - mean) / std
    # X_test_c = (X_test - mean) / std
    # X = (X - mean) / std
    print(f'X_train: {X_train.shape[0]} elements')
    print(f'X_test: {X_test.shape[0]} elements')
    print(f'y_train: {y_train.shape[0]} elements')
    print(f'y_test: {y_test.shape[0]} elements')

    cu_rf_params = {
        'n_estimators': n_estimators,
        'max_features': n_features,
        'max_depth' : max_depth,
        'split_algo': 0
        }

    cu_rf = cuRF(**cu_rf_params)

    start_fit = time.time()
    cu_rf.fit(X_train, y_train)
    end_fit = time.time()
    fit_time = round(end_fit - start_fit,2)


    start_predict = time.time()
    pred = cu_rf.predict(X)
    end_predict = time.time()
    predict_time = round(end_predict - start_predict,2)

    print("time to fit", fit_time)
    print("time to pred", predict_time) # done on gpu?

    print(np.bincount(pred))
    confmatTest = confusion_matrix(y_true=y_test, y_pred=cu_rf.predict(X_test))
    confmatTrain = confusion_matrix(y_true=y_train, y_pred=cu_rf.predict(X_train))

    #importances = cu_rf.feature_importances_
    # indices = np.argsort(importances)[::-1]

    train_score = cu_rf.score(X_train, y_train)
    test_score = cu_rf.score(X_test, y_test)

    print(train_score)
    print(test_score)

    visualization = build_vis(pred,y, (xl,xs, 3))


    fig, axs = plt.subplots(2, 3, figsize=(9, 6), sharey=False)
    # plt.subplots_adjust(right=.5, top=3)
    ex = Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
    fig.legend([ex,ex,ex,ex,ex,ex,ex,ex],
               ("Target: %s" % "Water",
                "Test Acc.: %s" % round(test_score,3),
                "Train Acc.: %s" % round(train_score,3),
                "Test Size: %s" % test_size,
                "Train: %s" % fit_time,
                "Predict: %s" % predict_time,
                "Estimators: %s" % n_estimators,
                "Max Features: %s" % n_features),
               loc='center left',
               ncol=4)
    axs[0,0].set_title('Sentinel2')
    axs[0,0].imshow(visRGB(xs,xl,xb,X))

    axs[0,1].set_title('Reference')
    axs[0,1].imshow(y.reshape(xl, xs), cmap='gray')

    axs[0,2].set_title('Model Prediciton')
    patches = [mpatches.Patch(color=[0,1,0], label='TP'),
               mpatches.Patch(color=[1,0,0], label='FP'),
               mpatches.Patch(color=[1,.5,0], label='FN'),
               mpatches.Patch(color=[0,0,1], label='TN')]
    axs[0,2].legend(loc='upper right',
                    handles=patches,
                    ncol=2,
                    bbox_to_anchor=(1, -0.15)) # moves the legend outside
    axs[0,2].imshow(visualization)

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
    axs[1,0].set_ylabel('true label')

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
    axs[1,1].set_ylabel('true label')
    axs[1,1].margins(x=10)


    # axs[1,2].set_title('Feature Importance')

    # axs[1,2].set_xlabel('Band')
    # axs[1,2].bar(range(X_train.shape[1]),
    #                 importances[indices],
    #                 align='center')
    # axs[1,2].set_xticks(range(X_train.shape[1]))
    # axs[1,2].set_xticklabels(x for _,x in enumerate(feat_labels[indices]))
    # axs[1,2].set_xlim([-1, X_train.shape[1]])
    # axs[1,2].set_ylim([0, .15])

    plt.tight_layout()
    plt.savefig('outs/%s.png' % "small_data_gpu")
    plt.show()


def build_vis(prediction, y, shape):

    visualization = np.zeros((len(y), 3))
    idx = 0
    trps = 0
    try:
        for pixel in zip(prediction, y):

            if int(pixel[0]) and pixel[1]:
                # True Positive
                trps += 1
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
        print("UHOH")

    print(trps)
    return visualization.reshape(shape)


def encode_one_hot(target, xs, xl, array=False):

    if array:
        result = np.zeros((xs*xl, len(target)))
    else:
        result = list()

    for idx, key in enumerate(target.keys()):

        ones = np.ones((xl * xs))
        s,l,b,tmp = read_binary("data/elhill/data_bcgw/%s" % target[key])

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

        # How did the caller ask for the data
        if array:
            result[:,idx] = arr
        else:
            result.append((key, arr))

    return result


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
