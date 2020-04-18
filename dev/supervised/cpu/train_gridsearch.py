from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import *
from Utils.Helper import rescale
# globals
root_path = "data/zoom/"
reference_data_root = f"{root_path}data_bcgw/"
raw_data_root = f"{root_path}data_img/"


def main():

    param_grid = [{
        'n_estimators': [50000],
        'max_features': [0.1, 0.3, 0.5],
        'max_depth': [1, 3, 6, 12],
        'verbose':[1],
        'n_jobs': [-1],
    }]

    classifiers = (
        RandomForestClassifier(),
    )
    clf_names =("RF")
    test_size = 0.3

    # Load Data
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


    for forest, name in zip(classifiers,clf_names):

        # Exhaustive search of the supplied paramters
        # refit trains the best estimator on all of the training data
        grid_search = GridSearchCV(forest, param_grid,
                                   scoring='accuracy',
                                   return_train_score=True,
                                   verbose=1,
                                   n_jobs=-1,
                                   refit=True,
                                   cv=2)


        y = encode_one_hot(target, xs, xl, array=True)

        print(f"y labels: {np.bincount(y.astype(int))}")
        y = y.reshape(int(xl)*int(xs))

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

        start_fit = time.time()

        grid_search.fit(X_train, y_train)


        end_fit = time.time()

        if not os.path.exists('models'):
            os.mkdir('models')
        if not os.path.exists('models/grid_search'):
            os.mkdir('models/grid_search')

        fn = f'models/grid_search/grid_search_result_{name}.pkl'
        joblib.dump(grid_search.best_estimator_, fn, compress = 1)

        cvresults = grid_search.cv_results_
        print(cvresults)


        # Need to verify that the best estimator will actually
        # be the estimator that was retrained on all of the training data
        start_predict = time.time()
        pred = grid_search.predict(X)
        end_predict = time.time()

        fit_time = round(end_fit - start_fit, 2)
        predict_time = round(end_predict - start_predict, 2)

        confmatTest = confusion_matrix(
            y_true=y_test, y_pred=grid_search.predict(X_test))
        confmatTrain = confusion_matrix(
            y_true=y_train, y_pred=grid_search.predict(X_train))

        train_score = grid_search.score(X_train, y_train)
        test_score = grid_search.score(X_test, y_test)

        visualization = build_vis(pred, y, (int(yl), int(ys), 3))

        fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=False)

        ex = Rectangle((0, 0), 0, 0, fc="w", fill=False,
                        edgecolor='none', linewidth=0)
        fig.legend([ex, ex, ex, ex, ex, ex, ex, ex, ex, ex, ex],
                    ("Target: %s" % "Water",
                    "Test Acc.: %s" % round(test_score, 3),
                    "Train Acc.: %s" % round(train_score, 3),
                    "Test Size: %s" % test_size,
                    "Train: %ss" % fit_time,
                    "Predict: %ss" % predict_time,
                    "Estimators: %s" % grid_search.best_estimator_.get_params()['n_estimators'],
                    "Max Features: %s" % grid_search.best_estimator_.get_params()['max_features'],
                    "Max Depth: %s" % grid_search.best_estimator_.get_params()['max_depth']),
                    #"SubSampling: %s" % grid_search.best_estimator_.get_params()['subsample']),

                    loc='lower right',
                    ncol=3)

        axs[0, 0].set_title('Reference')
        axs[0, 0].imshow(y.reshape(xl, xs), cmap='gray')

        axs[0, 1].set_title('Prediction')
        axs[0, 1].imshow(pred.reshape(xl, xs), cmap='gray')

        axs[0, 2].set_title('Visual ConfMatrix')
        patches = [mpatches.Patch(color=[0, 1, 0], label='TP'),
                    mpatches.Patch(color=[1, 0, 0], label='FP'),
                    mpatches.Patch(color=[1, .5, 0], label='FN'),
                    mpatches.Patch(color=[0, 0, 1], label='TN')]
        axs[0, 2].legend(loc='upper right',
                            handles=patches,
                            ncol=2,
                            bbox_to_anchor=(1, -0.15))  # moves the legend outside
        axs[0, 2].imshow(visualization)

        axs[1, 0].set_title('Test Data Confusion Matrix')

        axs[1, 0].matshow(confmatTest, cmap=plt.cm.Blues, alpha=0.5)
        for i in range(confmatTest.shape[0]):
            for j in range(confmatTest.shape[1]):
                axs[1, 0].text(x=j, y=i,
                                s=round(confmatTest[i, j], 3))
        axs[1, 0].set_xticklabels([0, 'False', 'True'])
        axs[1, 0].xaxis.set_ticks_position('bottom')
        axs[1, 0].set_yticklabels([0, 'False', 'True'])
        axs[1, 0].set_xlabel('predicted label')
        axs[1, 0].set_ylabel('reference label')

        axs[1, 1].set_title('Train Data Confusion Matrix')

        axs[1, 1].matshow(confmatTrain, cmap=plt.cm.Blues, alpha=0.5)

        for i in range(confmatTrain.shape[0]):
            for j in range(confmatTrain.shape[1]):
                axs[1, 1].text(x=j, y=i,
                                s=round(confmatTrain[i, j], 3))
        axs[1, 1].set_xticklabels([0, 'False', 'True'])
        axs[1, 1].xaxis.set_ticks_position('bottom')
        axs[1, 1].set_yticklabels([0, 'False', 'True'])
        axs[1, 1].set_xlabel('predicted label')
        axs[1, 1].set_ylabel('reference label')
        axs[1, 1].margins(x=10)

        plt.tight_layout()
    if not os.path.exists('outs'):
        print('creating outs directory in root')
        os.mkdir('outs')
    if not os.path.exists('outs/GridSearchForest/'):
        print('creating outs/GridSearchForest in root')
        os.mkdir('outs/GridSearchForest/')

    print(f'saving {fn.split("/")[2]} in outs/GridSearchForest')
    plt.savefig('outs/GridSearchForest/%s.png' % fn.split('/')[2])
    # plt.show()


def RGB(path):
    samples, lines, bands, X = read_binary(path)
    s = int(samples)
    l = int(lines)
    b = int(bands)
    data_r = X.reshape(b, s * l)
    rgb = np.zeros((l, s, 3))

    for i in range(0, 3):
        rgb[:, :, i] = data_r[3 - i, :].reshape((l, s))
    for i in range(0, 3):
        rgb[:, :, i] = rescale(rgb[:, :, i])
    del X
    return rgb


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

    return result


if __name__ == "__main__":

    main()
