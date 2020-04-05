from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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


    test_size = 0.3
    sklearn_rf_params = {
        'bootstrap': True,
        'ccp_alpha': 0.0,
        'class_weight': None,
        'criterion': 'gini',
        'max_depth': 45,
        'max_features': 'auto',
        'max_leaf_nodes': None,
        'max_samples': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'min_weight_fraction_leaf': 0.0,
        'n_estimators': 100,
        'n_jobs': -1,
        'oob_score': False,
        'random_state': 2,
        'verbose': 1,
        'warm_start': False}

    ## Load Data
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

    xs, xl, xb, X = read_binary(f'{raw_data_root}S2A.bin', to_string=False)

    X = X.reshape(xl * xs, xb)
    # Loop through all the target images
    for _, target_to_train in enumerate(target.keys()):

        ys,yl,yb, y = read_binary(f'{reference_data_root}%s' % target[target_to_train], to_string=False)

        assert xs == ys
        assert xl == yl


        y = binary_encode(target_to_train, y)
        y = y.reshape(int(yl)*int(ys))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        forest = RandomForestClassifier(**sklearn_rf_params)

        start_fit = time.time()
        forest.fit(X_train, y_train)
        end_fit = time.time()

        start_predict = time.time()
        pred = forest.predict(X)
        end_predict = time.time()

        fit_time = round(end_fit - start_fit,2)
        predict_time = round(end_predict - start_predict,2)

        # figure out the average depth of all the trees
        estimator_depths = [estimator.get_depth() for estimator in forest.estimators_]
        mean_depth = np.mean(estimator_depths)
        max_depth = mean_depth ##NOT ACTUALLY MAX DEPTH

        confmatTest = confusion_matrix(y_true=y_test, y_pred=forest.predict(X_test))
        confmatTrain = confusion_matrix(y_true=y_train, y_pred=forest.predict(X_train))

        train_score = forest.score(X_train, y_train)
        test_score = forest.score(X_test, y_test)

        visualization = build_vis(pred,y, (int(yl), int(ys), 3))



        fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
        # plt.subplots_adjust(right=.5, top=3)
        ex = Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
        fig.legend([ex,ex,ex,ex,ex,ex,ex,ex,ex,ex],
                ("Target: %s" % "Water",
                    "Test Acc.: %s" % round(test_score,3),
                    "Train Acc.: %s" % round(train_score,3),
                    "Test Size: %s" % test_size,
                    "Train: %ss" % fit_time,
                    "Predict: %ss" % predict_time,
                    "Estimators: %s" % sklearn_rf_params['n_estimators'],
                    "Max Features: %s" % sklearn_rf_params['max_features'],
                    "Max Depth: %s" % max_depth),

                loc='lower right',
                ncol=3)

        axs[0,0].set_title('Reference')
        axs[0,0].imshow(y.reshape(xl, xs), cmap='gray')

        axs[0,1].set_title('Prediction')
        axs[0,1].imshow(pred.reshape(xl, xs), cmap='gray')

        axs[0,2].set_title('Visual ConfMatrix')
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
        print('creating outs/RandForest in root')
        os.mkdir('outs/RandForest/')
    fn = f"sklearn_RF_{sklearn_rf_params['n_estimators']}trees_{max_depth}maxdepth"
    print(f'saving {fn} in outs/RandForest')
    plt.savefig('outs/RandForest/%s.png' % fn )
    plt.show()

def RGB(path):
    samples, lines, bands, X = read_binary(path)
    s = int(samples)
    l = int(lines)
    b = int(bands)
    data_r = X.reshape(b, s * l)
    rgb = np.zeros((l, s, 3))

    for i in range(0, 3):
        rgb[:, :, i] = data_r[3 - i, :].reshape((l, s))
    for i in range(0,3):
        rgb[:, :, i] = rescale(rgb[:, :, i])
    del X
    return rgb


def build_vis(prediction, y, shape):

    visualization = np.zeros((len(y), 3))
    for idx, pixel in enumerate(zip(prediction, y)):

        # compare the prediciton to the original
        if pixel[0] and pixel[1]:
            # True Positive
            visualization[idx,] = [0,1,0]

        elif pixel[0] and not pixel[1]:
            # False Positive
            visualization[idx,] = [1,0,0]

        elif not pixel[0] and pixel[1]:
            # False Negative
            visualization[idx,] = [1,.5,0]

        elif not pixel[0] and not pixel[1]:
            # True Negative
            visualization[idx, ] = [0,0,1]
            # visualization[idx, ] = rgb

        else:
            raise Exception("There was a problem predicting the pixel", idx)

    return visualization.reshape(shape)


def binary_encode(key, y):

    vals = np.sort(np.unique(y))
    ones = np.ones(y.shape)
    # create an array populate with the false value
    t = ones * vals[len(vals) - 1]
    if key == 'water':
        arr = np.not_equal(y,t)
    else:
        arr = np.logical_and(y,t)

    return arr


if __name__ == "__main__":

   main()