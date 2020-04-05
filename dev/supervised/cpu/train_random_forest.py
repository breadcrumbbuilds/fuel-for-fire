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

def main():

    test_size = 0.3
    n_estimators = 100
    max_depth = None #expanded until leaves are pure
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
        # # "vri" : "vri_s3_objid2.tif_proj.bin",
        "water" : "WATER.bin",
    }
 # output4_selectS2.bin
 # S2A.bin_4x.bin_sub.bin
 # read_binary('data/data_img/output4_selectS2.bin')
    training_image = 'S2A.bin'
    path = 'data/zoom'
    training_image_path = f'{path}/data_img/{training_image}'
    xs, xl, xb, X = read_binary(training_image_path)
    xs = int(xs)
    xl = int(xl)
    xb = int(xb)

    for _, target_to_train in enumerate(target.keys()):
        file = 'RandForest/RF_%s_%s.png' % (target_to_train, n_estimators)

        ys,yl,yb, y = read_binary(f'{path}/data_bcgw/%s' % target[target_to_train])

        assert int(xs) == int(ys)
        assert int(xl) == int(yl)


        X = X.reshape(xl*xs, xb)

        y = binary_encode(target_to_train, y)
        y = y.reshape(int(yl)*int(ys))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train_centered = (X_train - mean) / np.std(X_train, axis=0)
        X_test_centered = (X_test - mean) / std
        X_centered = (X - mean) / std

        feat_labels = [str(x) for x in range(X.shape[1])]
        feat_labels = np.asarray(feat_labels)
        forest = RandomForestClassifier(n_estimators=n_estimators,
                                        random_state=2,
                                        max_depth=max_depth,
                                        n_jobs=-1,
                                        verbose=1)

        start_fit = time.time()
        forest.fit(X_train_centered, y_train)
        print(forest.get_params())
        end_fit = time.time()
        fit_time = round(end_fit - start_fit,2)

        estimator_depths = [estimator.get_depth() for estimator in forest.estimators_]
        mean_depth = np.mean(estimator_depths)
        max_depth = mean_depth

        start_predict = time.time()
        pred = forest.predict(X_centered)
        end_predict = time.time()
        predict_time = round(end_predict - start_predict,2)

        confmatTest = confusion_matrix(y_true=y_test, y_pred=forest.predict(X_test_centered))
        confmatTrain = confusion_matrix(y_true=y_train, y_pred=forest.predict(X_train_centered))

        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]

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
                    "Estimators: %s" % n_estimators,
                    "Max Features: %s" % 'default',
                    "Max Depth: %s" % max_depth),

                loc='lower right',
                ncol=3)

        # axs[0,0].set_title('Sentinel2')
        # axs[0,0].imshow(visRGB(xs,xl,xb,X))
        axs[0,0].set_title('Reference')
        axs[0,0].imshow(y.reshape(xl, xs), cmap='gray')

        axs[0,1].set_title('Prediction')
        axs[0,1].imshow(pred.reshape(xl, xs), cmap='gray')
        # axs[0,0].set_title('Sentinel2')
        # axs[0,0].imshow(visRGB(xs,xl,xb,X))

        # axs[0,1].set_title('Reference')
        # axs[0,1].imshow(y.reshape(xl, xs), cmap='gray')

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

        axs[1,2].set_title('Feature Importance')

        axs[1,2].set_xlabel('Band')
        axs[1,2].bar(range(X_train.shape[1]),
                        importances[indices],
                        align='center')
        axs[1,2].set_xticks(range(X_train.shape[1]))
        axs[1,2].set_xticklabels(x for _,x in enumerate(feat_labels[indices]))
        axs[1,2].set_xlim([-1, X_train.shape[1]])
        axs[1,2].set_ylim([0, .15])

        plt.tight_layout()
        plt.savefig('outs/%s.png' % file)


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
   #rgb = rgb.reshape(164410, 3) # HACKY
    visualization = np.zeros((len(y), 3))
    for idx, pixel in enumerate(zip(prediction, y)):
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