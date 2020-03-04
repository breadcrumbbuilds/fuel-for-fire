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
"""
    Paralellized program to create RandomForests for all 9 classes and visualize the results

"""
def run(key):
    # colours = [color.hsv_to_rgb(vector( i / (next_label - 1), 1., 1.)) for i in range(0, next_label)]

    n_estimators = 50
    n_features = 2
    test_size =.25
    file = 'RandForest/%s_%s_%s.png' % (key, n_estimators, n_features)
    data = Helper.init_data()
    X = data.S2.Data()
    rgb = data.S2.rgb
    data_name = "Sentinel2"
    y = data.Target[key].Binary
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    feat_labels = [str(x) for x in range(X.shape[1])]
    feat_labels = np.asarray(feat_labels)
    # n jobs uses all the available cores - turn to 1 so can parallelize across all nine classes
    forest = RandomForestClassifier(n_estimators=n_estimators,
                                    random_state=2,
                                    max_features=n_features,
                                    n_jobs=1)
    start_fit = time.time()
    forest.fit(X_train, y_train)
    end_fit = time.time()
    fit_time = round(end_fit - start_fit,2)

    start_predict = time.time()
    predictions = forest.predict(X)
    end_predict = time.time()
    predict_time = round(end_predict - start_predict,2)

    confmatTest = confusion_matrix(y_true=y_test, y_pred=forest.predict(X_test))/len(y_test)
    confmatTrain = confusion_matrix(y_true=y_train, y_pred=forest.predict(X_train))/len(y_train)

    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    train_score = forest.score(X_train, y_train)
    test_score = forest.score(X_test, y_test)

    visualization = build_vis(rgb,predictions,y, (data.S2.lines, data.S2.samples, 3))


    fig, axs = plt.subplots(2, 3, figsize=(9, 6), sharey=False)
    # plt.subplots_adjust(right=.5, top=3)
    ex = Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
    fig.legend([ex,ex,ex,ex,ex,ex,ex,ex],
               ("Target: %s" % key.upper(),
                "Test Acc.: %s" % round(test_score,3),
                "Train Acc.: %s" % round(train_score,3),
                "Test Size: %s" % test_size,
                "Train: %ss" % fit_time,
                "Predict: %ss" % predict_time,
                "Estimators: %s" % n_estimators,
                "Max Features: %s" % n_features),
               loc='center left',
               ncol=4)
    axs[0,0].set_title('Sentinel2')
    axs[0,0].imshow(data.S2.rgb)

    axs[0,1].set_title('Reference')
    axs[0,1].imshow(y.reshape(data.Target[key].lines, data.Target[key].samples), cmap='gray')

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
    axs[1,0].set_xticklabels([0, 'True', 'False'])
    axs[1,0].xaxis.set_ticks_position('bottom')
    axs[1,0].set_yticklabels([0, 'True', 'False'])
    axs[1,0].set_xlabel('predicted label')
    axs[1,0].set_ylabel('true label')

    axs[1,1].set_title('Train Data Confusion Matrix')

    axs[1,1].matshow(confmatTrain, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(confmatTrain.shape[0]):
        for j in range(confmatTrain.shape[1]):
            axs[1,1].text(x=j, y=i,
                    s=round(confmatTrain[i,j],3))
    axs[1,1].set_xticklabels([0, 'True', 'False'])
    axs[1,1].xaxis.set_ticks_position('bottom')
    axs[1,1].set_yticklabels([0, 'True', 'False'])
    axs[1,1].set_xlabel('predicted label')
    axs[1,1].set_ylabel('true label')
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
    plt.show()


def build_vis(rgb,prediction, y, shape):
    rgb = rgb.reshape(164410, 3) # HACKY
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
            visualization[idx, ] = rgb[idx, ]
            # visualization[idx, ] = rgb[0,0,1]


        else:
            raise Exception("There was a problem predicting the pixel", idx)

    return visualization.reshape(shape)
def main():
    data = Helper.init_data()
    cpus = multiprocessing.cpu_count()
    print('Found %s cpus' % cpus)
    pool = multiprocessing.Pool(processes=cpus)
    pool.map(run, data.Target.keys())
    pool.close()


if __name__ == "__main__":

   main()