import tensorflow.keras as keras
import tensorflow as tf
import time
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Utils.Helper import rescale, create_batch_generator
from Utils.Misc import read_binary
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras import utils
# TO VIEW MODEL PERFORMANCE IN BROWSER RUN
#
# FROM ROOT DIR
# globals
root_path = "data/zoom/"
reference_data_root = f"{root_path}data_bcgw/"
raw_data_root = f"{root_path}data_img/"

def main():


    print("Numpy Version: %s" % np.__version__)
    print("Keras Version: %s" % keras.__version__)


    ## Config
    test_size = .3
    learning_rate = 0.001
    batch_size = 4096
    epochs = 100

    METRICS = [
      keras.metrics.BinaryAccuracy(name='b_acc'),
      keras.metrics.CategoricalAccuracy(name='cat_acc'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]


    ## Load Data
    target = {
        "broadleaf" : "BROADLEAF.bin",
        "ccut" : "CCUTBL.bin",
        "conifer" : "CONIFER.bin",
        "exposed" : "EXPOSED.bin",
        "herb" : "HERB.bin",
        "mixed" : "MIXED.bin",
        "river" : "Rivers.bin",
        # # "road" : "ROADS.bin",
        "shrub" : "SHRUB.bin",
        # "vri" : "vri_s3_objid2.tif_proj.bin",
        "water": "WATER.bin",
    }
    xs, xl, xb, X = read_binary(f'{raw_data_root}S2A.bin', to_string=False)
    X = X.reshape(xl * xs, xb)
    X = StandardScaler().fit_transform(X)  # standardize unit variance and 0 mean

    onehot = encode_one_hot(target, xs, xl, array=True)
    print(np.histogram(onehot, bins=len(target) + 1))

    # ## Preprocess
    X_train, X_test, y_train, y_test = train_test_split(X, onehot, test_size=test_size)
    tmp = np.zeros((X_train.shape[0], 13))
    tmp[:,:X_train.shape[1]] = X_train
    tmp[:,X_train.shape[1]] = y_train

    # Let's oversample each class so we don't have class imbalance
    vals, counts = np.unique(tmp[:,X_train.shape[1]], return_counts=True)
    maxval = np.amax(counts) + 50000

    ### WARNING, your validation data has leakage,
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
    # let's duplicate the positive X_train vals
    # for idx, pixel in enumerate(zip(X_train, y_train)):

    y_train = keras.utils.to_categorical(y_train, num_classes=len(target) + 1)
    print(y_train.shape)
    n_features = X_train.shape[1]
    n_classes = len(target) + 1
    rand_seed = 123 # reproducability

    np.random.seed(rand_seed)

    tf.random.set_seed(rand_seed)
    print(X_train.shape)
    print(y_train.shape)

    model = create_model(X_train.shape[1],y_train.shape[1])

    optimizer = keras.optimizers.Adam(
        lr=learning_rate, beta_1=0.9, beta_2=0.99
    )

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                metrics=METRICS)



    # to visualize training in the browser'
    root_logdir = os.path.join(os.curdir, "logs")
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    filepath= os.path.join(os.curdir, "outs/models/NaiveDeepNet/weights-improvement-{epoch:02d}.hdf5")
    modelsave_cb = keras.callbacks.ModelCheckpoint(filepath, monitor='cat_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
    callbacks = [tensorboard_cb]

    start_fit = time.time()

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        class_weight=class_weight,
                        verbose=1,
                        validation_split=0.1,
                        shuffle=True,
                        use_multiprocessing=True,
                        workers=-1,
                        callbacks=callbacks)

    end_fit = time.time()


    print(history.history)


    test_pred = model.predict(X_test)
    score = model.evaluate(X_test, y_test)
    print(score)

    start_predict = time.time()
    test_pred = model.predict(X_test)
    end_predict = time.time()

    fit_time = round(end_fit - start_fit, 2)
    predict_time = round(end_predict - start_predict, 2)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    pred = keras.metrics.binary_accuracy(onehot, X, threshold=0.5)
    pred = pred.numpy()
    val, counts = np.unique(pred, return_counts=True)
    for idx, ret in enumerate(zip(val, counts)):
        print(ret[0], ret[1])
    print()
    # confmatTest = confusion_matrix(
    #     y_true=y_test, y_pred=model.predict(X_test))
    # confmatTrain = confusion_matrix(
    #     y_true=y_train, y_pred=model.predict(X_train))

    # train_score = model.score(X_train, y_train)
    # test_score = model.score(X_test, y_test)

    visualization = build_vis(pred, onehot, (int(xl), int(xs), 3))

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=False)

    ex = Rectangle((0, 0), 0, 0, fc="w", fill=False,
                    edgecolor='none', linewidth=0)
    fig.legend([ex, ex, ex, ex, ex, ex, ex, ex, ex, ex, ex],
                ("Target: %s" % "Water",
                # "Test Acc.: %s" % round(test_score, 3),
                # "Train Acc.: %s" % round(train_score, 3),
                "Test Size: %s" % test_size,
                "Train: %ss" % fit_time,
                "Predict: %ss" % predict_time),

                loc='lower right',
                ncol=3)

    axs[0, 0].set_title('Reference')
    axs[0, 0].imshow(onehot.reshape(xl, xs), cmap='gray')

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

    # axs[1, 0].set_title('Test Data Confusion Matrix')

    # axs[1, 0].matshow(confmatTest, cmap=plt.cm.Blues, alpha=0.5)
    # for i in range(confmatTest.shape[0]):
    #     for j in range(confmatTest.shape[1]):
    #         axs[1, 0].text(x=j, y=i,
    #                         s=round(confmatTest[i, j], 3))
    # axs[1, 0].set_xticklabels([0, 'False', 'True'])
    # axs[1, 0].xaxis.set_ticks_position('bottom')
    # axs[1, 0].set_yticklabels([0, 'False', 'True'])
    # axs[1, 0].set_xlabel('predicted label')
    # axs[1, 0].set_ylabel('reference label')

    # axs[1, 1].set_title('Train Data Confusion Matrix')

    # axs[1, 1].matshow(confmatTrain, cmap=plt.cm.Blues, alpha=0.5)

    # for i in range(confmatTrain.shape[0]):
    #     for j in range(confmatTrain.shape[1]):
    #         axs[1, 1].text(x=j, y=i,
    #                         s=round(confmatTrain[i, j], 3))
    # axs[1, 1].set_xticklabels([0, 'False', 'True'])
    # axs[1, 1].xaxis.set_ticks_position('bottom')
    # axs[1, 1].set_yticklabels([0, 'False', 'True'])
    # axs[1, 1].set_xlabel('predicted label')
    # axs[1, 1].set_ylabel('reference label')
    # axs[1, 1].margins(x=10)

    plt.tight_layout()

    fn = f'models/NaiveDeepNet/result_'

    if not os.path.exists('outs'):
        print('creating outs directory in root')
        os.mkdir('outs')
    if not os.path.exists('outs/NaiveDeepNet/'):
        print('creating outs/NaiveDeepNet in root')
        os.mkdir('outs/NaiveDeepNet/')

    print(f'saving {fn.split("/")[2]} in outs/NaiveDeepNet')
    plt.savefig('outs/NaiveDeepNet/%s.png' % fn.split('/')[2])
    plt.show()


def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run__%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def one_hot_sanity_check(target, xs, xl):
    oh = encode_one_hot(target, xs, xl)
    for idx, val in enumerate(oh):
        print(val[0])
        plt.title(val[0])
        plt.imshow(val[1].reshape(xl,xs), cmap='gray')
        plt.show()


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

def create_model(input_dim, output_dim):
  return tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_dim),
    tf.keras.layers.Dense(512, activation='relu'),#, kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1024, activation='relu'),#, kernel_regularizer=regularizers.l2(0.001)),
    # tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(2048, activation='relu'),#, kernel_regularizer=regularizers.l2(0.001)),
    # tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(4096, activation='relu'),#, kernel_regularizer=regularizers.l2(0.001)),
    # tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(8192, activation='relu'),#, kernel_regularizer=regularizers.l2(0.001)),
    # tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(16384, activation='relu'),#, kernel_regularizer=regularizers.l2(0.001)),
    # tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(output_dim, activation='softmax')
  ])


if __name__ == "__main__":

   main()
