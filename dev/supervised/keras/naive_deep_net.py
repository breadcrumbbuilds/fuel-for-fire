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
import cv2
# TO VIEW MODEL PERFORMANCE IN BROWSER RUN
#
# FROM ROOT DIR
# globals
root_path = "data/zoom/"
reference_data_root = f"{root_path}data_bcgw/"
raw_data_root = f"{root_path}data_img/"

def create_model(input_dim, output_dim, test=False):
    if test:
        return tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_dim),
    tf.keras.layers.Dense(8, activation='relu'),#, kernel_initializer=regularizers.l2(0.001)),
    tf.keras.layers.Dense(output_dim, activation='softmax')
  ])
    else:
        return tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_dim),
    tf.keras.layers.Dense(8, activation='relu', kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, activation='relu', kernel_initializer="he_normal"),#, kernel_initializer=regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(2048, activation='relu', kernel_initializer="he_normal"),#, kernel_initializer=regularizers.l2(0.001)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(4096, activation='relu', kernel_initializer="he_normal"),#, kernel_initializer=regularizers.l2(0.001)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(8192, activation='relu', kernel_initializer="he_normal"),#, kernel_initializer=regularizers.l2(0.001)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(16384, activation='relu'),#, kernel_regularizer=regularizers.l2(0.001)),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(output_dim, activation='softmax')
  ])


def main():

    test = True

    print("Numpy Version: %s" % np.__version__)
    print("Keras Version: %s" % keras.__version__)

    """--------------------------------------------------------------
    * Configuration
    """
    ## Config
    test_size = .20

    if test:
        epochs = 1
        batch_size = 8192
    else:
        epochs = 1000
        batch_size = 512

    learning_rate = 0.001
    n_folds = 10
    METRICS = [
      # keras.metrics.CategoricalAccuracy(name='cat_acc'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.SensitivityAtSpecificity(.95)
]

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

    """----------------------------------------------------------------------------------------------------------------------------
    * Data Setup and Preprocessing
    """
    xs, xl, xb, X = read_binary(f'{raw_data_root}S2A.bin', to_string=False)
    X = X.reshape(xl * xs, xb)
    X = StandardScaler().fit_transform(X)  # standardize unit variance and 0 mean
    onehot = encode_one_hot(target, xs, xl, array=True)

    print("X shape", X.shape)
    print("X dtype", X.dtype)
    print("onehot shape", onehot.shape)
    print("onehot dtype",onehot.dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, onehot, test_size=test_size, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=0) # seperate val data

    print("Oversampling targets...")
    X_train, y_train = oversample(X_train, y_train, n_classes=len(target) + 1, extra_samples=50000)

    print(np.histogram(onehot, bins=len(target) + 1))

    y_train = keras.utils.to_categorical(y_train, num_classes=len(target) + 1)
    y_test = keras.utils.to_categorical(y_test, num_classes=len(target) + 1)
    y_val = keras.utils.to_categorical(y_val, num_classes=len(target) + 1)


    n_features = X_train.shape[1]
    n_classes = len(target) + 1
    rand_seed = 123 # reproducability
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)


    """----------------------------------------------------------------------------------------------------------------------------
    * Model
    """
    model = create_model(X_train.shape[1],y_train.shape[1], test=test)

    optimizer = keras.optimizers.Adam(
        lr=learning_rate, beta_1=0.9, beta_2=0.99
    )

    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=METRICS)



    model.build(input_shape=X_train.shape)
    print(model.summary())

    """----------------------------------------------------------------------------------------------------------------------------
    * Callbacks
    """
    # to visualize training in the browser'
    root_logdir = os.path.join(os.curdir, "logs")
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    filepath= os.path.join(os.curdir, "outs/models/NaiveDeepNet/weights-improvement-{epoch:02d}.hdf5")
    modelsave_cb = keras.callbacks.ModelCheckpoint(filepath, monitor='cat_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
    callbacks = [tensorboard_cb]


    """----------------------------------------------------------------------------------------------------------------------------
    * Training
    """
    start_fit = time.time()

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        class_weight=class_weight,
                        verbose=1,
                        validation_split=0.0,
                        validation_data=(X_val, y_val),
                        shuffle=True,
                        use_multiprocessing=True,
                        workers=-1,
                        callbacks=callbacks)

    end_fit = time.time()


    print(history.history)

    # read in the larger image
    # predict over the image

    """----------------------------------------------------------------------------------------------------------------------------
    * Evaluation
    """
    test_pred = model.predict(X_test)


    start_predict = time.time()
    test_pred = model.predict(X_test)
    end_predict = time.time()

    fit_time = round(end_fit - start_fit, 2)
    predict_time = round(end_predict - start_predict, 2)

    pred = model.predict(X)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)


    pred_confidence = np.amax(pred, axis=1)
    pred_class = np.argmax(pred, axis=1)



    # Hear we can build something to visualize the confidence
    # in reference to the class
    confmatTest = confusion_matrix(
        y_true=np.argmax(y_test, axis=1), y_pred=model.predict_classes(X_test))
    confmatTrain = confusion_matrix(
        y_true=np.argmax(y_train, axis=1), y_pred=model.predict_classes(X_train))


    visualization = build_vis(pred_class, onehot, (int(xl), int(xs), 3))

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
    axs[0, 1].imshow(pred_class.reshape(xl, xs), cmap='gray')

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


    axs[1,0].matshow(confmatTest, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(confmatTest.shape[0]):
        for j in range(confmatTest.shape[1]):
            axs[1,0].text(x=j, y=i,
                    s=round(confmatTest[i,j],3))
    axs[1,0].set_xticklabels([0,
                            "conifer"
                            "ccut",
                            "water",
                            "broadleaf",
                            "shrub",
                            "mixed",
                            "herb",
                            "exposed",
                            "river",
                              ])
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


def checkit(passed):
    print()
    print("CHECKIT")
    print(type(passed))
    print(passed)
    print(passed.shape)
    print()

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


def create_sub_images(X, cols, rows, bands):
    # TODO: Be nice to automate this.. need some type of LCD function ...
    sub_cols = cols//2 # not sure how to automate this yet but I know that these dims will create 10 sub images
    sub_rows = rows//5
    # shape of the sub images [sub_cols, sub_rows, bands]
    print("New subimage shape (%s, %s, %s)" % (sub_cols, sub_rows, bands))

    # container for the sub images
    sub_images = np.zeros((10, sub_cols, sub_rows, bands))

    # this will grab a sub set of the original image beginning with the top left corner, then the right top corner
    # and iteratively move down the image from left to right

    """
    Original image         subimages
    --------                --------
    |      |                [  ][  ]
    |      |                [  ][  ]
    |      |                [  ][  ]
    |      |                [  ][  ]
    |      |                [  ][  ]
    --------                --------
    """
    index = 0 # to index the container above for storing each sub image
    for row in range(5): # represents the 5 'rows' of this image
        for col in range(2): # represents the left and right side of the image split down the middle
            checkit(X)
            sub_images[index, :,:,:] = X[sub_cols * col : sub_cols * (col + 1), sub_rows * row : sub_rows * (row + 1), :]
            index += 1

    print("images, width, height, features", sub_images.shape)
    return sub_images


def oversample(X_train, y_train, n_classes=10, extra_samples=50000):

    tmp = np.zeros((X_train.shape[0], X_train.shape[1] + 1))

    tmp[:,:X_train.shape[1]] = X_train
    tmp[:,X_train.shape[1]] = y_train

    # Let's oversample each class so we don't have class imbalance
    vals, counts = np.unique(tmp[:,X_train.shape[1]], return_counts=True)
    maxval = np.amax(counts) + extra_samples

    ### WARNING, your validation data has leakage,
    for idx in range(n_classes):
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
    X_train = tmp[:,:X_train.shape[1]]
    y_train = tmp[:,X_train.shape[1]]

    return X_train, y_train


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



if __name__ == "__main__":

   main()
