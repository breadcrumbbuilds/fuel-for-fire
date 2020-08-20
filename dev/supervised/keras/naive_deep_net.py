from keras import utils
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
import time
import sys
import os.path

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import math
import sys
from joblib import dump, load
sys.path.append(os.curdir) # so python can find Utils
from Utils.Misc import *
from Utils.Helper import rescale, create_batch_generator

# TO VIEW MODEL PERFORMANCE IN BROWSER RUN
#
# FROM ROOT DIR
# globals
root_path = "data/zoom/"
reference_data_root = f"{root_path}data_bcgw/"
raw_data_root = f"{root_path}data_img/"

def create_model(input_dim=11, output_dim=1, test=False):
    if test:
        return tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_dim),
    tf.keras.layers.Dense(8, activation='relu'),#, kernel_initializer=regularizers.l2(0.001)),
    tf.keras.layers.Dense(output_dim, activation='softmax')
  ])
    else:
        return tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_dim),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),

    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(512, activation='relu'),#, kernel_initializer=regularizers.l2(0.001)),
    # # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(1024, activation='relu'),#, kernel_initializer=regularizers.l2(0.001)),
    # # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(2048, activation='relu'),
    # # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(4096, activation='relu'),
    # # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(8192, activation='relu'),
    # # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(16384, activation='relu'),
    # # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(output_dim, activation='sigmoid')
  ])


def main():

    test = False

    print("Numpy Version: %s" % np.__version__)
    print("Keras Version: %s" % keras.__version__)

    """----------------------------------------------------------------------------------------------------------------------------
    * Configuration
    """
    # Config
    test_size = .20
    epochs = 250
    batch_size = 1024
    lr = 0.001

    targets = {
        # "conifer": "CONIFER.bin",
        "water": "WATER.bin"
    }

    """----------------------------------------------------------------------------------------------------------------------------
    * Data Setup and Preprocessing
    """
    """ Run From Project Root
    """
    root = "data/full/"
    train_root_path = f"{root}/prepared/train/"
    reference_data_root = f"{root}data_bcgw/"
    raw_data_root = f"{root}data_img/"
    data_output_directory, results_output_directory, models_output_directory = get_working_directories("KFold/MLP")

    X = np.load(f'{train_root_path}/full-img.npy').reshape(4835, 3402, 11)

    sub_img_shape = (4835//5,3402//2)
    fold_length = X.shape[0] * X.shape[1] // 10
    X_train_subbed, X_val_subbed = split_train_val(X, sub_img_shape) # split the orig data into 5 sub images
    save_rgb(X_train_subbed, sub_img_shape, data_output_directory, 'training') # save the rgb in the output dir for later use
    save_rgb(X_val_subbed, sub_img_shape, data_output_directory, 'validation') # save the rgb in the output dir for later use
    del X
    """Prepare maps for training"""
    for target in targets:
        cols, rows, bands, y = read_binary(f'{reference_data_root}{targets[target]}', to_string=False)
        y = convert_y_to_binary(target, y, cols, rows).reshape(rows, cols)
        y_train_subbed, y_validation_subbed = split_train_val(y, sub_img_shape)
        del y
        # save the original maps to disk
        save_subimg_maps(y_train_subbed, sub_img_shape, data_output_directory, target, "training_map")
        save_subimg_maps(y_validation_subbed, sub_img_shape, data_output_directory, target, "validation_map")

        train_kfold_model(target, X_train_subbed, y_train_subbed, X_val_subbed, y_validation_subbed, epochs, batch_size, lr, sub_img_shape, data_output_directory, models_output_directory, "initial", initial_model=True)
        proba_predictions = None
        predictions_list = list()
        full_pred = None
        for image_idx in range(5):
            this_prediction = load_np(os.path.join(data_output_directory, f"val_{target}_initial_proba-prediction_{image_idx}.npy")).ravel()
            if full_pred is None:
                full_pred = this_prediction
            else:
                full_pred = np.concatenate((full_pred, this_prediction))

        create_seeded_percentile_models(target, X_val_subbed, full_pred, X_train_subbed, y_train_subbed, n_est, fold_length, sub_img_shape, data_output_directory, models_output_directory, percentile=85)


"""----------------------------------------------------------------------------------------------------------------------------
* Utility Functions
"""

def create_seeded_percentile_models(target, X_subbed_list, prediction, X_val_list, y_val_list, n_est, fold_length, sub_img_shape, data_output_directory, models_output_directory, percentile=50):
    """ Creates seeded models based on the percentile passed """
    y_subbed_training = create_percentile_map(prediction, percentile, fold_length, data_output_directory)
    save_subimg_maps(y_subbed_training, sub_img_shape, data_output_directory, target, f"{percentile}-percentile_map")
    print(f"Seeded Percentile Model: {percentile}")
    seeded_models = train_kfold_model(target, X_subbed_list, y_subbed_training, X_val_list, y_val_list, n_est, sub_img_shape, data_output_directory, models_output_directory, f"seeded-{percentile}percentile")
    return seeded_models


def create_percentile_map(prediction, percentile, fold_length, data_output_directory, seperate=False):
    """ Returns the boolean intersection of the percentile on each image in predictions list
    """
    y_subbed_list = list()
    probability_map = prediction > np.percentile(prediction, percentile)
    for idx in range(5):
        x_start= idx * fold_length
        x_end = (idx+1) * fold_length
        y_subbed_list.append(probability_map[x_start : x_end])
    return y_subbed_list


def train_kfold_model(target, X_training_list, y_training_list, X_val_list, y_val_list, epochs, batch_size, lr, sub_img_shape, data_output_directory, models_output_directory, filename, initial_model=False):
    """ Creates 5 fold models, returns a list of models, one per fold """
    print("Starting Training")

    METRICS = [
        # keras.metrics.CategoricalAccuracy(name='cat_acc'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.Accuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.SensitivityAtSpecificity(.95)
    ]
    """----------------------------------------------------------------------------------------------------------------------------
    * Model
    """
    model = create_model(11,1)

    optimizer = keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.9)
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=METRICS)
    """----------------------------------------------------------------------------------------------------------------------------
    * Callbacks
    """
    # to visualize training in the browser'
    root_logdir = os.path.join(os.curdir, "logs")
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    filepath= os.path.join(os.curdir, "outs/models/NaiveDeepNet/weights-improvement-{epoch:02d}.hdf5")
    modelsave_cb = keras.callbacks.ModelCheckpoint(filepath, monitor='cat_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=10,
                                            verbose=0,
                                            mode='auto',
                                            baseline=None,
                                            restore_best_weights=True)
    callbacks = [tensorboard_cb]
    """----------------------------------------------------------------------------------------------------------------------------
    * Training
    """


    class_weight = {0: 1.,
                1: 2.}


    X_train_all = None
    y_train_all = None
    for train_idx in range(5):
        X_train = X_training_list[train_idx]
        y_train = y_training_list[train_idx]
        random_indexed = np.random.shuffle(y_train)
        X_train = np.squeeze(X_train[random_indexed])
        y_train = np.squeeze(y_train[random_indexed])
        X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])
        if len(y_train.shape) > 1:
            y_train = y_train.ravel()
        del random_indexed
        if X_train_all is None:
            X_train_all = X_train
            y_train_all = y_train
        else:
            X_train_all = np.concatenate((X_train_all, X_train))
            y_train_all = np.concatenate((y_train_all, y_train))
    X_train_all = (X_train_all - X_train_all.min(0)) / X_train_all.ptp(0)
    print(X_train_all)
    y_train_all = y_train_all.astype(int)
    print(f"X_train shape: {X_train_all.shape}")
    print(f"y_train shape: {y_train_all.shape}")
    vals, counts = np.unique(y_train_all, return_counts=True)
    print(f"y_train distribution: \n{vals}\n{counts}")
    model.build(X_train_all.shape)
    print(model.summary())
    start_fit = time.time()
    history = model.fit(X_train_all, y_train_all,
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weight,
        verbose=1,
        validation_split=0.0,
        validation_data=(X_val_list[0].reshape(X_val_list[0].shape[0] * X_val_list[0].shape[1], X_val_list[0].shape[2]), y_val_list[0].reshape(X_val_list[0].shape[0] * X_val_list[0].shape[1])),
        shuffle=True,
        use_multiprocessing=True,
        workers=-1,
        callbacks=callbacks)
    end_fit = time.time()

    print(history.history)
    save_model(model, models_output_directory, f"{target}-{filename}")
    print("Validation Prediction")
    for idx_i, X_val in enumerate(X_val_list):
        X_val = X_val.reshape(X_val.shape[0] * X_val.shape[1], X_val.shape[2])
        class_prediction = model.predict(X_val)
        save_np(class_prediction, os.path.join(data_output_directory, f"val_{target}_{filename}_proba-prediction_{idx_i}"))
        # proba_prediction = clf.predict_proba(X_val)[:,1] # the true values probability
        # save_np(proba_prediction, os.path.join(data_output_directory, f"val_{target}_{filename}_proba-prediction_{idx_i}"))
    print("Finished Training\n")


def save_model(model, models_output_directory, filename):
    print("Saving Model")
    path = os.path.join(models_output_directory, f"{filename}.joblib")
    dump(model, path)
    print(f"+w {path}")


def split_train_val(data, shape):
    """ Splits X into 5 sub images of equal size and return the sub images in a list """
    train = list()
    val = list()
    for x in range(5):
            for y in range(2):
                x_start = x * shape[0]
                x_end = (x+1) * shape[0]
                y_start = y * shape[1]
                y_end = (y+1) * shape[1]
                if len(data.shape) > 2:
                    if y == 0:
                        train.append(data[ x_start:x_end , y_start : y_end, :])
                    else:
                        val.append(data[x_start:x_end, y_start:y_end, :])
                else:
                    if y == 0:
                        train.append(data[x_start:x_end , y_start : y_end])
                    else:
                        val.append(data[x_start:x_end, y_start:y_end])
    return train, val



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
        plt.imshow(val[1].reshape(xl, xs), cmap='gray')
        plt.show()


def save_subimg_maps(y_subbed_list, sub_img_shape, data_output_directory, target, filename):
    print("Saving sub image maps")
    for x, sub_img in enumerate(y_subbed_list):
        save_np(sub_img, os.path.join(data_output_directory, f"{target}-{filename}_{x}"))


def save_rgb(subimgs, sub_img_shape, output_directory, name):
    """ Saves each subimgs RGB interpretation to output_directory """
    print("Creating RGB sub images")
    sub_imgs = list()
    temp = np.zeros((sub_img_shape[0], sub_img_shape[1], 3))
    rgb_stretched = np.zeros((sub_img_shape[0], sub_img_shape[1], 3))
    full_img = None
    for x, data in enumerate(subimgs):
        rgb = np.zeros((sub_img_shape[0], sub_img_shape[1], 3))
        for i in range(0,3):
            rgb[:,:, i] = data[:,:, 4 - i]
        if full_img is None:
            full_img = rgb
        else:
            full_img = np.concatenate((full_img, rgb))
    for i in range(0,3):
        full_img[:,:,i] = rescale(full_img[:,:,i], two_percent=False)
    save_np(full_img, os.path.join(output_directory, f"rgb_{name}_image-twopercentstretch"))
    print()


if __name__ == "__main__":

    main()
