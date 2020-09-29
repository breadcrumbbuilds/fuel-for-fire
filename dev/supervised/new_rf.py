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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


def main():
    """ Run From Project Root
    """
    root = "data/full/"
    train_root_path = f"{root}/prepared/train/"
    reference_data_root = f"{root}data_bcgw/"
    raw_data_root = f"{root}data_img/"


    for n_est in [100]:
        data_output_directory, results_output_directory, models_output_directory = get_working_directories(f"RandomForest/max-depth-16/{n_est}-trees/undersample.5/iterative")

        X = np.load(f'{train_root_path}/full-img.npy').reshape(4835, 3402, 11)

        sub_img_shape = (4835//5,3402//2)
        fold_length = X.shape[0] * X.shape[1] // 10
        X_train_subbed, X_val_subbed = split_train_val(X, sub_img_shape) # split the orig data into 5 sub images
        save_rgb(X_train_subbed, sub_img_shape, data_output_directory, 'training') # save the rgb in the output dir for later use
        save_rgb(X_val_subbed, sub_img_shape, data_output_directory, 'validation') # save the rgb in the output dir for later use
        del X
        targets = {
            "water": "WATER.bin"
        }
        """Prepare maps for training"""
        for target in targets:
            cols, rows, bands, y = read_binary(f'{reference_data_root}{targets[target]}', to_string=False)
            y = convert_y_to_binary(target, y, cols, rows).reshape(rows, cols)
            y_train_subbed, y_validation_subbed = split_train_val(y, sub_img_shape)
            del y
            # save the original maps to disk
            save_subimg_maps(y_train_subbed, sub_img_shape, data_output_directory, target, "training_map")
            save_subimg_maps(y_validation_subbed, sub_img_shape, data_output_directory, target, "validation_map")

            train_kfold_model(target, X_train_subbed, y_train_subbed, X_val_subbed, y_validation_subbed, n_est, sub_img_shape, data_output_directory, models_output_directory, "initial", initial_model=True, subsample=True)
            proba_predictions = None
            predictions_list = list()
            full_pred = None
            for image_idx in range(5):
                this_prediction = load_np(os.path.join(data_output_directory, f"val_{target}_initial_proba-prediction_{image_idx}.npy")).ravel()
                if full_pred is None:
                    full_pred = this_prediction
                else:
                    full_pred = np.concatenate((full_pred, this_prediction))


            # create_seeded_percentile_models(target, X_val_subbed, full_pred, X_train_subbed, y_train_subbed, n_est, fold_length, sub_img_shape, data_output_directory, models_output_directory, percentile=93, subsample=True)
            create_seeded_percentile_models(target, X_val_subbed, full_pred, X_train_subbed, y_train_subbed, n_est, fold_length, sub_img_shape, data_output_directory, models_output_directory, percentile=95, subsample=True)
            create_seeded_percentile_models(target, X_val_subbed, full_pred, X_train_subbed, y_train_subbed, n_est, fold_length, sub_img_shape, data_output_directory, models_output_directory, percentile=97, subsample=True)
            create_seeded_percentile_models(target, X_val_subbed, full_pred, X_train_subbed, y_train_subbed, n_est, fold_length, sub_img_shape, data_output_directory, models_output_directory, percentile=99, subsample=True)


def save_model(model, models_output_directory, filename):
    print("Saving Model")
    path = os.path.join(models_output_directory, f"{filename}.joblib")
    dump(model, path)
    print(f"+w {path}")


def create_seeded_percentile_models(target, X_subbed_list, prediction, X_val_list, y_val_list, n_est, fold_length, sub_img_shape, data_output_directory, models_output_directory, percentile=50, subsample=False):
    """ Creates seeded models based on the percentile passed """
    y_subbed_training = create_percentile_map(prediction, percentile, fold_length, data_output_directory)
    save_subimg_maps(y_subbed_training, sub_img_shape, data_output_directory, target, f"{percentile}-percentile_map")
    print(f"Seeded Percentile Model: {percentile}")
    seeded_models = train_kfold_model(target, X_subbed_list, y_subbed_training, X_val_list, y_val_list, n_est, sub_img_shape, data_output_directory, models_output_directory, f"seeded-{percentile}percentile", subsample=subsample)
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


def train_kfold_model(target, X_training_list, y_training_list, X_val_list, y_val_list, n_est, sub_img_shape, data_output_directory, models_output_directory, filename, initial_model=False, subsample=False):
    """ Creates 5 fold models, returns a list of models, one per fold """
    print("Starting Training")
    models = list()
    params = {
                        'n_estimators': n_est,
                        # 'max_features': 0.5,
                        'max_depth': 16,
                        'verbose': 0,
                        'n_jobs': -1,
                        # 'bootstrap': False,
                        'oob_score': True,
                        'warm_start': True
                    }
    clf = RandomForestClassifier(**params)
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

    try:
        if subsample:
            for x in range(5):
                print("Subsampling enabled")
                print("Fitting several times...")

                rus = RandomUnderSampler(sampling_strategy=.5)
                X_train, y_train = rus.fit_resample(X_train_all, y_train_all)
                print(f"X_train shape: {X_train.shape}")
                print(f"y_train shape: {y_train.shape}")
                vals, counts = np.unique(y_train, return_counts=True)
                print(f"y_train distribution: \n{vals}\n{counts}")

                clf.fit(X_train, y_train)
                clf.n_estimators += n_est
        else:
            print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            vals, counts = np.unique(y_train, return_counts=True)
            print(f"y_train distribution: \n{vals}\n{counts}")
            clf.fit(X_train_all, y_train_all)
    except Exception as e:

        print(f"Error, skipping image: {e}")
    save_model(clf, models_output_directory, f"{target}-{filename}")
    print("Validation Prediction")
    for idx_i, X_val in enumerate(X_val_list):
        X_val = X_val.reshape(X_val.shape[0] * X_val.shape[1], X_val.shape[2])
        class_prediction = clf.predict(X_val)
        proba_prediction = clf.predict_proba(X_val)[:,1] # the true values probability
        print(class_prediction.shape)
        print(proba_prediction.shape)
        save_np(class_prediction, os.path.join(data_output_directory, f"val_{target}_{filename}_class-prediction_{idx_i}"))
        save_np(proba_prediction, os.path.join(data_output_directory, f"val_{target}_{filename}_proba-prediction_{idx_i}"))
    print("Finished Training\n")


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
