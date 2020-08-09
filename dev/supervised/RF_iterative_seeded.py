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


def main():
    """ Run From Project Root
    """
    root = "data/full/"
    train_root_path = f"{root}/prepared/train/"
    reference_data_root = f"{root}data_bcgw/"
    raw_data_root = f"{root}data_img/"
    data_output_directory, results_output_directory, models_output_directory = get_working_directories("KFold/Seeded")

    X = np.load(f'{train_root_path}/full-img.npy')

    sub_img_shape = (4835//5,3402//2)
    fold_length = X.shape[0] // 10

    X_train_subbed, X_val_subbed = split_train_val(X, fold_length) # split the orig data into 5 sub images
    save_rgb(X_train_subbed, sub_img_shape, data_output_directory, 'train') # save the rgb in the output dir for later use
    save_rgb(X_train_subbed, sub_img_shape, data_output_directory, 'validation') # save the rgb in the output dir for later use

    del X

    targets = {
        "water": "WATER.bin",
        "conifer" : "CONIFER.bin",
        "herb" : "HERB.bin",
        "shrub" : "SHRUB.bin"
    }
    for target in targets:
        cols, rows, bands, y = read_binary(f'{reference_data_root}{targets[target]}', to_string=False)
        y = convert_y_to_binary(target, y, cols, rows)
        y_subbed_training, y_subbed_validation = split_train_val(y, fold_length)
        del y
        # save the original maps to disk
        save_subimg_maps(y_subbed_training, sub_img_shape, data_output_directory, target, "training_map")
        save_subimg_maps(y_subbed_validation, sub_img_shape, data_output_directory, target, "validation_map")

        """ Initial K-Fold training """
        print("Training")
        n_est = 25

        train_kfold_model(target, X_subbed_list, y_subbed_training, y_subbed_validation, n_est, sub_img_shape, data_output_directory, models_output_directory, "initial", initial_model=True)


        """ Each of the initial five models predicted on the validation set.
            Take the intersection of the five models and treat that as the
            seeded data's ground truth """
        proba_predictions = None
        for x in range(5):
            this_prediction = load_np(os.path.join(data_output_directory, f"val_{target}_initial_proba-prediction-{x}.npy")).ravel()
            if proba_predictions is None:
                proba_predictions = this_prediction
            else:
                proba_predictions = np.logical_and(proba_predictions, this_prediction)

        create_seeded_percentile_models(target, X_subbed_list, proba_predictions, n_est, fold_length, sub_img_shape, data_output_directory, models_output_directory, percentile=85)
        create_seeded_percentile_models(target, X_subbed_list, proba_predictions, n_est, fold_length, sub_img_shape, data_output_directory, models_output_directory, percentile=90)
        create_seeded_percentile_models(target, X_subbed_list, proba_predictions, n_est, fold_length, sub_img_shape, data_output_directory, models_output_directory, percentile=95)
        create_seeded_percentile_models(target, X_subbed_list, proba_predictions, n_est, fold_length, sub_img_shape, data_output_directory, models_output_directory, percentile=99)

        # mean = np.mean(proba_predictions)
        # probability_map = proba_predictions > mean
        # y_subbed_list = create_sub_imgs(probability_map, fold_length)
        # save_subimg_maps(y_subbed_list, sub_img_shape, data_output_directory, target, f"mean-{mean}")
        # seeded_models = train_kfold_model(target, X_subbed_list, y_subbed_list, n_est, sub_img_shape, data_output_directory, models_output_directory, f"seeded-mean-{mean}")


def save_model(model, models_output_directory, filename):
    print("Saving Model")
    path = os.path.join(models_output_directory, f"{filename}.joblib")
    dump(model, path)
    print(f"+w {path}")


def create_seeded_percentile_models(target, X_subbed_list, proba_predictions, n_est, fold_length, sub_img_shape, data_output_directory, models_output_directory, percentile=50):
    """ Creates seeded models based on the percentile passed """
    y_subbed_training = create_percentile_map(proba_predictions, percentile, fold_length, data_output_directory)
    save_subimg_maps(y_subbed_training, sub_img_shape, data_output_directory, target, f"{percentile}-percentile_map")
    print(f"Seeded Percentile Model: {percentile}")
    seeded_models = train_kfold_model(target, X_subbed_list, y_subbed_list, n_est, sub_img_shape, data_output_directory, models_output_directory, f"seeded-{percentile}percentile")
    return seeded_models


def create_percentile_map(data, percentile, fold_length, data_output_directory, seperate=False):
    """ Creates a binary map for each fold, taking the percentile of that fold
        data: numpy array
        percentile: floating point value that serves as threshold
        fold_length: describes the size of each sub image (fold)
        seperate: if True, calculates the threshold on each subimg, if False, calculates the threshold on the entire image
    """
    y_subbed_list = list()
    if seperate:
        print(f"Creating Percentile: {percentile} maps")
        for x in range(5):
            if x % 2 != 0:
                X = data[x * fold_length : (x+1) * fold_length]
                p = np.percentile(X, percentile)
                probability_map = X > p
                y_subbed_list.append(probability_map)
            else:
                temp = load_np(os.path.join(data_output_directory, f"water_initial_class-prediction-{x}.npy"))
                temp = temp > p
                y_subbed_list.append(temp)
    else:
        p = np.percentile(data, percentile)
        probability_map = data > p
        for x in range(10):
            if x % 2 != 0:
                y_subbed_list.append(probability_map[x * fold_length : (x+1) * fold_length])
            else:
                temp = load_np(os.path.join(data_output_directory, f"water_initial_class-prediction-{x}.npy"))
                temp = temp > p
                y_subbed_list.append(temp) # this won't be used, just a place holder

    return y_subbed_list


def train_kfold_model(target, X_subbed_list, y_subbed_list, n_est, sub_img_shape, data_output_directory, models_output_directory, filename, initial_model=False):
    """ Creates 5 fold models, returns a list of models, one per fold """
    print("Starting Training")
    X_val = None
    y_val = None
    models = list()
    for test_idx in range(10):
        if test_idx % 2 != 0 and initial_model:
            """Odd index == validation image, save for later"""
            if X_val is None:
                print("Initializing Validation")
                """Not Intialized yet"""
                X_val = X_subbed_list[test_idx]
                y_val = y_subbed_list[test_idx]
            else:
                print("Concat to Validation")
                print(X_val.shape)
                """Add to the validation set"""
                X_val = np.concatenate((X_val, X_subbed_list[test_idx]))
                y_val = np.concatenate((y_val, y_subbed_list[test_idx]))
                print(X_val.shape)
            continue
        else:
            if test_idx % 2 == 0 and not initial_model:
                """Is a seeded model, so the even images are val"""
                if X_val is None:
                    """Not Intialized yet"""
                    X_val = X_subbed_list[test_idx]
                    y_val = y_subbed_list[test_idx]
                else:
                    """Add to the validation set"""
                    X_val = np.concatenate((X_val, X_subbed_list[test_idx]))
                    y_val = np.concatenate((y_val, y_subbed_list[test_idx]))
                continue
        print(f"Test Index: {test_idx}")
        X_test = X_subbed_list[test_idx]
        y_test = y_subbed_list[test_idx]
        params = {
                        'n_estimators': n_est,
                        'max_features': 0.3,
                        'max_depth': 3,
                        'verbose': 0,
                        'n_jobs': -1,
                        # 'bootstrap': False,
                        'oob_score': True,
                        'warm_start': True
                    }
        clf = RandomForestClassifier(**params)
        for train_idx in range(10):
            if test_idx == train_idx:
                continue
            if train_idx % 2 != 0 and initial_model:
                continue
            if train_idx % 2 == 0 and not initial_model:
                continue
            print(f"Fitting Image {train_idx}")
            X_train = X_subbed_list[train_idx]
            y_train = y_subbed_list[train_idx]
            if len(X_test.shape) > 2:
                X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])
                y_train = y_train.ravel()
            random_indexed = np.random.shuffle(y_train)
            X_train = np.squeeze(X_train[random_indexed])
            y_train = np.squeeze(y_train[random_indexed])
            X_train = X_train[::100,:]
            y_train = y_train[::100]
            del random_indexed
            print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            vals, counts = np.unique(y_train, return_counts=True)
            print(f"y_train distribution: \n{vals}\n{counts}")
            if(len(vals) < 2):
                clf.n_estimators += n_est
                print("Only one value in distribtion, skipping")
                continue
            try:
                clf.fit(X_train, y_train)
            except:
                print("Error, skipping image")
            clf.n_estimators += n_est
        print(f"Prediction Test Index {test_idx}")
        try:
            class_prediction = clf.predict(X_test)
            proba_prediction = clf.predict_proba(X_test)[:,1]

            save_np(class_prediction.reshape(sub_img_shape), os.path.join(data_output_directory, f"{target}_{filename}_class-prediction-{test_idx}"))
            save_np(proba_prediction.reshape(sub_img_shape), os.path.join(data_output_directory, f"{target}_{filename}_proba-prediction-{test_idx}"))
        except:
            print("Failed to predict, moving on")
        try:
            save_model(clf, models_output_directory, f"{filename}-{test_idx}")
            models.append(clf)
        except:
            print("Failed to save model, moving on")
    print("Validation Prediction")
    for idx, model in enumerate(models):
        class_prediction = model.predict(X_val)
        proba_prediction = model.predict_proba(X_val)[:,1]
        save_np(class_prediction, os.path.join(data_output_directory, f"val_{target}_{filename}_class-prediction-{idx}"))
        save_np(proba_prediction, os.path.join(data_output_directory, f"val_{target}_{filename}_proba-prediction-{idx}"))
    print("Finished Training\n")


def split_train_val(data, fold_length):
    """ Splits X into 5 sub images of equal size and return the sub images in a list """
    train = list()
    val = list()
    if len(data.shape) > 1:
        for x in range(10):
            if x % 2 == 0:
                print(f'Training: {x}')
                train.append(data[x * fold_length : (x+1) * fold_length, :])
            else:
                print(f'Validation: {x}')
                val.append(data[x * fold_length : (x+1) * fold_length, :])
    else:
        for x in range(10):
            if x % 2 == 0:
                print(f'Training: {x}')
                train.append(data[x * fold_length : (x+1) * fold_length])
            else:
                print(f'Validation: {x}')
                val.append(data[x * fold_length : (x+1) * fold_length])
    return train, val


def save_subimg_maps(y_subbed_list, sub_img_shape, data_output_directory, target, filename):
    print("Saving sub image maps")
    for x, sub_img in enumerate(y_subbed_list):
        save_np(sub_img.reshape(sub_img_shape), os.path.join(data_output_directory, f"{target}_{filename}-{x}"))


def save_rgb(subimgs, sub_img_shape, output_directory, name):
    """ Saves each subimgs RGB interpretation to output_directory """
    print("Creating RGB sub images")
    sub_imgs = list()
    rgb = np.zeros((sub_img_shape[0], sub_img_shape[1], 3))
    rgb_stretched = np.zeros((sub_img_shape[0], sub_img_shape[1], 3))
    for x, data in enumerate(subimgs):
        for i in range(0,3):
            rgb[:,:, i] = data[:, 4 - i].reshape(sub_img_shape)
        for i in range(0,3):
            rgb[:,:, i] = rescale(rgb[:,:, i], two_percent=False)
            rgb_stretched[:,:, i] = rescale(np.copy(rgb[:,:, i]), two_percent=True)
        save_np(rgb, os.path.join(output_directory, f"rgb_{name}_subimage-{x}"))
        save_np(rgb, os.path.join(output_directory, f"rgb_{name}_subimage-{x}-twopercentstretch"))
    print()

if __name__ == "__main__":
   main()
