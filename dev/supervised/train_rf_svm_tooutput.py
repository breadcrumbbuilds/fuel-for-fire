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

    Using K-Fold, create K Models,
    for each model,
    predict the class and probability of each pixel in that model's test image
    save these results to use later
    find the appropriate threshold for the probability prediciton
    Use this to create a seeded map
    Train a model with the seeded data
    predict class and probability
    """

    root = "data/full/"
    train_root_path = f"{root}/prepared/train/"
    reference_data_root = f"{root}data_bcgw/"
    raw_data_root = f"{root}data_img/"
    data_output_directory, results_output_directory = get_working_directories("KFold/Seeded")
    X = np.load(f'{train_root_path}/full-img.npy')
    sub_img_shape = (4835//5,3402)
    fold_length = X.shape[0] // 5

    X_subbed_list = create_sub_imgs(X, fold_length) # split the orig data into 5 sub images
    save_rgb(X_subbed_list, sub_img_shape, data_output_directory) # save the rgb in the output dir for later use

    targets = {
        #"conifer" : "CONIFER.bin",
        "water": "WATER.bin"
    }
    for target in targets:
        cols, rows, bands, y = read_binary(f'{reference_data_root}{targets[target]}', to_string=False)
        y = convert_y_to_binary(target, y, cols, rows)
        y_subbed_list = create_sub_imgs(y, fold_length)
        for x, sub_img in enumerate(y_subbed_list):
            save_np(sub_img.reshape(sub_img_shape), os.path.join(data_output_directory, f"{target}_map-{x}"))

        """ Initial K-Fold training """
        print("Training")
        n_est = 2
        initial_models = train_kfold_model(target, X_subbed_list, y_subbed_list, n_est, sub_img_shape, data_output_directory, "initial")
        for x, model in enumerate(seeded_models):
            dump(model, os.path.join(data_output_directory, f"initial_rf_{x}.joblib"))
            # Save the model here

        """ Find Threshold Value """
        proba_predictions = None
        for x in range(5):
            if proba_predictions is None:
                proba_predictions = load_np(os.path.join(data_output_directory, f"{target}_initial_proba-prediction-{x}.npy"))
            else:
                proba_predictions = np.concatenate((proba_predictions, load_np(os.path.join(data_output_directory, f"{target}_initial_proba-prediction-{x}.npy"))))

        percentile_60 = np.percentile(proba_predictions, 60)
        probability_map = proba_predictions > percentile_60
        y_subbed_list = create_sub_imgs(probability_map, fold_length)
        seeded_models = train_kfold_model(target, X_subbed_list, y_subbed_list, n_est, sub_img_shape, data_output_directory, "seeded-60percentile")
        dump(seeded_model, os.path.join(data_output_directory, f"seeded_rf_{percentile_60}percentile.joblib"))
        for x, model in enumerate(seeded_models):
            dump(model, os.path.join(data_output_directory, f"seeded_rf_{percentile_60}percentile_{x}.joblib"))


        percentile_75 = np.percentile(proba_predictions, 75)
        probability_map = proba_predictions > percentile_75
        y_subbed_list = create_sub_imgs(probability_map, fold_length)
        seeded_models = train_kfold_model(RandomForestClassifier(**params),target, X_subbed_list, y_subbed_list, n_est, sub_img_shape, data_output_directory, "seeded-75percentile")
        dump(seeded_model, os.path.join(data_output_directory, f"seeded_rf_{percentile_75}percentile.joblib"))
        for x, model in enumerate(seeded_models):
            dump(model, os.path.join(data_output_directory, f"seeded_rf_{percentile_75}percentile_{x}.joblib"))


        percentile_90 = np.percentile(proba_predictions, 90)
        probability_map = proba_predictions > percentile_90
        y_subbed_list = create_sub_imgs(probability_map, fold_length)
        seeded_models = train_kfold_model(target, X_subbed_list, y_subbed_list, n_est, sub_img_shape, data_output_directory, "seeded-90percentile")
        dump(seeded_model, os.path.join(data_output_directory, f"seeded_rf_{percentile_90}percentile.joblib"))
        for x, model in enumerate(seeded_models):
            dump(model, os.path.join(data_output_directory, f"seeded_rf_{percentile_90}percentile_{x}.joblib"))


        mean = np.mean(proba_predictions)
        probability_map = proba_predictions > mean
        y_subbed_list = create_sub_imgs(probability_map, fold_length)
        seeded_models = train_kfold_model(target, X_subbed_list, y_subbed_list, n_est, sub_img_shape, data_output_directory, f"seeded-mean-{mean}")
        dump(seeded_model, os.path.join(data_output_directory, f"seeded_rf_mean{mean}.joblib"))
        for x, model in enumerate(seeded_models):
            dump(model, os.path.join(data_output_directory, f"seeded_rf_{mean}mean_{x}.joblib"))



        """ Seeded K-Fold training """


def train_kfold_model(target, X_subbed_list, y_subbed_list, n_est, sub_img_shape, data_output_directory, filename):
    models = list()
    for test_idx in range(5):
        print(f"Test Index: {test_idx}")
        X_test = X_subbed_list[test_idx]
        y_test = y_subbed_list[test_idx]
        params = {
                        'n_estimators': n_est,
                        'max_depth': 6,
                        'verbose': 1,
                        'n_jobs': -1,
                        # 'bootstrap': False,
                        'oob_score': True,
                        'warm_start': True
                    }
        clf = RandomForestClassifier(**params)
        for train_idx in range(5):
            if test_idx == train_idx:
                continue
            print(f"Fitting Image {train_idx}")
            X_train = X_subbed_list[train_idx]
            y_train = y_subbed_list[train_idx]
            random_indexed = np.random.shuffle(y_train)
            X_train = np.squeeze(X_train[random_indexed])
            y_train = np.squeeze(y_train[random_indexed])
            clf.fit(X_train, y_train)
            clf.n_estimators += n_est
        print(f"Prediction Test Index {test_idx}")
        class_prediction = clf.predict(X_test)
        proba_prediction = clf.predict_proba(X_test)[:,1]
        save_np(class_prediction.reshape(sub_img_shape), os.path.join(data_output_directory, f"{target}_{filename}_class-prediction-{test_idx}"))
        save_np(proba_prediction.reshape(sub_img_shape), os.path.join(data_output_directory, f"{target}_{filename}_proba-prediction-{test_idx}"))
        plt.imshow(class_prediction.reshape(sub_img_shape), cmap='gray')
        plt.show()
        plt.imshow(proba_prediction.reshape(sub_img_shape), cmap='gray')
        plt.show()
        models.append(clf)
    return models


def create_sub_imgs(data, fold_length):
    """ Splits X into 5 sub images of equal size and return the sub images in a list """
    result = list()
    if len(data.shape) > 1:
        for x in range(5):
            result.append(data[x * fold_length : (x+1) * fold_length, :])
    else:
        for x in range(5):
            result.append(data[x * fold_length : (x+1) * fold_length])
    return result



def save_rgb(subimgs, sub_img_shape, output_directory):
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
            rgb_stretched[:,:, i] = rescale(rgb[:,:, i], two_percent=True)
        # plt.imshow(rgb)
        # plt.show()
        save_np(rgb, os.path.join(output_directory, f"rgb_subimage-{x}"))
        save_np(rgb, os.path.join(output_directory, f"rgb_subimage-{x}-twopercentstretch"))
    print()

if __name__ == "__main__":
   main()