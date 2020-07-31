import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import math
import sys
sys.path.append(os.curdir) # so python can find Utils
from Utils.Misc import *

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
        "conifer" : "CONIFER.bin",
        "water": "WATER.bin"
    }
    for target in targets:
        reference_path = f'{reference_data_root}{targets[target]}'
        cols, rows, bands, y = read_binary(reference_path, to_string=False)
        y = convert_y_to_binary(target, y, cols, rows)
        y_subbed_list = create_sub_imgs(y, fold_length)
        for x, sub_img in enumerate(y_subbed_list):
            save_np(sub_img.reshape(cols, rows), os.path.join(data_output_directory, f"{target}_map-{x}"))


        # Now conduct K Fold Training
        print("Commence K-Fold Training")
        for test_idx in range(5):
            print(f"Test Index: {test_idx}")
            X_test = X_subbed_list[test_idx]
            y_test = y_subbed_list[test_idx]
            X_train_total = None
            for train_idx in range(5):
                if test_idx == train_idx:
                    continue
                if X_train_total is None:
                    X_train_total = X_subbed_list[train_idx]
                    y_train_total = y_subbed_list[train_idx]
                else:
                    X_train_total = np.concatenate((X_train_total, X_subbed_list[train_idx]))
                    y_train_total = np.concatenate((y_train_total, y_subbed_list[train_idx]))
            random_indexed = np.random.shuffle(y_train_total)
            X_train_total = X_train_total[random_indexed]
            y_train_total = y_train_total[random_indexed]



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