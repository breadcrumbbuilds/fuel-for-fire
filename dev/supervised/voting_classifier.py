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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score

def main():
    """ Run From Project Root
    """
    root = "data/full/"
    train_root_path = f"{root}/prepared/train/"
    reference_data_root = f"{root}data_bcgw/"
    raw_data_root = f"{root}data_img/"

    data_output_directory, results_output_directory, models_output_directory = \
        get_working_directories(f"VotingClassifier/")

    sample_ratio = 5
    spatial_shape = (4835, 3402, 11)
    training_shape = (4835 * 3402, 11)
    X = np.load(f'{train_root_path}/full-img.npy').reshape(spatial_shape)
    save_rgb(X, spatial_shape, data_output_directory, "full")
    X = X.reshape(training_shape)
    X_train = X[::sample_ratio,:] # simple sample

    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X)

    targets = {
        "water": "WATER.bin",
        "herb": "HERB.bin",
        "broadleaf": "BROADLEAF.bin",
        "conifer": "CONIFER.bin",
        "mixed": "MIXED.bin",
        "ccut": "CCUTBL.bin",
        "exposed": "EXPOSED.bin"
    }

    classifier_names = ['LogisticRegression',
                        'RandomForest',
                        'ExtraTrees',
                        'GradientBoosting',
                        'Voting',]

    clf1 = LogisticRegression(random_state=1, max_iter=1000, n_jobs=-1)
    clf2 = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=3, n_jobs=-1)
    clf3 = ExtraTreesClassifier(random_state=1, n_estimators=100, max_depth=3, n_jobs=-1)
    clf4 = GradientBoostingClassifier(random_state=1, n_estimators=100, max_depth=3)
    # clf5 = SVC(kernel='linear')
    voter = VotingClassifier(estimators=[('lr', clf1),
                            ('rf', clf2),
                            ('et', clf3),
                            ('gb', clf4)],
                            # ('svc', clf5)],
                            voting='soft', n_jobs=-1, verbose=1)
    for target in targets:
        cv_scores = []
        for name, clf in zip(classifier_names, [clf1, clf2, clf3, clf4, voter]):
            print(name.upper())
            print('________________________________________________')

            """ Read in map and resample"""
            cols, rows, bands, y = read_binary(f'{reference_data_root}{targets[target]}', to_string=False)
            y = convert_y_to_binary(target, y, cols, rows).reshape(rows * cols)
            y_train = y[::sample_ratio] # simple sample whole image
            y_val = y[::7]
            rus = RandomUnderSampler(sampling_strategy=.5) # subsample the minority class
            X_train_sample, y_train_sample = rus.fit_resample(X_train, y_train)
            vals, counts = np.unique(y_train_sample, return_counts=True)
            print(f"X_train shape: {X_train_sample.shape}")
            print(f"y_train shape: {y_train_sample.shape}")
            print(f"y_train distribution: \n{vals}\n{counts}")

            """ Fit and produce outputs """
            print("Fitting")
            clf = clf.fit(X_train_sample, y_train_sample)
            print("Cross Validation")
            cv_scores.append((name, cross_val_score(clf, X_scaled[::7,:], y_val, scoring='balanced_accuracy', cv=5, n_jobs=-1).mean()))
            print("Probability Prediction")
            probability_pred = clf.predict_proba(X_scaled)[:,1]
            save_np(probability_pred, os.path.join(data_output_directory, f"{name}-{target}-proba_pred"))
            print("Class Prediction")
            class_pred = clf.predict(X_scaled)
            save_np(class_pred, os.path.join(data_output_directory, f"{name}-{target}-class_pred"))

            plt.title(f"{name} {target} Probability Prediction")
            plt.imshow(probability_pred.reshape(4835, 3402), cmap='gray', vmin=0, vmax=1, interpolation='none')
            plt.colorbar()
            plt.savefig(f"{results_output_directory}/{name}-{target}-proba_pred")
            plt.clf()

            plt.title(f"{name} {target} Class Prediction")
            plt.imshow(class_pred.reshape(4835, 3402), cmap='gray', vmin=0, vmax=1, interpolation='none')
            plt.colorbar()
            plt.savefig(f"{results_output_directory}/{name}-{target}-class_pred")
            plt.clf()
            print()
        with open(f'{results_output_directory}/{target}_cross_val_scores.txt', 'w') as f:
            for x in cv_scores:
                f.write(f'{x}\n')

if __name__ == "__main__":
   main()
