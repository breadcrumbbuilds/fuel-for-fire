import sys
import os
import json
import multiprocessing as mp
import numpy as np
sys.path.append(os.curdir)
from Utils.Misc import read_binary, bsq_to_scikit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

""" Read Data """
cols, rows, bands, data = read_binary('data/update-2020-09/stack_v2.bin', to_string=False)
X = data[:,:11]
for x in range(11, 24):
    print(x)
    y = data[:, x]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.90)
    rus = RandomUnderSampler(sampling_strategy=1)
    X_train_sub, y_train_sub = rus.fit_resample(X_train, y_train)
    scaler = StandardScaler().fit(X_train_sub)
    X_train_sub = scaler.transform(X_train_sub)
    """ Model Pipeline """
    pipeline = Pipeline([
    	('kmeans', KMeans()),
    	('rf', RandomForestClassifier(n_jobs=-1, max_depth=3, max_features=.1, n_estimators=100))
    ])
    param_grid = dict(kmeans__n_clusters=range(10,100, 10),
                      rf__max_depth=range(3,5, 2),
                      rf__max_features=[0.1, 'auto'],
                      # rf__n_estimators=[50, 500]
                      )
    grid_clf = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)
    """ Fit, Predict, and Display Results """
    grid_clf.fit(X_train_sub,y_train_sub)
    res = grid_clf.predict_proba(scaler.transform(X))[:,1]

    print('Predicting')
    y_test_pred = grid_clf.predict(scaler.transform(X_test))
    y_train_pred = grid_clf.predict(X_train_sub)

    print('Calculating Scores')
    test_score = round(balanced_accuracy_score(y_test, y_test_pred), 3)
    train_score = round(balanced_accuracy_score(y_train_sub, y_train_pred), 3)
    figure, axes = plt.subplots(1, 2, sharex=True, figsize=(20,10))
    axes[0].imshow(y.reshape(rows, cols), cmap='gray')
    axes[0].set_title(f'Binary Map {x}')
    axes[1].imshow(res.reshape(rows, cols), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Predict Proba')

    figure.suptitle(f'Test: {test_score}, Train: {train_score}')
    plt.tight_layout()
    plt.savefig(f'grid_pipe{x}')

    json_object = json.dumps(grid_clf.best_params_, indent = 4)
    with open(f"{x}_bestparams.json", "w") as outfile:
    	outfile.write(json_object)