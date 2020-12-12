import sys
import os
import json
import numpy as np
sys.path.append(os.curdir)
from Utils.Misc import read_binary, \
    get_working_directories, \
    save_np, \
    save_model, \
    join_path, \
    value_counts, \
    mkdir
from Utils.Data import read_update_202009, torgb
from sklearn.metrics import precision_recall_curve,plot_precision_recall_curve,plot_confusion_matrix,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Modify me if you don't have enough RAM
param_grid = dict(
    kmeans__n_clusters=range(25, 51, 25),
    rf__max_features=[0.1, 0.3],
    rf__n_estimators=[100, 333],
    us__sampling_strategy=[0.5,0.75,1]
                    )
test_size = .90
verbosity = 1

band_names = [
    'B4 (665 nm)',
    'B3 (560 nm)',
    'B2 (490 nm)',
    'B8 (842 nm)',
    'SRB5 (705 nm)',
    'SRB6 (740 nm)',
    'SRB7 (783 nm)',
    'SRB8A (865 nm)',
    'SRB11 (1610 nm)',
    'SRB12 (2190 nm)',
    'SRB9 (945 nm)',
    'C-2',
    'C-3',
    'C-4',
    'C-5',
    'C-7',
    'D-1',
    'D-1-2',
    'NonFuel',
    'Water',
    'M-1 C25',
    'M-1-2 C35',
    'M-1-2 C50',
    'M-1-2 C65'
    ]

experiment_root = 'pipeline/kmeans-rf'
sub_dirs = ['data', 'params', 'results', 'model']
directory = get_working_directories(experiment_root, sub_dirs)
""" Read Data """
data_dir = 'data/update-2020-09/stack_v2.bin'
X, y = read_update_202009(data_dir)
rgb = torgb(join_path(directory['root'], data_dir.split('/')[2].replace('.bin', '')),
            X,
            (rows, cols, bands))

for x in reversed(range(11, 24)): # each class label
    band_name = band_names[x].replace(' ', '')
    working_directories = {}
    this_dir = \
        mkdir(join_path(join_path(directory['exp_root']), band_name))

    y = data[:, x]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    """ Model Pipeline """
    pipeline = Pipeline([
        ('us', RandomUnderSampler()),
        ('scale', StandardScaler()),
    	('kmeans', KMeans()),
    	('rf', RandomForestClassifier(n_jobs=-1, max_depth=3))
    ])

    grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=verbosity, n_jobs=-1)

    """ Fit, Predict, and Display Results """
    grid_clf.fit(X_train,y_train)

    print('Predicting')
    y_full_probapred = grid_clf.predict_proba(X)[:,1]
    save_np(y_full_probapred, join_path(this_dir, f'probapred'))

    y_full_pred = grid_clf.predict(X)
    save_np(y_full_pred, join_path(this_dir, f'pred'))

    y_test_pred = grid_clf.predict(X_test)
    y_train_pred = grid_clf.predict(X_train)

    print('Calculating Scores')
    test_score = round(balanced_accuracy_score(y_test, y_test_pred), 3)
    train_score = round(balanced_accuracy_score(y_train, y_train_pred), 3)
    figure, axes = plt.subplots(1, 4, sharex=True, figsize=(20,10))
    y_shaped = y.reshape(rows, cols)
    save_np(y_full_pred, join_path(this_dir, f'reference'))

    axes[0].imshow(rgb, cmap='gray')
    axes[0].set_title(f'RGB')

    axes[1].imshow(y_shaped, cmap='gray')
    axes[1].set_title(f'Reference Map')

    pred_shaped = y_full_pred.reshape(rows, cols)
    axes[2].imshow(pred_shaped, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Predict')

    probapred_shaped = y_full_probapred.reshape(rows, cols)
    axes[3].imshow(probapred_shaped, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Predict Proba')

    figure.suptitle(f'{band_name} - Test: {test_score}, Train: {train_score}, Test Size: {test_size}')
    plt.tight_layout()
    plt.savefig(join_path(this_dir, f'rgb_reference_pred'))
    plt.clf()
    average_precision = average_precision_score(y_test, y_test_pred)
    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    disp = plot_precision_recall_curve(grid_clf, X_test, y_test)
    disp.ax_.set_title(f'{band_name} - Precision-Recall curve: '
                    'AP={0:0.2f}'.format(average_precision))
    plt.savefig(join_path(this_dir,
                          f'precision_recall_curve'))
    plt.clf()
    np.set_printoptions(precision=2)

    titles_options = [("Confusion matrix, without normalization", None),
                    ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(grid_clf, X_test, y_test,
                                    display_labels=[False, True],
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
        disp.ax_.set_title(f'{band_name} - {title}')

        print(title)
        print(disp.confusion_matrix)
        plt.tight_layout()
        plt.savefig(join_path(this_dir, if normalize f'confusion_matrix_normalize' else: f'confusion_matrix'))
        plt.clf()
    json_object = json.dumps(grid_clf.best_params_, indent = 4)
    with open(join_path(this_dir, f"best_params.json"), "w") as outfile:
    	outfile.write(json_object)

    save_model(grid_clf, join_path(this_dir, 'model'))