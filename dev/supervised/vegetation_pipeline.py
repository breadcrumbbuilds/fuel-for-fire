import sys
import os
import json
import multiprocessing as mp
import numpy as np
sys.path.append(os.curdir)
from Utils.Misc import read_binary, get_working_directories, save_np, save_model, join_path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
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
    'D-1/2',
    'NonFuel',
    'Water',
    'M-1 C25',
    'M-1/2 C35',
    'M-1/2 C50',
    'M-1/2 C65'
    ]

is_vegetation = [False,
                 False,
                 False,
                 False,
                 False,
                 False,
                 False,
                 False,
                 False,
                 False,
                 False,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 True,
                 False,
                 False,
                 True,
                 True,
                 True,
                 True,
                 ]

directory = get_working_directories(
    'pipeline/vegetation-binary', ['data', 'params', 'results', 'model'])
""" Read Data """
cols, rows, bands, data = read_binary('data/update-2020-09/stack_v2.bin', to_string=False)
X = data[:,:11]
test_size = .8
y =  np.zeros((cols * rows), dtype=int)
for x in range(11,24):
    if is_vegetation[x]:
    	y = data[:, x].astype(int) | y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
rus = RandomUnderSampler(sampling_strategy=1)
X_train_sub, y_train_sub = rus.fit_resample(X_train, y_train)
scaler = StandardScaler().fit(X_train_sub)
X_train_sub = scaler.transform(X_train_sub)

pipeline = Pipeline([
    	('kmeans', KMeans()),
    	('rf', RandomForestClassifier(n_jobs=-1))
    ])

param_grid = dict(kmeans__n_clusters=range(1,101, 25),
					rf__max_depth=[3, 8],
					rf__max_features=[0.1, 0.5],
					rf__n_estimators=[50, 500]
					)
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=0, n_jobs=-1)

""" Fit, Predict, and Display Results """
grid_clf.fit(X_train_sub,y_train_sub)

X_scaled = scaler.transform(X)
print('Predicting')
y_full_probapred = grid_clf.predict_proba(X_scaled)[:,1]
save_np(y_full_probapred, join_path(directory['data'], f'vegetation_probapred'))

y_full_pred = grid_clf.predict(X_scaled)
save_np(y_full_pred, join_path(directory['data'], f'vegetation_pred'))

y_test_pred = grid_clf.predict(scaler.transform(X_test))
y_train_pred = grid_clf.predict(X_train_sub)

print('Calculating Scores')
test_score = round(balanced_accuracy_score(y_test, y_test_pred), 3)
train_score = round(balanced_accuracy_score(y_train_sub, y_train_pred), 3)

figure, axes = plt.subplots(1, 3, sharex=True, figsize=(20,10))
y_shaped = y.reshape(rows, cols)
save_np(y_full_pred, join_path(directory['data'], f'vegetation_reference'))
axes[0].imshow(y_shaped, cmap='gray')
axes[0].set_title(f'Reference Map')

pred_shaped = y_full_pred.reshape(rows, cols)
axes[1].imshow(pred_shaped, cmap='gray', vmin=0, vmax=1)
axes[1].set_title('Predict')

probapred_shaped = y_full_probapred.reshape(rows, cols)
axes[2].imshow(probapred_shaped, cmap='gray', vmin=0, vmax=1)
axes[2].set_title('Predict Proba')

# figure.colorbar(probapred_shaped, ax=1, orientation='horizontal', fraction=.5)
figure.suptitle(f'KMeans -> RF: vegetation - Test: {test_score}, Train: {train_score}, Test Size: {test_size}')
plt.tight_layout()
plt.savefig(join_path(directory['results'], f'vegetation'))

json_object = json.dumps(grid_clf.best_params_, indent = 4)
with open(join_path(directory['params'], f"vegetation.json"), "w") as outfile:
	outfile.write(json_object)

save_model(grid_clf, join_path(directory['model'], f'vegetation'))
plt.imshow(y.reshape(rows, cols), cmap='gray')
plt.show()