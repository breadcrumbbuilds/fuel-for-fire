import sys
import os
import json
import multiprocessing as mp
import numpy as np
sys.path.append(os.curdir)
from functools import partial
from tensorflow import keras
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


directory = get_working_directories(
    'keras/test', ['data', 'params', 'results', 'model'])
""" Read Data """
cols, rows, bands, data = read_binary('data/update-2020-09/stack_v2.bin', to_string=False)
X = data[:,:11]
y = data[:,11:]

test_size=.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size)


print(X_train.shape, y_train.shape)
# rus = RandomUnderSampler(sampling_strategy=1)
# X_train_sub, y_train_sub = rus.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

RegularizedDense = partial(keras.layers.Dense, activation='relu',
                           kernel_initializer='he_normal',
                           kernel_regularizer=keras.regularizers.l2(0.001))
RegularizedDense2 = partial(keras.layers.Dense, activation='selu',
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(0.001))
input_ = keras.layers.Input(shape=[11])
hidden1 = RegularizedDense(128)(input_)
hidden2 = RegularizedDense(128)(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
hidden3 = RegularizedDense(256)(concat)
hidden4 = RegularizedDense(256)(hidden3)

hidden5a = RegularizedDense(256)(concat)
hidden6a = RegularizedDense(256)(hidden5a)
hidden7a = RegularizedDense(1024)(hidden6a)

concat2 = keras.layers.Concatenate()([hidden2, hidden4, hidden7a])
hidden5b = RegularizedDense(256)(concat2)
hidden6b = RegularizedDense(256)(hidden5b)
hidden7b = RegularizedDense(512)(hidden6b)
hidden8b = RegularizedDense(512)(hidden7b)
hidden9b = RegularizedDense(2048)(hidden8b)


offshoot5b1 = RegularizedDense(256, activation='selu')(hidden5b)
offshoot5b2 = RegularizedDense(256, activation='selu')(offshoot5b1)
offshoot5b3 = RegularizedDense(256, activation='selu')(offshoot5b2)
offshoot5b4 = RegularizedDense(2048, activation='selu')(offshoot5b3)

offshoot6b1 = RegularizedDense(256, activation='selu')(hidden6b)
offshoot6b2 = RegularizedDense(256, activation='selu')(offshoot6b1)
offshoot6b3 = RegularizedDense(256, activation='selu')(offshoot6b2)
offshoot6b4 = RegularizedDense(2048, activation='selu')(offshoot6b3)

concat3 = keras.layers.Concatenate()([offshoot5b4, offshoot6b4])

offshoot7b1 = RegularizedDense2(512)(hidden7b)
offshoot7b2 = RegularizedDense2(512)(offshoot7b1)
offshoot7b3 = RegularizedDense2(512)(offshoot7b2)
offshoot7b4 = RegularizedDense2(2048)(offshoot7b3)

offshoot8b1 = RegularizedDense2(512)(hidden8b)
offshoot8b2 = RegularizedDense2(512)(offshoot8b1)
offshoot8b3 = RegularizedDense2(512)(offshoot8b2)
offshoot8b4 = RegularizedDense2(2048)(offshoot8b3)

concat4 = keras.layers.Concatenate()([offshoot7b4, offshoot8b4])


concat5 = keras.layers.Concatenate()([hidden9b, concat3, concat4])

hidden10 = hidden9b = RegularizedDense2(8192)(concat5)

output = keras.layers.Dense(13, activation='softmax')(hidden10)

model = keras.Model(inputs=[input_], outputs=[output])

print(model.summary())

METRICS = [
      keras.metrics.CategoricalAccuracy(name='accuracy'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]



model.compile(optimizer=keras.optimizers.Adam(lr=0.003333),
              loss='categorical_crossentropy',
              metrics=METRICS)
checkpoint_cb = keras.callbacks.ModelCheckpoint(join_path(directory['model'],"model.h5"), save_best_only=True)
print(f'Tensorboard at {directory["root"]}')
tboard_cb = keras.callbacks.TensorBoard(directory['root'])
model.fit(X_train, y_train,
          use_multiprocessing=True,
          workers=16,
          epochs=100,
          validation_data=(X_valid, y_valid),
          batch_size=8192,
          callbacks=[checkpoint_cb, tboard_cb],
          )
model = model.load(join_path(directory['model'], 'model.h5'))

pred = model.predict(scaler.tranform(X))

for p in zip(pred, y):
    print(p)

