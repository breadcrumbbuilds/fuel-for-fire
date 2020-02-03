import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Utils.Misc import *
from Utils.Data import Data
from Utils.DataTest import *
from Utils.Model import LayersMultiLayerPerceptron2_50
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    data = Data("data", "data_img", "data_bcgw")
    X = data.S2.ravel()
    y = data.labels_onehot()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mean_vals, std_val = DataTest.mean_center_normalize(X_train)

    X_train_centered = (X_train - mean_vals) / std_val
    X_test_centered = (X_test - mean_vals) / std_val

    del X_train, X_test

    print(X_train_centered.shape, y_train.shape)
    print(X_test_centered.shape, y_test.shape)

    mlpmodel = LayersMultiLayerPerceptron2_50(X_test_centered.shape[1], 9, learning_rate=0.001)

    sess = tf.Session(graph=mlpmodel.g)
    training_costs = LayersMultiLayerPerceptron2_50.train_mlp(sess,
                                                    mlpmodel,
                                                    X_train_centered,
                                                    y_train,
                                                    num_epochs=100)

    y_pred = LayersMultiLayerPerceptron2_50.predict_mlp(sess, mlpmodel, X_test_centered)

    print('Test Accuracy: %.2f%%' % (
        100*np.sum(y_pred == y_test) / y_test.shape[0]
    ))