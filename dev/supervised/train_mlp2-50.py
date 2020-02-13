import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from rasterbox.Rasterbox import Rasterbox as rb
from Utils.targets import get_bcgw_targets
from Models.Tensorflow import LayersMultiLayerPerceptron2_50
from Utils.Helper import Helper
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    targets = get_bcgw_targets()

    data = rb("data", "data_img", "data_bcgw", targets)
    X = data.Combined.Data()
    y = data.Onehot.Data()
    X = X.reshape(164410, 23)

    # split and normalize data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mean_vals, std_val = Helper.mean_center_normalize(X_train)

    X_train_centered = (X_train - mean_vals) / std_val
    X_test_centered = (X_test - mean_vals) / std_val

    del X_train, X_test

    print("Training Samples" )
    print(X_train_centered.shape, y_train.shape)
    print()
    print('Testing Samples')
    print(X_test_centered.shape, y_test.shape)
    print()


    mlpmodel = LayersMultiLayerPerceptron2_50(X_test_centered.shape[1], len(targets), learning_rate=0.001)

    sess = tf.Session(graph=mlpmodel.g)
    training_costs = LayersMultiLayerPerceptron2_50.train_mlp(sess,
                                                    mlpmodel,
                                                    X_train_centered,
                                                    y_train,
                                                    num_epochs=100)

    # y_pred = LayersMultiLayerPerceptron2_50.predict_mlp(sess, mlpmodel, X_test_centered)

    # print('Test Accuracy: %.2f%%' % (
    #     100*np.sum(y_pred == y_test) / y_test.shape[0]
    # ))
