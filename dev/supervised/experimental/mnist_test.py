import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
tf.disable_v2_behavior()
from Utils.Data import DataTest
from Utils.Model import LayersMultiLayerPerceptron

if __name__ == "__main__":

    X_train, y_train = DataTest.load_mnist('./mnist/', kind='train')
    print('TRAIN - Rows: %d, Columns: %d' %(X_train.shape[0],X_train.shape[1]))

    X_test, y_test = DataTest.load_mnist('./mnist/', kind='t10k')
    print('TEST - Rows: %d, Columns: %d' %(X_test.shape[0],X_test.shape[1]))

    ## mean centering and normalization
    mean_vals, std_val = DataTest.mean_center_normalize(X_train)

    X_train_centered = (X_train - mean_vals) / std_val
    X_test_centered = (X_test - mean_vals) / std_val

    del X_train, X_test

    print(X_train_centered.shape, y_train.shape)
    print(X_test_centered.shape, y_test.shape)

    mlpmodel = LayersMultiLayerPerceptron(X_test_centered.shape[1], 10)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True),
                        graph=mlpmodel.g)
    training_costs = LayersMultiLayerPerceptron.train_mlp(sess,
                                                    mlpmodel,
                                                    X_train_centered,
                                                    y_train,
                                                    num_epochs=10)

    y_pred = LayersMultiLayerPerceptron.predict_mlp(sess, mlpmodel, X_test_centered)


    print('Test Accuracy: %.2f%%' % (
        100*np.sum(y_pred == y_test) / y_test.shape[0]
    ))

    plt.plot(range(1, len(training_costs) + 1), training_costs)
    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('Training Costs')
    plt.show()

    plt.scatter(X_train_centered, y_train, marker='s', s=50, label='Training Data')
    plt.plot(range(X_train_centered.shape[0]),
             y_pred,
             color='gray', marker='o',
             markersize=6, linewidth=3,
             label='MLP Model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()