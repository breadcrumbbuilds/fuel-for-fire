import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from Models import TFLinreg


if __name__ == "__main__":

    X_train = np.arange(10).reshape(10,1)
    y_train = np.array([1.,1.3,3.1,
        2.,5.,6.3,6.6,7.4,8.,9.])

    lrmodel = TFLinreg(x_dim=X_train.shape[1], learning_rate=0.01)

    sess = tf.Session(graph=lrmodel.g)
    training_costs = TFLinreg.train_linreg(sess, lrmodel, X_train, y_train)

    plt.plot(range(1, len(training_costs) + 1), training_costs)
    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('Training Costs')
    plt.show()

    plt.scatter(X_train, y_train, marker='s', s=50, label='Training Data')
    plt.plot(range(X_train.shape[0]),
             TFLinreg.predict_linreg(sess,lrmodel,X_train),
             color='gray', marker='o',
             markersize=6, linewidth=3,
             label='LinReg Model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()
