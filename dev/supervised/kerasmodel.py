import tensorflow.keras as keras
import tensorflow as tf


import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from sklearn.model_selection import train_test_split
from Utils.Helper import rescale, create_batch_generator
from Utils.Misc import read_binary


def main():
    print("Numpy Version: %s" % np.__version__)
    print("Keras Version: %s" % keras.__version__)

    test_size = .25
    learning_rate = 0.001
    ## Load Data
    target = {
        "broadleaf" : "BROADLEAF_SP.tif_proj.bin",
        "ccut" : "CCUTBL_SP.tif_proj.bin",
        "conifer" : "CONIFER_SP.tif_proj.bin",
        "exposed" : "EXPOSED_SP.tif_proj.bin",
        "herb" : "HERB_GRAS_SP.tif_proj.bin",
        "mixed" : "MIXED_SP.tif_proj.bin",
        "river" : "RiversSP.tif_proj.bin",
        #"road" : "RoadsSP.tif_proj.bin",
        "shrub" : "SHRUB_SP.tif_proj.bin",
        #"vri" : "vri_s3_objid2.tif_proj.bin",
        "water" : "WATERSP.tif_proj.bin",
    }

    xs, xl, xb, X = read_binary('data/data_img/output4_selectS2.bin')
    xs = int(xs)
    xl = int(xl)
    xb = int(xb)
    X = X.reshape(xl*xs, xb)

    # build one hot
    one_hot = np.zeros((xs * xl, len(target)))
    for idx, key in enumerate(target.keys()):
        s,l,b,one_hot[:,idx] = read_binary("data/data_bcgw/%s" % target[key])


    ## Preprocess
    X_train, X_test, y_train, y_test = train_test_split(X, one_hot, test_size=test_size)
    mean_vals = np.mean(X_train, axis=0)
    std_vals = np.std(X_train)
    X_train_centered = (X_train - mean_vals) / std_vals
    X_test_centered = (X_test - mean_vals) / std_vals

    del X_train, X_test
    print(X_train_centered.shape, y_train.shape)
    print(X_test_centered.shape, y_test.shape)

    ## Model

    n_features = X_train_centered.shape[1]
    n_classes = len(target)
    rand_seed = 123 # reproducability

    np.random.seed(rand_seed)

    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(rand_seed)
        tf_x = tf.placeholder(dtype=tf.float32,
                              shape=(None, n_features),
                              name='tf_x')

        tf_y = tf.placeholder(dtype=tf.int8,
                              shape=None,
                              name='tf_y')
        y_onehot = tf.one_hot(indices=tf_y,
                              units=50,
                              activation=tf.tanh,
                              name='layer1')
        h2 = tf.layers.dense(inputs=h1,
                             units=50,
                             activation=tf.tanh,
                             name='layer2')
        logits = tf.layers.dense(inputs=h2,
                                 units=n_features,
                                 activation=None,
                                 name='layer3')
        predictions = {
            'classes' : tf.argmax(logits, axis=1,
                                  name='predicted_classes'),
            'probabilities' : tf.nn.softmax(logits,
                                            name='softmax_tensor')
        }

    with g.as_default():
        cost = tf.losses.softmax_cross_entropy(
            onehot_labels=y_onehot, logits=logits
            )

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate
        )
        train_op = optimizer.minimize(loss=cost)

        init_op = tf.global_variables_initializer()




def visVRI():
    samples, lines, bands, y = read_binary('data/data_bcgw/vri_s3_objid2.tif_proj.bin')
    s = int(samples)
    l = int(lines)
    b = int(bands)
    y = y.reshape(l, s)
    plt.imshow(y)
    plt.show()


def visRGB():
    samples, lines, bands, X = read_binary('data/data_img/output4_selectS2.bin')
    s = int(samples)
    l = int(lines)
    b = int(bands)
    data_r = X.reshape(b, s * l)
    rgb = np.zeros((l, s, 3))

    for i in range(0, 3):
        rgb[:, :, i] = data_r[4 - i, :].reshape((l, s))
    for i in range(0,3):
        rgb[:, :, i] = rescale(rgb[:, :, i])
    del X
    plt.imshow(rgb)
    plt.show()


if __name__ == "__main__":

   main()
### SHOW RGB Image
# # data has to switch around for matplotlib
# data_r = data.reshape(b, s * l)
# rgb = np.zeros((l, s, 3))

# for i in range(0, 3):
#     rgb[:, :, i] = data_r[4 - i, :].reshape((l, s))
# for i in range(0,3):
#     rgb[:, :, i] = rescale(rgb[:, :, i])
# del data_r
# plt.imshow(rgb)
# plt.savefig('outs/New_Image_Scaled')
# plt.show()


# rgb = (rgb - mean) / std
# rgb = rgb.reshape(3, s, l)
# print(rgb.shape)
# plt.imshow(rgb)
# plt.show()