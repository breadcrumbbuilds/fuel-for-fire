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

    ## Config
    test_size = .25
    learning_rate = 0.000333
    batch_size = 16384
    epochs = 10

    ## Load Data
    target = {
        # "broadleaf" : "BROADLEAF_SP.tif_proj.bin",
        # "ccut" : "CCUTBL_SP.tif_proj.bin",
        # "conifer" : "CONIFER_SP.tif_proj.bin",
        # "exposed" : "EXPOSED_SP.tif_proj.bin",
        # "herb" : "HERB_GRAS_SP.tif_proj.bin",
        # "mixed" : "MIXED_SP.tif_proj.bin",
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
        s,l,b,tmp = read_binary("data/data_bcgw/%s" % target[key])
        one_hot[:,idx] = np.logical_or(tmp,one_hot[:,idx])

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

    tf.random.set_seed(rand_seed)

    print('\nFirst 3 labels (one-hot):\n',y_train[:3])

    model = keras.models.Sequential()

    model.add(
        keras.layers.Dense(
            units=50,
            input_dim=X_train_centered.shape[1],
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh'
        ))

    model.add(
        keras.layers.Dense(
            units=100,
            input_dim=50,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh'
        ))

    model.add(
        keras.layers.Dense(
            units=200,
            input_dim=100,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh'
        ))

    # model.add(
    #     keras.layers.Dense(
    #         units=400,
    #         input_dim=200,
    #         kernel_initializer='glorot_uniform',
    #         bias_initializer='zeros',
    #         activation='tanh'
    #     ))

    # model.add(
    #     keras.layers.Dense(
    #         units=800,
    #         input_dim=400,
    #         kernel_initializer='glorot_uniform',
    #         bias_initializer='zeros',
    #         activation='tanh'
    #     ))

    # model.add(
    #     keras.layers.Dense(
    #         units=400,
    #         input_dim=200,
    #         kernel_initializer='glorot_uniform',
    #         bias_initializer='zeros',
    #         activation='tanh'
    #     ))

    model.add(
        keras.layers.Dense(
            units=100,
            input_dim=200,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh'
        ))

    model.add(
        keras.layers.Dense(
            units=50,
            input_dim=100,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh'
        ))

    model.add(
        keras.layers.Dense(
            units=25,
            input_dim=50,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh'
        ))

    model.add(
        keras.layers.Dense(
            units=y_train.shape[1],
            input_dim=25,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='relu'
        ))

    sgd_optimizer = keras.optimizers.SGD(
        lr=learning_rate, decay=1e-7, momentum=.9
    )

    model.compile(optimizer=sgd_optimizer,
                  loss='categorical_crossentropy')

    history = model.fit(X_train_centered, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)

    y_train_pred = model.predict_classes(X_train_centered, verbose=1)

    correct_preds = np.sum(y_train == y_train_pred, axis=0)
    train_acc = correct_preds / y_train.shape[0]

    print('Training Accuracy: %.2f%%' % (train_acc * 100))

    y_test_pred = model.predict_classes(X_test_centered,
                                        verbose=1)
    correct_preds = np.sum(y_test == y_test_pred, axis=0)
    test_acc = correct_preds / y_test.shape[0]
    print('Test accuracy: %.2f%%' % (test_acc * 100))





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