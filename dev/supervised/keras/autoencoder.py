import tensorflow.keras as keras
import tensorflow as tf
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


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
    test_size = .3
    learning_rate = 0.0000333
    batch_size = 256
    epochs = 500


    ## Load Data
    target = {
        "broadleaf" : "BROADLEAF_SP.tif_proj.bin",
        "ccut" : "CCUTBL_SP.tif_proj.bin",
        "conifer" : "CONIFER_SP.tif_proj.bin",
        "exposed" : "EXPOSED_SP.tif_proj.bin",
        "herb" : "HERB_GRAS_SP.tif_proj.bin",
        "mixed" : "MIXED_SP.tif_proj.bin",
        "river" : "RiversSP.tif_proj.bin",
        # "road" : "RoadsSP.tif_proj.bin",
        "shrub" : "SHRUB_SP.tif_proj.bin",
        # "vri" : "vri_s3_objid2.tif_proj.bin",
        "water" : "WATERSP.tif_proj.bin",
    }

    xs, xl, xb, X = read_binary('data/data_img/output4_selectS2.bin')
    xs = int(xs)
    xl = int(xl)
    xb = int(xb)
    X = X.reshape(xl*xs, xb)


    onehot = encode_one_hot(target, xs, xl, array=True)
    test = np.zeros((xl*xs, len(target)))



    # ## Preprocess
    X_train, X_test, y_train, y_test = train_test_split(X, onehot, test_size=test_size)
    X_train_centered = np.zeros(X_train.shape)
    X_test_centered = np.zeros(X_test.shape)

    # Need to normalize each band independently
    for idx in range(X_train.shape[1]) :
        print(idx)
        train_mean_vals = np.mean(X_train[:,idx], axis=0)
        test_mean_vals = np.mean(X_test[:,idx], axis=0)
        train_std_vals = np.std(X_train[:,idx])
        test_std_vals = np.std(X_test[:,idx])

        X_train_centered[:,idx] = (X_train[:,idx] - train_mean_vals) / train_std_vals
        X_test_centered[:,idx] = (X_test[:,idx] - test_mean_vals) / test_std_vals

    print(X_train_centered)
    print(X_test_centered)

    # mean_vals = np.mean(X_train, axis=0)
    # std_vals = np.std(X_train)
    # X_train_centered = (X_train - mean_vals) / std_vals
    # X_test_centered = (X_test - mean_vals) / std_vals

    del X_train, X_test
    print(X_train_centered.shape, y_train.shape)
    print(X_test_centered.shape, y_test.shape)



    # ## Model

    n_features = X_train_centered.shape[1]
    n_classes = len(target)
    rand_seed = 123 # reproducability

    np.random.seed(rand_seed)

    tf.random.set_seed(rand_seed)

    print('\nFirst 3 labels (one-hot):\n',y_train[:3])

    model = keras.models.Sequential([
        keras.layers.Dense(
            units=11,
            input_dim=X_train_centered.shape[1],
            kernel_initializer='glorot_uniform',
            bias_initializer='normal',
            activation='relu'),
       ## MIDDLE LAYER
       keras.layers.Dense(
            units=11,
            input_dim=11,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='relu'),
       keras.layers.Dense(
            units=8,
            input_dim=11,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='relu'),
       keras.layers.Dense(
            units=5,
            input_dim=8,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='relu'),
       # Embedded layer
       keras.layers.Dense(
            units=3,
            input_dim=3,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='relu'),
       keras.layers.Dense(
            units=5,
            input_dim=3,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='relu'),
        keras.layers.Dense(
            units=8,
            input_dim=5,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='relu'),
        keras.layers.Dense(
            units=11,
            input_dim=8,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='relu'),
        keras.layers.Dense(
            units=X_train_centered.shape[1],
            input_dim=11,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='linear' # identity
        )])

    sgd_optimizer = keras.optimizers.SGD(
        lr=learning_rate, decay=1e-6, momentum=.95
    )

    model.compile(optimizer=sgd_optimizer,
                  loss='mean_squared_error',
                metrics=['accuracy', 'mse', 'mae' ])

    # to visualize training in the browser'
    root_logdir = os.path.join(os.curdir, "logs")
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    filepath= os.path.join(os.curdir, "outs/models/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5")
    modelsave_cb = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callbacks = [tensorboard_cb, modelsave_cb]

    history = model.fit(X_train_centered, X_train_centered,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.2,
                        shuffle=True,
                        use_multiprocessing=True,
                        workers=-1,
                        callbacks=callbacks)

    # now run the training data through the network
    # pull out all of the three d representations WITH the class
    # associated and we can plot this!

    # run the test data through
    # For each sample in the test set, compare that samples
    # embedded layer (a 3d representation of the original)
    # to the closest sample of each class in the training set
    # sum these values, then divide the individual classes distance
    # by the overall distance observed to get a 'confidence' interval
    # of the test sample

    print(history.history)
    # train_results = model.predict(X_train_centered)
    # print(train_results)
    plt.title("val accuracy")
    plt.plot(history.history['val_accuracy'])
    plt.savefig(get_run_logdir(os.path.join(os.curdir, "outs/autoencoder")))

    # print(history)
    # # ## Testing

    # y_train_pred = model.predict_classes(X_train_centered,
    #                                      verbose=1)
    # correct_preds = np.sum(y_train == y_train_pred, axis=0)
    # train_acc = correct_preds / y_train.shape[0]
    # print('Training Accuracy: %.2f%%' % (train_acc * 100))


    # y_test_pred = model.predict_classes(X_test_centered,
    #                                     verbose=1)
    # correct_preds = np.sum(y_test == y_test_pred, axis=0)
    # test_acc = correct_preds / y_test.shape[0]
    # print('Test accuracy: %.2f%%' % (test_acc * 100))



def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run__%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def one_hot_sanity_check(target, xs, xl):
    oh = encode_one_hot(target, xs, xl)
    for idx, val in enumerate(oh):
        print(val[0])
        plt.title(val[0])
        plt.imshow(val[1].reshape(xl,xs), cmap='gray')
        plt.show()


def encode_one_hot(target, xs, xl, array=False):

    if array:
        result = np.zeros((xs*xl, len(target)))
    else:
        result = list()

    for idx, key in enumerate(target.keys()):
        ones = np.ones((xl * xs))
        s,l,b,tmp = read_binary("data/data_bcgw/%s" % target[key])

        # same shape as the raw image
        assert int(s) == int(xs)
        assert int(l) == int(xl)

        # last index is the targets false value
        vals = np.sort(np.unique(tmp))

        # create an array populate with the false value
        t = ones * vals[len(vals) - 1]

        if key == 'water':
            arr = np.not_equal(tmp,t)
        else:
            arr = np.logical_and(tmp,t)

        # How did the caller ask for the data
        if array:
            result[:,idx] = arr
        else:
            result.append((key, arr))

    return result


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