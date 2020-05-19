import cv2
from keras import utils
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
import time
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import read_binary
from Utils.Helper import rescale, create_batch_generator

# TO VIEW MODEL PERFORMANCE IN BROWSER RUN
#
# FROM ROOT DIR
# globals
root_path = "data/full/"
reference_data_root = f"{root_path}data_bcgw/"
raw_data_root = f"{root_path}data_img/"

target = {
        "conifer" : "CONIFER.bin",
        "ccut" : "CCUTBL.bin",
        "water": "WATER.bin",
        "broadleaf" : "BROADLEAF.bin",
        "shrub" : "SHRUB.bin",
        "mixed" : "MIXED.bin",
        "herb" : "HERB.bin",
        "exposed" : "EXPOSED.bin",
        "river" : "Rivers.bin",
        # "road" : "ROADS.bin",
        # "vri" : "vri_s3_objid2.tif_proj.bin",
    }

def create_model(input_dim, output_dim, test=False):
    if test:
        return tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_dim),
    tf.keras.layers.Dense(8, activation='relu'),#, kernel_initializer=regularizers.l2(0.001)),
    tf.keras.layers.Dense(output_dim, activation='softmax')
  ])
    else:
        return tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_dim),
    tf.keras.layers.Dense(512, activation='relu'),#, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1024, activation='relu'),# kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.001)),#, kernel_initializer=regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1024, activation='relu'),# kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2048, activation='relu'),# kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2048, activation='relu'),# kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(4096, activation='relu'),# kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(4096, activation='relu'),# kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),




    # tf.keras.layers.Dense(4096, activation='relu', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.001)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(8192, activation='relu', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.001)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(16384, activation='relu', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.001)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(output_dim, activation='softmax')
  ])

def main():

    test = False

    print("Numpy Version: %s" % np.__version__)
    print("Keras Version: %s" % keras.__version__)

    """----------------------------------------------------------------------------------------------------------------------------
    * Configuration
    """
    # Config
    test_size = .20

    if test:
        epochs = 1
        batch_size = 8192
    else:
        epochs = 10
        batch_size = 4096

    lr = 0.003333

    target = {
        "conifer" : "CONIFER.bin",
        "ccut" : "CCUTBL.bin",
        "water": "WATER.bin",
        "broadleaf" : "BROADLEAF.bin",
        "shrub" : "SHRUB.bin",
        "mixed" : "MIXED.bin",
        "exposed" : "EXPOSED.bin",
        "herb" : "HERB.bin",
        "river" : "Rivers.bin",
        # "road" : "ROADS.bin",
        # "vri" : "vri_s3_objid2.tif_proj.bin",
    }

    labels = ['unlabelled']

    for l in target.keys():
        labels.append(l)

    METRICS = [
      # keras.metrics.CategoricalAccuracy(name='cat_acc'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.SensitivityAtSpecificity(.95)
    ]


    """----------------------------------------------------------------------------------------------------------------------------
    * Initial Load
    """




    n_classes = len(target) + 1
    rand_seed = 123 # reproducability
    np.random.seed(rand_seed)
    tf.random.set_seed(rand_seed)

    """----------------------------------------------------------------------------------------------------------------------------
    * Output Directories
    """
    if not os.path.exists('outs'):
        print('creating outs directory in root')
        os.mkdir('outs')
    if not os.path.exists('outs/NaiveDeepNet/'):
        print('creating outs/NaiveDeepNet in root')
        os.mkdir('outs/NaiveDeepNet/')
    outdir = get_run_logdir('outs/NaiveDeepNet/')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    model_dir = f'{outdir}/models'
    os.mkdir(model_dir)


    """----------------------------------------------------------------------------------------------------------------------------
    * Callbacks
    """
    root_logdir = os.path.join(os.curdir, "logs")
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    model_filepath = os.path.join(model_dir, "weights-improvement-{epoch:02d}-{val_loss:04d}.hdf5")
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    modelsave_cb = keras.callbacks.ModelCheckpoint(model_filepath, monitor='cat_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=10,
                                            verbose=0,
                                            mode='auto',
                                            baseline=None,
                                            restore_best_weights=True)
    callbacks = [tensorboard_cb, es_cb]


    """----------------------------------------------------------------------------------------------------------------------------
    * Model
    """
    model = create_model(11, len(target)+1)

    optimizer = keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)


    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=METRICS)


    """-----------------------------------------------------------
    * Cross Val
    """
    for idx in range(5):

        # Create generators
        params = {'dim': (32,11),
          'batch_size': 32,
          'n_classes': 10,
          'n_channels': 11,
          'shuffle': True
        }

        partition = {
        'train' : [f'oversampled/{i}' for i in range(5) if i != idx],
        'validation' : [idx]
        }
        print(partition)

        training_generator = DataGenerator(partition['train'], labels, **params)
        validation_generator = DataGenerator(partition['validation'], labels, **params)
        for t in training_generator:
            print(t)


        n_features = 11

        """----------------------------------------------------------------------------------------------------------------------------
        * Training
        """
        start_fit = time.time()

        history = model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            epochs=3,
                            verbose=1,
                            shuffle=True,
                            use_multiprocessing=True,
                            workers=-1,

                            callbacks=callbacks)

        end_fit = time.time()

    #         """----------------------------------------------------------------------------------------------------------------------------
    #         * Training
    #         """
    #         start_fit = time.time()

    #         history = model.fit(X_train, y_train,
    #                             batch_size=batch_size,
    #                             epochs=1,
    #                             class_weight=class_weight,
    #                             verbose=1,
    #                             validation_split=0.0,
    #                             validation_data=(X_val, y_val),
    #                             shuffle=True,
    #                             use_multiprocessing=True,
    #                             workers=-1,
    #                             callbacks=callbacks)

    #         end_fit = time.time()


    # # read in the larger image
    # # predict over the image

    # """----------------------------------------------------------------------------------------------------------------------------
    # * Evaluation
    # """
    # print("Predicting X_test")
    # # test set prediction
    # start_predict = time.time()
    # test_pred = model.predict(X_test, batch_size=1024)
    # end_predict = time.time()

    # # train set prediction
    # print("Predicting X_train")
    # train_pred = model.predict(X_train,  batch_size=1024)

    # # full prediction
    # print("Predicting X")
    # pred = model.predict(X.reshape(X.shape[1] * X.shape[2], X.shape[3]),  batch_size=1024)

    # # Convert to one dimensional arrays of confidence and class
    # test_pred_confidence = np.amax(test_pred, axis=1)
    # test_pred_class = np.argmax(test_pred, axis=1)
    # test_pred_zipped = zip(np.argmax(y_test,axis=1), # this returns the sparse array of labels
    #                        test_pred_class,
    #                        test_pred_confidence)

    # train_pred_confidence = np.amax(train_pred, axis=1)
    # train_pred_class = np.argmax(train_pred, axis=1)
    # train_pred_zipped = zip(np.argmax(y_train,axis=1), # this returns the sparse array of labels
    #                        train_pred_class,
    #                        train_pred_confidence)

    # pred_confidence = np.amax(pred, axis=1)
    # pred_class = np.argmax(pred, axis=1)
    # pred_zipped = zip (onehot,
    #                    pred_class,
    #                    pred_confidence)

    # # time to predict the test set
    # predict_time = round(end_predict - start_predict, 2)
    # fit_time = round(end_fit - start_fit, 2)

    # print("Creating confusion matrices")
    # # Confusion matricces for the test and train datasets
    # confmatTest = confusion_matrix(
    #     y_true=np.argmax(y_test, axis=1), y_pred=test_pred_class)
    # confmatTrain = confusion_matrix(
    #     y_true=np.argmax(y_train, axis=1), y_pred=train_pred_class)

    # # visualization = build_vis(pred_class, onehot, (int(xl), int(xs), 3))

    # # fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=False)

    # # ex = Rectangle((0, 0), 0, 0, fc="w", fill=False,
    # #                 edgecolor='none', linewidth=0)
    # # fig.legend([ex, ex, ex, ex, ex, ex, ex, ex, ex, ex, ex],
    # #             ("Target: %s" % "Water",
    # #             # "Test Acc.: %s" % round(test_score, 3),
    # #             # "Train Acc.: %s" % round(train_score, 3),
    # #             "Test Size: %s" % test_size,
    # #             "Train: %ss" % fit_time,
    # #             "Predict: %ss" % predict_time),

    # #             loc='lower right',
    # #             ncol=3)

    # """----------------------------------------------------------------------------------------------------------------------------
    # * Output
    # """

    # # Create the directory for our output
    # if not os.path.exists('outs'):
    #     print('creating outs directory in root')
    #     os.mkdir('outs')
    # if not os.path.exists('outs/NaiveDeepNet/'):
    #     print('creating outs/NaiveDeepNet in root')
    #     os.mkdir('outs/NaiveDeepNet/')
    # outdir = get_run_logdir('outs/NaiveDeepNet/')
    # if not os.path.exists(outdir):
    #     os.mkdir(outdir)

    # print(f"Writing results to {outdir}")
    # reference_file = f'{outdir}/reference'
    # prediction_file = f'{outdir}/prediction'
    # train_cmat_file = f'{outdir}/train_confmat'
    # test_cmat_file = f'{outdir}/test_confmat'
    # history_file = f'{outdir}/history'
    # loss_file = f'{outdir}/loss'
    # train_conf_file = f'{outdir}/train_prediction'
    # val_conf_file = f'{outdir}/val_prediction'
    # summary_file = f'{outdir}/summary.txt'
    # cat_acc_file = f'{outdir}/categorical_acc'


    # plt.title('Reference')
    # plt.imshow(onehot.reshape(xl, xs), cmap='gray')
    # plt.savefig(reference_file)
    # print(f'+w\n{reference_file}')


    # plt.title('Prediction')
    # plt.imshow(pred_class.reshape(xl, xs), cmap='gray')
    # plt.savefig(prediction_file)
    # print(f'+w\n{prediction_file}')


    # plt.title("Test Confusion Matrix")
    # plt.matshow(confmatTest, cmap=plt.cm.Blues, alpha=0.5)
    # plt.gcf().subplots_adjust(left=.5)
    # for i in range(confmatTest.shape[0]):
    #     for j in range(confmatTest.shape[1]):
    #         plt.text(x=j, y=i,
    #                 s=round(confmatTest[i,j],3), fontsize=6, horizontalalignment='center')
    # labels = ['unlabeled']
    # for label in target.keys():
    #     labels.append(label)
    # plt.xticks(np.arange(10), labels=labels)
    # plt.yticks(np.arange(10), labels=labels)
    # plt.tick_params('both', labelsize=8, labelrotation=45)
    # plt.xlabel('predicted label')
    # plt.ylabel('reference label', rotation=90)
    # plt.savefig(test_cmat_file)
    # print(f'+w\n{test_cmat_file}')
    # plt.clf()


    # plt.title("Train Confusion Matrix")
    # plt.matshow(confmatTrain, cmap=plt.cm.Blues, alpha=0.5)
    # for i in range(confmatTrain.shape[0]):
    #     for j in range(confmatTrain.shape[1]):
    #         plt.text(x=j, y=i,
    #                 s=round(confmatTrain[i,j],3), fontsize=6, horizontalalignment='center')
    # plt.xticks(np.arange(10), labels=labels)
    # plt.yticks(np.arange(10), labels=labels)
    # plt.tick_params('both', labelsize=8, labelrotation=45)
    # plt.xlabel('predicted label')
    # plt.ylabel('reference label', rotation=90)
    # plt.margins(0.2)
    # plt.savefig(train_cmat_file)
    # print(f'+w\n{train_cmat_file}')
    # plt.clf()


    # plt.title('Model loss')
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Val'], loc='upper left')
    # plt.savefig(loss_file)
    # print(f'+w\n{loss_file}')
    # plt.clf()


    # plt.title('Train Prediciton Counts')
    # plt.plot(history.history['tp'])
    # plt.plot(history.history['fp'])
    # plt.plot(history.history['fn'])
    # plt.plot(history.history['fp'])
    # plt.ylabel('No. of Pixels', rotation=90)
    # plt.xlabel('Epoch')
    # plt.legend(['TP', 'FP', 'FN', 'FP'], loc="upper left")
    # plt.savefig(train_conf_file)
    # print(f'+w\n{train_conf_file}')
    # plt.clf()


    # plt.title('Validation Prediciton Counts')
    # plt.plot(history.history['val_tp'])
    # plt.plot(history.history['val_fp'])
    # plt.plot(history.history['val_fn'])
    # plt.plot(history.history['val_fp'])
    # plt.ylabel('No. of Pixels', rotation=90)
    # plt.xlabel('Epoch')
    # plt.legend(['TP', 'FP', 'FN', 'FP'], loc="upper left")
    # plt.savefig(val_conf_file)
    # print(f'+w\n{val_conf_file}')
    # plt.clf()


    # plt.ylabel('value', rotation=90)
    # plt.xlabel("Epoch")
    # plt.legend(['Train', 'Validation'])
    # plt.savefig(cat_acc_file)
    # print(f'+w\n{cat_acc_file}')
    # plt.clf()

    # with open(summary_file, 'w') as f:
    #     model.summary(print_fn=lambda x: f.write(x + '\n'))


def checkit(passed):
    print()
    print("CHECKIT")
    print(type(passed))
    print(passed.shape)
    print()

def get_file_path(path):
    dirs = path.split('/')
    path = ""

    for dir in dirs:
        print(dir)
        if not os.path.exists(dir):
            path = os.path.join(path, os.mkdir(dir))
        else:
            path += f"/{dir}"
        print(path)

def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run__%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


class DataGenerator(keras.utils.Sequence):


    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 1, 11), n_channels=11, n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indices]

        X, y = self.__data_generation(list_IDs_temp)

        yield X, y


    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generate data containing batch_size samples'
        # Init
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.load(f'data/prepared/train/cropped/{i}-data.npy')
            y[i] = np.load(f'data/prepared/train/cropped/{i}-label.npy')
            X[i,] = X[i,].reshape(X[i,].shap[1] * X[i,].shape[2], X[i,].shape[0])
            y[i] = y[i].ravel()

        return StandardScaler().fit_transform(X), keras.utils.to_categorical(y, num_class=self.n_classes)


if __name__ == "__main__":

    main()
