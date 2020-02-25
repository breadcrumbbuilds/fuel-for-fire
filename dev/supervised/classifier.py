import os
import sys
import math
import copy
import time
import pickle
import datetime
from misc import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rasterio
from rasterio import plot
from rasterio.plot import show
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
'''
partial setup instructions for ubuntu 18:
  python3 -m pip install pandas
  python3 -m pip install rasterio
  python3 -m pip install --upgrade pytest
'''

global data_frame
data_frame = None  # global variable for data frame (need for parallelism)
# assume data dimensions are the same for every input file
global lines
global samples
lines, samples = None, None  # read these from header file


def get_data(fp):
    """Assumes the filepath provided contains both
        .hdr and .bin files within the fp provided in the function
        call.

        returns a list of file paths to .bin files
    """
    rasterBin = []
    for f in fp:
        if not exist(f):
            err("couldn't find path:\n\t" + f)

        for root, dirs, files in os.walk(f, topdown=False):
            for file in files:
                if ".hdr" not in file:
                    rasterBin.append(os.path.join(f, file))
    return rasterBin


def get_date_string():
    day = datetime.datetime.now().day
    hour = datetime.datetime.now().hour
    min = datetime.datetime.now().minute

    return str(day) + "_" + str(hour) + "_" + str(min) + "/"


def populate_data_frame(rasterBin, showplots=False):
    global samples, lines
    """Receives a list of file paths to .bin files.
        A ground truth raster (predefined) is read,
        band by band, and stored in the data_frame.
        Additional 'labeled' truth data images are
        read and saved to the dataframe. These values
        are also decoded to binary interpretations, and
        the binary interpretation is stored in its own
        column.

        Showplots argument added to visualize the raw data
        and the truth data.

        Returns a populated pandas dataframe.
    """
    data_frame = pd.DataFrame()
    print("putting stuff in data_Frame:")
    for raster in rasterBin:
        print("raster", raster)
        name = raster.split(os.path.sep)[-1].split(".")[0]

        if 'bool' in name:
            err('unexpected: bool substr of filename')

        hdr = '.'.join(raster.split(".")[:-1]) + '.hdr'
        if not exist(hdr):
            err("couldn't find hdr file")

        # pragmatic programming: make sure all the dimensions match
        samples2, lines2, bands2 = read_hdr(hdr)
        samples2, lines2, bands2 = int(samples2), int(lines2), int(bands2)
        if not samples or not lines:
            samples, lines = samples2, lines2
        else:
            if samples2 != samples or lines2 != lines:
                err("unexpected dimensions: " +
                    str(samples2) + "x" + str(lines2) + ', expected: ' +
                    str(samples) + 'x' + str(lines))

        if "S2A" in name:
            dataset = rasterio.open(raster)
            for idx in dataset.indexes:
                print(name + "_" + str(idx))
                data_frame[name + "_" + str(idx)] = dataset.read(idx).ravel()

        elif "L8" in name:
            dataset = rasterio.open(raster)
            for idx in dataset.indexes:
                print(name + "_" + str(idx))
                data_frame[name + "_" + str(idx)] = dataset.read(idx).ravel()

        elif "WATERSP" in name:
            water = rasterio.open(raster).read(1)
            print('water_val')
            data_frame['water_val'] = water.ravel()
            print('water_bool')
            data_frame['water_bool'] = data_frame['water_val'] != 128

        elif "RiversSP" in name:
            river = rasterio.open(raster).read(1)
            print('river_val')
            data_frame['river_val'] = river.ravel()
            print('river_bool')
            data_frame['river_bool'] = data_frame['river_val'] == 1.0

        elif "BROADLEAF_SP" in name:
            broadleaf = rasterio.open(raster).read(1)
            print('broadleaf_val')
            data_frame['broadleaf_val'] = broadleaf.ravel()
            print('broadleaf_bool')
            data_frame['broadleaf_bool'] = data_frame['broadleaf_val'] == 1.0

        elif "SHRUB_SP" in name:
            shrub = rasterio.open(raster).read(1)
            print('shrub_val')
            data_frame['shrub_val'] = shrub.ravel()
            print('shrub_bool')
            data_frame['shrub_bool'] = data_frame['shrub_val'] != 0.0

        elif "MIXED_SP" in name:
            mixed = rasterio.open(raster).read(1)
            print('mixed_val')
            data_frame['mixed_val'] = mixed.ravel()
            print('mixed_bool')
            data_frame['mixed_bool'] = data_frame['mixed_val'] != 0.0

        elif "CONIFER_SP" in name:
            conifer = rasterio.open(raster).read(1)
            print('conifer_val')
            data_frame['conifer_val'] = conifer.ravel()
            print('conifer_bool')
            data_frame['conifer_bool'] = data_frame['conifer_val'] != 0.0

        elif "HERB_GRAS_SP" in name:
            herb = rasterio.open(raster).read(1)
            print('herb_val')
            data_frame['herb_val'] = herb.ravel()
            print('herb_bool')
            data_frame['herb_bool'] = data_frame['herb_val'] != 0.0

        elif "CCUTBL_SP" in name:
            clearcut = rasterio.open(raster).read(1)
            print('clearcut_val')
            data_frame['clearcut_val'] = clearcut.ravel()
            print('clearcut_bool')
            data_frame['clearcut_bool'] = data_frame['clearcut_val'] != 0.0

        elif "EXPOSED_SP" in name:
            exposed = rasterio.open(raster).read(1)
            print('exposed_val')
            data_frame['exposed_val'] = exposed.ravel()
            print('exposed_bool')
            data_frame['exposed_bool'] = data_frame['exposed_val'] != 0.0

        else:
            continue
            layer = rasterio.open(raster)
            if len(layer.indexes) > 1:
                err("expected one band only")
            layer = rasterio.open(raster).read(1)

            # pragmatic programming: make sure "bool" isn't in filename already
            if 'bool' in name:
                err('unexpected "bool" in filename')

            # pragmatic programming: make sure no name collision
            if name in data_frame:
                err(name + " already in data_frame")

            d = layer.ravel()
            counts = hist(d)

            enough_points = min(counts.values()) > 1000

            # only add the layer to the stack, if it has at least two values
            if len(counts) > 1 and enough_points:

                print(len(counts), name, counts)
                data_frame[name] = d

                print(name + '_bool')
                data_frame[name + '_bool'] = data_frame[name] != 0.

    if showplots:
        show_original_image(data_frame, 'l')
        show_original_image(data_frame, 's')
        #show_truth_data_subplot(data_frame)

    print("data_frame.columns", data_frame.columns)
    return data_frame


def create_class_dictionary(data_frame):
    """dictionary for making simple requests for data from
        the dataframe. Key corresponds to a class name
        ie, water. The value associated with any key is
        simply the key with '_bool' appended to the end
        of the string.

        returns a dictionary

    """
    column_values = [
        col_name for col_name in data_frame.columns if "bool" in col_name]
    column_keys = []
    for key in column_values:
        column_keys.append(key.replace("_bool", ""))
    dictionary = dict(zip(column_keys, column_values))

    return dictionary

def show_original_image(df, data):
    """builds an array of the RGB values of the
        truth image, scaling the original values
        between 0 - 1 (matplotlib req.) and
        displays that image in a plot.

Landsat 8:
Spectral Band 	Wavelength 	Resolution 	Solar Irradiance
Band 1 - Coastal / Aerosol 	0.433 – 0.453 μm 	30 m 	2031 W/(m2μm)
Band 2 - Blue 	0.450 – 0.515 μm 	30 m 	1925 W/(m2μm)
Band 3 - Green 	0.525 – 0.600 μm 	30 m 	1826 W/(m2μm)
Band 4 - Red 	0.630 – 0.680 μm 	30 m 	1574 W/(m2μm)
Band 5 - Near Infrared 	0.845 – 0.885 μm 	30 m 	955 W/(m2μm)
Band 6 - Short Wavelength Infrared 	1.560 – 1.660 μm 	30 m 	242 W/(m2μm)
Band 7 - Short Wavelength Infrared 	2.100 – 2.300 μm 	30 m 	82.5 W/(m2μm)
Band 8 - Panchromatic 	0.500 – 0.680 μm 	15 m 	1739 W/(m2μm)
Band 9 - Cirrus 	1.360 – 1.390 μm 	30 m 	361 W/(m2μm)

Sentinel-2 bands 	Sentinel-2A 	Sentinel-2B
Central wavelength (nm) 	Bandwidth (nm) 	Central wavelength (nm) 	Bandwidth (nm) 	Spatial resolution (m)
Band 1 – Coastal aerosol 	442.7 	21 	442.2 	21 	60
Band 2 – Blue 	492.4 	66 	492.1 	66 	10
Band 3 – Green 	559.8 	36 	559.0 	36 	10
Band 4 – Red 	664.6 	31 	664.9 	31 	10
Band 5 – Vegetation red edge 	704.1 	15 	703.8 	16 	20
Band 6 – Vegetation red edge 	740.5 	15 	739.1 	15 	20
Band 7 – Vegetation red edge 	782.8 	20 	779.7 	20 	20
Band 8 – NIR 	832.8 	106 	832.9 	106 	10
Band 8A – Narrow NIR 	864.7 	21 	864.0 	22 	20
Band 9 – Water vapour 	945.1 	20 	943.2 	21 	60
Band 10 – SWIR – Cirrus 	1373.5 	31 	1376.9 	30 	60
Band 11 – SWIR 	1613.7 	91 	1610.4 	94 	20
Band 12 – SWIR 	2202.4 	175 	2185.7 	185 	20
    """

    arr = np.zeros((lines, samples, 3))

    if data == 's':
        red = rescale(df.S2A_4.values.reshape(lines, samples))
        green = rescale(df.S2A_3.values.reshape(lines, samples))
        blue = rescale(df.S2A_2.values.reshape(lines, samples))
    elif data == 'l':
        red = rescale(df.L8_4.values.reshape(lines, samples)) # red.values.reshape(lines, samples))
        green = rescale(df.L8_3.values.reshape(lines, samples)) # green.values.reshape(lines, samples))
        blue = rescale(df.L8_2.values.reshape(lines, samples))
    else:
        print("Specify an image to show in show_original_image(df, __) " +
              "('s' for sentinel2 or 'l' for landsat8)")

    arr[:, :, 0] = red
    arr[:, :, 1] = green
    arr[:, :, 2] = blue

    # plt.gcf().set_size_inches(7, 7. * float(lines) / float(samples))
    plt.imshow(arr)
    plt.tight_layout()
    impath = "original_image" + "_" + data + ".png"
    print("+w " + impath)
    plt.savefig(impath)


def concatenate_dataframes(df1, df2):

    os_list = [df1, df2]

    X_full = pd.concat(os_list)
    return X_full


def get_x_data(df, image_type):
    """Returns a slice of the original dataframe corresponding
        to the training data
    """
    if image_type == "all":
        X = df.loc[:, :'L8_11']  # :"L8_longwave_infrared2"]
    elif image_type == "s":
        X = df.loc[:, :'S2A_12']  # :"S2_swir2"]
    elif image_type == "l":
        X = df.loc[:, 'L8_1': 'L8_11']
        # "L8_coastal_aerosol": "L8_longwave_infrared2"]
    return X


def get_training_set(data, X_true, X_false, class_, image_type='all'):
    """Concatenates true pixels and false pixels into a single dataframe to
        create a dataset.

        Returns X, a pandas dataframe, and y, a pandas series
    """
    X_full = concatenate_dataframes(X_true, X_false)

    X = get_x_data(X_full, image_type)

    y = X_full[class_]
    return X, y


def normalize_data(X):
    X_norm = StandardScaler(copy=True, with_mean=True,
                            with_std=True).fit_transform(X)
    return X_norm


def get_sample(data_frame, classes, image_type='all', undersample=True,
               normalize=True):
    """retrieves a class balanced sample of data from the dataframe.
        That is, false class, balanced with true class, with options to
        normalize the feature data, or oversample.

        returns features, X and targets, y

    """

    class_dict = create_class_dictionary(data_frame)

    if isinstance(classes, (list)):
        data_frame, class_, class_dict = create_union_column(
            data_frame, classes, class_dict)
    else:
        class_ = class_dict[classes.lower()]

    X_true = data_frame[data_frame[class_] is True]
    X_false = data_frame[data_frame[class_] is False]

    if undersample:

        if true_sample_is_smaller(X_true, X_false):
            # true is smaller, so take a subset of the false data
            X_false = data_frame[data_frame[class_]
                                 is False].sample(len(X_true))

        else:
            # false is smaller, so take a subset of the true data
            X_true = data_frame[data_frame[class_]
                                is True].sample(len(X_false))

        X, y = get_training_set(
            data_frame, X_true, X_false, class_, image_type)

        if normalize:
            return normalize_data(X), y
        else:
            return X, y

    else:  # oversampling

        if true_sample_is_smaller(X_true, X_false):
            X_true = oversample(X_true, X_false)

        else:
            X_false = oversample(X_false, X_true)

        X, y = get_training_set(
            data_frame, X_true, X_false, class_, image_type)

        if normalize:
            return normalize_data(X), y
        else:
            return X, y


def print_classifier_metrics(y_test, y_pred):
    """

    """
    debug = False

    cm = confusion_matrix(y_test, y_pred)

    if debug:
        print("Confusion Matrix")
        print("[tn fp]")
        print("[fn tp]\n")
        for arr in cm:
            print(arr)
    cm_p = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if debug:
        print("")
        for arr in cm:
            print(arr)

    return cm, cm_p


def plot_confusion_matrix_image(df, clf, true_val, image_type='all'):
    """UNDER DEVELOPMENT

    """

    # grab the test data (includes data the system was trained on)
    raw_data = get_x_data(df, image_type)

    try:
        y_pred = clf.predict(raw_data)  # predict on all the data
    except Exception:
        print("Error: It's likely that you are trying to plot using an image" +
              "type that isn't the same as the image type in the classifier")
        print("Ensure that your image_type argument in get_sample() equals " +
              "the image_type argument in plot_confusion_matrix_image()")
        sys.exit()

    y_true = df[true_val + "_bool"]  # store the true values

    arr = np.zeros([lines * samples], dtype='int')

    for x in range(len(y_pred)):  # iterate the length of the arrays
        if y_true[x]:
            if y_pred[x]:
                arr[x] = 0
                # this is true positive
            else:
                arr[x] = 5
                # This is false positive
        else:
            if y_pred[x]:
                arr[x] = 10
                # this is false negative
            else:
                arr[x] = 15
                # this is true negative
    arr = arr.reshape(lines, samples)  # 401, 410
    plt.xlabel('width (px)')
    plt.ylabel('height (px)')

    plt.clf()
    plt.gcf().set_size_inches(7, 7. * float(lines) / float(samples))
    plt.imshow(arr)
    plt.tight_layout()
    t = datetime.datetime.now().time()
    impath = true_val + str(t) + '.png'
    print('+w ' + impath)
    plt.savefig(impath)


def rescale(arr, two_percent=True):
    """

    """
    arr_min = arr.min()
    arr_max = arr.max()
    scaled = (arr - arr_min) / (arr_max - arr_min)

    if two_percent:
        # 2%-linear stretch transformation for hi-contrast vis
        values = copy.deepcopy(scaled)
        values = values.reshape(np.prod(values.shape))
        values = values.tolist()
        values.sort()
        npx = len(values)  # number of pixels
        if values[-1] < values[0]:
            print('error: failed to sort')
            sys.exit(1)
        v_min = values[int(math.floor(float(npx)*0.02))]
        v_max = values[int(math.floor(float(npx)*0.98))]
        scaled -= v_min
        rng = v_max - v_min
        if rng > 0.:
            scaled /= (v_max - v_min)

    return scaled


def train(X_train, X_test, y_train, y_test):
    debug = False
    """sklearn SGDClassifier

    """
    sgd_clf = SGDClassifier(random_state=42, verbose=False,
                            max_iter=1000, tol=1.e-3)

    np.set_printoptions(precision=3)

    if debug:
        print("\n\nBegin Fitting SGD\n")

    sgd_clf = sgd_clf.fit(X_train, y_train)
    y_pred = sgd_clf.predict(X_test)

    if debug:
        print("\n{:*^30}\n".format("Training complete"))
    score = "{:.3f}".format(sgd_clf.score(X_test, y_test))

    if debug:
        print("Test score:", score)

    cm, cm_p = print_classifier_metrics(y_test, y_pred)
    return sgd_clf, score, cm, cm_p

'''
def trainGB(X_train, X_test, y_train, y_test):
    gbrt = GradientBoostingClassifier(random_state=0)

    np.set_printoptions(precision=3)

    print("\n\nBegin Fitting GBC\n")

    gbrt = gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_test)

    score = "{:.3f}".format(gbrt.score(X_test, y_test))
    print("Test score:", score)
    cm = print_classifier_metrics(y_test, y_pred)

    return gbrt, score, cm
'''

def train_all_variations(df):
    """Rudimentary test run of a SGD classifier with all of the possible
        iterations with the given functionality. Pass a data_frame, the
        script will save a png representation of each class with the highest
        performace on the raw data. Variations include: Training on Sentinel2
        data, Landsat8 data, or both at the same time; Normalizing the data or
        not normalizing; Undersampling or oversampling; and each of the 9
        classes.
    """
    it = ['all', 'l', 's']
    us = [True, False]
    nm = [True, False]
    class_dictionary = create_class_dictionary(df)

    for class_ in class_dictionary.keys():  # for each class
        max_score = 0.0
        for s in us:  # for each type of sampling
            for n in nm:  # for normalizing or not
                for i in it:  # for each image type
                    X, y = get_sample(df, class_, image_type=i,
                                      undersample=s, normalize=n)
                    print("")
                    print("{:*^50}".format(""))
                    print("{:<40}".format("Class: " + class_))
                    print("{:<40}".format("Undersampling:" + str(s)))
                    print("{:<40}".format("Normalize: " + str(n)))
                    print("{:<40}".format("Image_Type: " + i))
                    clf, score, cm = train(X, y)

                    # Save the top scored plot
                    if max_score < float(score):
                        max_score = float(score)
                        plot_confusion_matrix_image(
                            df, clf, class_, image_type=i)


def calculate_mean_metrics(cm_list, n_folds, total_score, percentage=False):
    debug = False

    if debug:
        print("Calculating Mean Metrics")

    TN, FP, FN, TP = 0., 0., 0., 0.
    for conf_matrix in cm_list:
        TN = TN + conf_matrix[0, 0]
        FP = FP + conf_matrix[0, 1]
        FN = FN + conf_matrix[1, 0]
        TP = TP + conf_matrix[1, 1]
    if not percentage:
        n_folds = 1
    TNstr = "{:.3f}".format(TN/n_folds)
    FPstr = "{:.3f}".format(FP/n_folds)
    FNstr = "{:.3f}".format(FN/n_folds)
    TPstr = "{:.3f}".format(TP/n_folds)
    mean_score = "{:.3f}".format(total_score/n_folds)

    if debug:
        print("TN:", TNstr)
        print("FP:", FPstr)
        print("FN:", FNstr)
        print("TP:", TPstr)
        print("MeanScore:", mean_score)

    return TNstr, FPstr, FNstr, TPstr, mean_score


def build_folds(X_true, X_false, n_folds):
    # https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation
    # size
    size = len(X_true) * 2
    fold_size = int(math.floor(size/n_folds))
    half_fold_size = int(math.floor(fold_size/2))

    # Shuffle the data
    X_true.sample(frac=1)
    X_false.sample(frac=1)

    folds = list()
    for f in range(n_folds):
        X_tsample = X_true.sample(half_fold_size)
        X_fsample = X_false.sample(half_fold_size)

        # Delete the data from the original dataframes
        X_true.drop(index=X_tsample.index.values.tolist(), inplace=True)
        X_false.drop(index=X_fsample.index.values.tolist(), inplace=True)

        fold = concatenate_dataframes(X_tsample, X_fsample)
        fold.sample(frac=1)  # shuffle fold
        folds.append(fold)

    # Append the left over data from using the floor method
    fold = concatenate_dataframes(X_true, X_false)
    folds[0] = folds[0].append(fold)

    debug = False

    if debug:
        print("{:/^30}".format("Data Folded"))
        print("Number of Folds:", n_folds)
        print("Size of each fold:", str(fold_size))
        print("Leftover datapoints added to fold 0:", len(fold))

    return folds


def fold(data_frame, class_, n_folds=5, disjoint=False):
    """
    """
    if disjoint:
        data_frame = remove_intersections(data_frame, class_)

    cd = create_class_dictionary(data_frame)

    # Retrieve the original data. Does .copy() do anything after the slice?
    class_ = cd[class_]
    X_true = data_frame[data_frame[class_]].copy()
    X_false = data_frame[~ data_frame[class_]].copy()

    # Decide which class needs to take a subset
    if len(X_true) < len(X_false):
        X_false = X_false.sample(len(X_true))
    else:
        X_true = X_true.sample(len(X_false))

    return build_folds(X_true, X_false, n_folds)


def train_test_split_folded_data(fd, test_data_idx, class_, image_type="all",
                                 normalize=False):
    cd = create_class_dictionary(fd[0])

    train_is_empty = True

    for f in range(len(fd)):
        if f == test_data_idx:  # This is our test set
            X_test = get_x_data(fd[test_data_idx], image_type)
            y_test = fd[test_data_idx][cd[class_]]

        else:
            if train_is_empty:
                X_train = get_x_data(fd[test_data_idx], image_type)
                y_train = fd[test_data_idx][cd[class_]]
                train_is_empty = False
            else:
                X_train = concatenate_dataframes(get_x_data(fd[test_data_idx],
                                                            image_type),
                                                 X_train)
                y_train = concatenate_dataframes(fd[test_data_idx][cd[class_]],
                                                 y_train)
    if normalize:
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)

    return X_train, X_test, y_train, y_test


def train_folded(folded_data, class_, n_folds=5, image_type='all',
                 normalize=True):
    cm_list = list()
    cm_p_list = list()
    total_score = 0

    for idx in range(len(folded_data)):
        X_train, X_test, y_train, y_test = \
          train_test_split_folded_data(folded_data, idx, class_,
                                       image_type=image_type,
                                       normalize=normalize)
        clf, score, cm, cm_p = train(X_train, X_test, y_train, y_test)

        cm_list.append(cm)
        cm_p_list.append(cm_p)
        total_score = total_score + float(score)

    TN, FP, FN, TP, mean_score = calculate_mean_metrics(cm_list,
                                                        len(folded_data),
                                                        total_score)
    TN_p, FP_p, FN_p, TP_p, mean_score = \
        calculate_mean_metrics(cm_p_list, len(folded_data), total_score,
                               True)

    # add precision
    precision = "{:.3f}".format(float(TP) / (float(TP) + float(FP)))
    return [TN, FP, FN, TP, TN_p, FP_p, FN_p, TP_p, mean_score, precision, clf]


# execute this in parallel
def train_variation(params):
    global data_frame

    # unpack input parameters
    class_, nf, d, n, i = params

    # selection
    folded_data = fold(data_frame, class_, n_folds=nf, disjoint=d)

    # training
    TN, FP, FN, TP, TN_p, FP_p, FN_p, TP_p, mean_score, precision, clf = \
        train_folded(folded_data, class_,
                     n_folds=nf, image_type=i,
                     normalize=n)
    result = [TN, FP, FN, TP,
              TN_p, FP_p, FN_p, TP_p,
              mean_score, precision, clf, params]
    print(result)
    if len(result) != 12:
        err("result")

    return(result)


def train_all_variations_folded(df, n_f=[2, 5, 10], disjoint=[True, False],
                                norm=[True], it=[all]):
    newpath = "log/folded_classifier_" + get_date_string()

    # recursively create folder if not exist yet
    w = newpath.split(os.path.sep)
    for i in range(1, len(w)):
        p = (os.path.sep).join(w[0: i])
        if not os.path.exists(p):
            os.mkdir(p)
    cd = create_class_dictionary(df)
    f = open(newpath + "results.csv", "wb")
    f.write(("Class,N_Folds,Image_Type,Disjoint,Normalize," +
             "TN,FP,FN,TP,TN/n,FP/n,FN/n,TP/n,Accuracy,Precision").encode())

    runs = []   # input combinations
    for class_ in cd.keys():
        for nf in n_f:
            for d in disjoint:
                for n in norm:
                    for i in it:
                        runs.append([class_, nf, d, n, i])

    # parallel processing goes here
    t0 = time.time()  # start watch
    data = parfor(train_variation, runs)  # outputs, one for each input combo
    t1 = time.time()  # stop watch

    # now write the log files etc.
    for result in data:
        print("result", result)
        [TN, FP, FN, TP,
         TN_p, FP_p, FN_p, TP_p,
         accuracy, precision, clf, params] = result

        # unpack params
        class_, nf, d, n, i = params

        # one line of log output
        line_to_write = ('\n' + class_ + "," + str(nf) + "," +
                         i + "," + str(d) + "," + str(n) + "," + TN +
                         "," + FP + "," + FN + "," + TP + "," + TN_p +
                         "," + FP_p + "," + FN_p + "," + TP_p + "," +
                         accuracy + "," + precision)
        print(line_to_write)
        f.write(line_to_write.encode())

    print(len(runs), "models fit in", t1 - t0, "seconds,",
          (t1 - t0) / len(runs), "seconds per model")
    return data


if __name__ == "__main__":
    data_folder = "data_bcgw"

    if not exist(data_folder) or not os.path.isdir(data_folder):
        err("please run from fuel-for-fire/ folder," +
            " with 20191207data.tar.gz extracted there")

    # split up new ground reference files, by value, result: binary maps
    if exist("data_vri/") and not exist("data_vri/binary/"):
        run("python3 dev/class_split.py data_vri/")

    dirs, data = ["data_img/", "data_bcgw/", "data_vri/binary/"], None
    data_frame = None
    if not exist('data_frame.pkl'):
        data_frame = populate_data_frame(get_data(dirs),
                                         showplots=True)  # False)
        pickle.dump([data_frame, lines, samples], open('data_frame.pkl', 'wb'))
    else:
        [data_frame, lines, samples] = \
            pickle.load(open('data_frame.pkl', 'rb'))

    image_type = 'all'
    if not exist('data.pkl'):
        '''
        data = train_all_variations_folded(data_frame,
                                           n_f=range(2, 21),
                                           disjoint=[True],
                                           norm=[True],
                                           it=['all'])
        '''
        data = train_all_variations_folded(data_frame,
                                           n_f=[2], #[2, 5, 10],
                                           # range(2, 21),
                                           disjoint=[False],
                                           norm=[True],
                                           it=['all']) #image_type])  # image type

        pickle.dump(data, open('data.pkl', 'wb'))
    else:
        data = pickle.load(open('data.pkl', 'rb'))

    # extract the Sentinel 2 / Landsat 8 stack from the data frame
    X = get_x_data(data_frame, 'all') #image_type)
    a = np.zeros((lines, samples, 3))
    a[:, :, 0] = X.S2A_4.values.reshape(lines, samples)
    a[:, :, 1] = X.S2A_3.values.reshape(lines, samples)
    a[:, :, 2] = X.S2A_2.values.reshape(lines, samples)
    a = (a - np.min(a)) / np.max(a)


    values = a.ravel()

    print("a", a)
    print("a.shape", a.shape)

    # ash was hoping to output class maps from the data next
    for di in data:
        [TN, FP, FN, TP, TN_p, FP_p, FN_p, TP_p,
         accuracy, precision, clf, params] = di

        # unpack params
        class_, nf, d, n, i = params

        print("class", class_, "di", di)
        print("\tparams", params)

        groundref_name = class_ + '_bool'
        if not groundref_name in data_frame:
            err("failed to find: " + groundref)

        # converted to float at end, so matplotlib would show data values on the plot
        groundref = data_frame[groundref_name].values.reshape(lines, samples).astype(float)

        # make prediction on full data
        y_pred = clf.predict(X)
        y = np.array([(1. if y_i is True else 0.) for y_i in y_pred],
                     dtype=float)

        print("hist", hist(y))
        samples, lines = int(samples), int(lines)
        y = y.reshape(lines, samples)

        plt.close('all')
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        ax1.imshow(a)
        ax2.imshow(groundref, cmap = 'binary_r') # why we have to reverse colourmap, don't know!
        ax3.imshow(y) #, cmap='binary')

        ax1.set_title('image')
        ax2.set_title('reference: ' + groundref_name)
        ax3.set_title('predicted: ' + groundref_name)
        plt.tight_layout()
        plt.show()

