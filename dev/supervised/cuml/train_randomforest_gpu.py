import cuml
from cuml import RandomForestClassifier as cuRF
from cuml.preprocessing.model_selection import train_test_split
import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import read_binary
from sklearn.model_selection import train_test_split
import cudf
import dask_cudf




def main():
    print("Numpy Version: %s" % np.__version__)
    print("cuML Version: %s" % cuml.__version__)

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
    onehot = onehot.astype(np.int32)
    # test = np.zeros((xl*xs, len(target)))


    X_train, X_test, y_train, y_test = train_test_split(X,onehot, test_size=.3)

    print(f'X_train: {X_train.shape[0]} elements')
    print(f'X_test: {X_test.shape[0]} elements')
    print(f'y_train: {y_train.shape[0]} elements')
    print(f'y_test: {y_test.shape[0]} elements')



    # ## Preprocess
    # X_train, X_test, y_train, y_test = train_test_split(X, onehot, test_size=test_size)

    # X_train_cudf = cudf.DataFrame()
    # X_test_cudf = cudf.DataFrame()
    # y_train_cudf = cudf.DataFrame()
    # y_test_cudf = cudf.DataFrame()
    # for i in range(X.shape[1]):
    #     X_train_cudf.add_column(i, X_train[:,i])
    #     X_test_cudf.add_column(i, X_test[:,i])

    #     # Normalize each band independently
    #     X_test_cudf[i] = (X_test_cudf[i] - X_test_cudf[i].mean()) / X_test_cudf[i].std()
    #     X_train_cudf[i] = (X_train_cudf[i] - X_train_cudf[i].mean()) / X_train_cudf[i].std()
    #     print(X_test_cudf[i].max(), X_test_cudf[i].min())
    # for i in range(onehot.shape[1]):
    #     y_train_cudf.add_column(i, y_train[:,i])
    #     y_test_cudf.add_column(i, y_test[:,i])

    cu_rf_params = {
        'n_estimators': 25,
        'max_depth': 56,
        'n_streams': 8 }

    cu_rf = cuRF(**cu_rf_params)

    cu_rf.fit(X_train, y_train)
    # print(X_train_cudf)
    # print(X_test_cudf)
    # print(y_train_cudf)
    # print(y_test_cudf)

    # X_train_centered = np.zeros(X_train.shape)
    # X_test_centered = np.zeros(X_test.shape)

    # # Need to normalize each band independently
    # for idx in range(X_train.shape[1]) :
    #     print(idx)
    #     train_mean_vals = np.mean(X_train[:,idx], axis=0)
    #     test_mean_vals = np.mean(X_test[:,idx], axis=0)
    #     train_std_vals = np.std(X_train[:,idx])
    #     test_std_vals = np.std(X_test[:,idx])

    #     X_train_centered[:,idx] = (X_train[:,idx] - train_mean_vals) / train_std_vals
    #     X_test_centered[:,idx] = (X_test[:,idx] - test_mean_vals) / test_std_vals

    # print(X_train_centered)
    # print(X_test_centered)

    # # mean_vals = np.mean(X_train, axis=0)
    # # std_vals = np.std(X_train)
    # # X_train_centered = (X_train - mean_vals) / std_vals
    # # X_test_centered = (X_test - mean_vals) / std_vals

    # del X_train, X_test
    # print(X_train_centered.shape, y_train.shape)
    # print(X_test_centered.shape, y_test.shape)



    # # ## Model

    # n_features = X_train_centered.shape[1]
    # n_classes = len(target)
    # rand_seed = 123 # reproducability

    # np.random.seed(rand_seed)

    # print('\nFirst 3 labels (one-hot):\n',y_train[:3])


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

if __name__ == "__main__":

   main()