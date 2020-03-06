import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from sklearn.model_selection import train_test_split
from Utils.Helper import rescale
from Utils.Misc import read_binary


def main():
    print("Numpy Version: %s" % np.__version__)
    print("Keras Version: %s" % keras.__version__)

    test_size = .25
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