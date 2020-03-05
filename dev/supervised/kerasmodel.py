import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from Utils.Helper import rescale
from Utils.Misc import read_binary


def main():
    print("Numpy Version: %s" % np.__version__)
    print("Keras Version: %s" % keras.__version__)

    target = {
        "broadleaf" : "BROADLEAF_SP.tif_proj.bin",
        "ccut" : "CCUTBL_SP.tif_proj.bin",
        "conifer" : "CONIFER_SP.tif_proj.bin",
        "exposed" : "EXPOSED_SP.tif_proj.bin",
        "herb" : "HERB_GRAS_SP.tif_proj.bin",
        "mixed" : "MIXED_SP.tif_proj.bin",
        "river" : "RiversSP.tif_proj.bin",
        "road" : "RoadsSP.tif_proj.bin",
        "shrub" : "SHRUB_SP.tif_proj.bin",
        "vri" : "vri_s3_objid2.tif_proj.bin",
        "water" : "WATERSP.tif_proj.bin",
    }

    xs, xl, xb, X = read_binary('data/data_img/output4_selectS2.bin')
    xs = int(xs)
    xl = int(xl)
    xb = int(xb)
    X = X.reshape(xl*xs, xb)

    ys, yl, yb, y = read_binary("data/data_bcgw/%s" % target['road'])
    ys = int(ys)
    yl = int(yl)
    yb = int(yb)
    y = y.reshape(yl,ys)



    print(X.shape)
    print(y.shape)


def visVRI():
    samples, lines, bands, y = read_binary('data/data_bcgw/vri_s3_objid2.tif_proj.bin')
    s = int(samples)
    l = int(lines)
    b = int(bands)
    y = y.reshape(l, s)

    print(y.shape)



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