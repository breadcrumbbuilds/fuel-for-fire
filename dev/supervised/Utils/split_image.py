import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import matplotlib.pyplot as plt
from Utils.Misc import read_binary
from Utils.Helper import rescale

def main():

    # TODO: Split both the data and the target
    s, l, b, X = read_binary('data/data_img/output4_selectS2.bin', to_string=False)

    # TODO: Save split files with coordinates (together with targets?)

# cols, rows, bands, X = read_binary('data/data_img/output4_selectS2.bin', to_string=False)

def vis_split_RGB():
    s, l, b, X = read_binary('data/data_img/output4_selectS2.bin', to_string=False)

    data_r = X.reshape(b, s * l)
    rgb = np.zeros((l, s, 3))

    for i in range(0, 3):
        rgb[:, :, i] = data_r[4 - i, :].reshape((l, s))
    for i in range(0,3):
        rgb[:, :, i] = rescale(rgb[:, :, i], two_percent=True)
    del X

    s_cols = s//2
    s_rows = l//5
    print(l)
    print(s)
    print(s_cols * s_rows)
    print(s_rows)

    fig, axs = plt.subplots(5, 2, figsize=(9, 6), sharey=False)
    axs[0,0].imshow(rgb[0:s_rows, 0:s_cols, :])
    axs[0,1].imshow(rgb[0:s_rows, s_cols:s_cols*2, :])

    axs[1,0].imshow(rgb[s_rows:s_rows*2, 0:s_cols, :])
    axs[1,1].imshow(rgb[s_rows:s_rows*2, s_cols:s_cols*2, :])

    axs[2,0].imshow(rgb[s_rows*2:s_rows*3, 0:s_cols, :])
    axs[2,1].imshow(rgb[s_rows*2:s_rows*3, s_cols:s_cols*2, :])

    axs[3,0].imshow(rgb[s_rows*3:s_rows*4, 0:s_cols, :])
    axs[3,1].imshow(rgb[s_rows*3:s_rows*4, s_cols:s_cols*2, :])

    axs[4,0].imshow(rgb[s_rows*4:s_rows*5, 0:s_cols, :])
    axs[4,1].imshow(rgb[s_rows*4:s_rows*5, s_cols:s_cols*2, :])

    # plt.imshow(rgb[0:s_rows, 0:s_cols, :])
    # plt.imshow(rgb)
    plt.tight_layout(pad=.1)
    plt.savefig('outs/SplitImage.png')
    plt.show()


if __name__ == "__main__":

   main()

# X = X.reshape((cols,rows,bands))

# rgb = np.zeros((X.shape[0], X.shape[1], 3))
# for i in range(0, 3):
#         rgb[:, :, i] = X[4 - i, :].reshape((X.shape[0], X.shape[1]))
# for i in range(0,3):
#     rgb[:, :, i] = rescale(X[:, :, i])

# plt.imshow(rgb)
# plt.show()
# sub_images = list()
# s_cols = cols//2
# s_rows = rows//5



# for x in range(1):
#     for y in range(5):
#         X_sub = X[x * s_rows : s_rows * (x+1), y * s_cols : s_cols * (y+1), :]
#         print(X_sub.shape)
#         plt.show()
#         sub_images.append(X_sub)
# X = X[:s_cols,:s_rows, :]


# for img in sub_images:

#     img_rs = img.reshape(img.shape[2], img.shape[0] * img.shape[1]) # flip bands to the first axis
#     print(img_rs.shape)
#     rgb = np.zeros((img.shape[0], img.shape[1], 3))


#     for i in range(0, 3):
#         rgb[:, :, i] = img_rs[4 - i, :].reshape((img.shape[0], img.shape[1]))
#     for i in range(0,3):
#         rgb[:, :, i] = rescale(rgb[:, :, i])
#     print(rgb.shape)

#     plt.imshow(rgb)
#     plt.show()

#     print(img.shape)