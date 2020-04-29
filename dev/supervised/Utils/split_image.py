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
    # cols, rows, bands, X = read_binary('data/full/data_img/S2A.bin', to_string=False)
    vis_split_RGB()
    # # We are creating spatially consistent sub images, so need the original to be in spatial format

    # X = X.reshape((cols, rows, bands))
    # # Let's stuff X with the labels (append them), then the labels will be correctly indexed as well

    # print("Original image shape", X.shape)

    # # this creates equally sized subimages of the original image
    # sub_images = create_sub_images(X, cols, rows, bands)

    # # figure out how many white pixels we need to add total
    # w = div_by_10(sub_images.shape[1])
    # h =  div_by_10(sub_images.shape[2])
    # t, r, l, b = padding(w, h)

    # img = sub_images[1,:,:,:]
    # print(img.shape)
    # # make array of zeros that has the image with the paddings size
    # tmp = np.zeros((img.shape[0] + l + r, img.shape[1] + t + b, img.shape[2]), dtype=np.float32)

    # tmp[l+1 : img.shape[0] + r, t + 1 : img.shape[1] + b, :] = img

    # hist, bins = np.histogram(tmp)
    # plt.hist(hist)
    # plt.title("histogram of padded data")
    # plt.show()
    # val, count = np.unique(tmp, return_counts=True)
    # print()

    # for x in range(sub_images.shape[0]): # for each sub image

def padding(w, h):
    l, r = split_padding(w)
    t, b = split_padding(h)
    l = int(l)
    r = int(r)
    t = int(t)
    b = int(b)

    return t, r, l, b


def split_padding(dim):
    a = dim // 2
    b = dim - a
    a = int(a)
    b = int(b)
    return a, b


def div_by_10(dim):
    pixels_to_add = 0

    while dim % 10 != 0:
        dim += 1
        pixels_to_add += 1
    return pixels_to_add
"""
    Indexing the original image to produce a
    collection that contains the entire image
    chunked into 10 squares"""
def create_sub_images(X, cols, rows, bands):
    # TODO: Be nice to automate this.. need some type of LCD function ...
    sub_cols = cols//2 # not sure how to automate this yet but I know that these dims will create 10 sub images
    sub_rows = rows//5
    # shape of the sub images [sub_cols, sub_rows, bands]
    print("New subimage shape (%s, %s, %s)" % (sub_cols, sub_rows, bands))

    # container for the sub images
    sub_images = np.zeros((10, sub_cols, sub_rows, bands))

    # this will grab a sub set of the original image beginning with the top left corner, then the right top corner
    # and iteratively move down the image from left to right

    """
    Original image         subimages
    --------                --------
    |      |                [  ][  ]
    |      |                [  ][  ]
    |      |                [  ][  ]
    |      |                [  ][  ]
    |      |                [  ][  ]
    --------                --------
    """
    index = 0 # to index the container above for storing each sub image
    for row in range(5): # represents the 5 'rows' of this image
        for col in range(2): # represents the left and right side of the image split down the middle

            sub_images[index, :,:,:] = X[sub_cols * col : sub_cols * (col + 1), sub_rows * row : sub_rows * (row + 1), :]
            index += 1

    print("images, width, height, features", sub_images.shape)
    return sub_images

""" A hardcoded sanity check on how to visualize splitting the
    original image into 10 sub images
"""
def vis_split_RGB():
    s, l, b, X = read_binary('data/full/data_img/S2A.bin', to_string=False)

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

    # index the image x_start : x_end, y_start : y_end,  bands
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
