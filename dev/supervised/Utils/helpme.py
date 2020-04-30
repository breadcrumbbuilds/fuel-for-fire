import os
import sys
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import read_binary
from sklearn.preprocessing import StandardScaler


def mkdir(path):
    if not os.path.exists(path):
        print(f"+w {path}")
        os.mkdir(path)
        return True
    else:
        return False


def create_sub_images(X, y, rows, cols):
    # TODO: Be nice to automate this.. need some type of LCD function ...
    # not sure how to automate this yet but I know that these dims will create 10 sub images
    sub_cols = cols//2
    sub_rows = rows//5
    # shape of the sub images [sub_cols, sub_rows, bands]

    subimages = []
    sublabels = []
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
    index = 0  # to index the container above for storing each sub image
    for row in range(5):  # represents the 5 'rows' of this image
        # represents the left and right side of the image split down the middle
        for col in range(2):
            img = X[:,
                    sub_rows * row: sub_rows * (row + 1),
                    sub_cols * col: sub_cols * (col + 1)]

            label = y[sub_rows * row: sub_rows * (row + 1),
                    sub_cols * col: sub_cols * (col + 1)]

            subimages.append(img)
            sublabels.append(label)
            index += 1


    return subimages, sublabels



def encode_one_hot(reference_data_root, xs, xl, array=True):
    """encodes the provided dict into a dense numpy array
    of class values.

    Caveats: The result of this encoding is dependant and naive, in that
    any conflicts of pixel labels are not intelligently resolved. For our
    purposes, at least until now, we don't care. If an instance belongs
    to multiple classes, that instance will be considered a member of
    the last class it encounters, ie, the target that comes latest
    in the dictionary
    """

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

    if array:
        result = np.zeros((xs*xl, len(target)))
    else:
        result = list()

    result = np.zeros((xl * xs))
    reslist = []
    for idx, key in enumerate(target.keys()):
        ones = np.ones((xl * xs))
        s, l, b, tmp = read_binary(f"{reference_data_root}/%s" % target[key])

        # same shape as the raw image
        assert int(s) == int(xs)
        assert int(l) == int(xl)

        # last index is the targets false value
        vals = np.sort(np.unique(tmp))

        # create an array populate with the false value
        t = ones * vals[len(vals) - 1]

        if key == 'water':
            arr = np.not_equal(tmp, t)
        else:
            arr = np.logical_and(tmp, t)
        # at this stage we have an array that has
        # ones where the class exists
        # and zeoes where it doesn't
        _, c = np.unique(arr, return_counts=True)
        reslist.append(c)
        result[arr > 0] = idx+1

    return result


def rawread(raw_data, raw_labels):
    cols, rows, bands, X = read_binary(f'{raw_data}/S2A.bin', to_string=False)
    y = encode_one_hot(raw_labels, cols, rows)


    X = X.reshape(bands, rows, cols)
    y = y.reshape(rows, cols)

    return X, y


def save_np(arr, path):
    np.save(path, arr)
    print(f'+w {path}')


def create_paths(root):
    """--------------------------------------------------------------------
    * Initialize directory structure
    --------------------------------------------------------------------"""
    # the assumed dirs
    raw_data = os.path.join(root, "data_img")
    raw_labels = os.path.join(root, "data_bcgw")
    
    # our new prepared data root
    prep_dir = os.path.join(root, 'prepared') # consider this the root

    # root of the training data
    train_dir = os.path.join(prep_dir, 'train')

    # crop the training data
    crop_dir = os.path.join(train_dir, 'cropped')

    # store the original cropped and oversampled seperately
    orig_dir = os.path.join(crop_dir, 'original')
    osampled_dir = os.path.join(crop_dir, 'oversampled')

    """--------------------------------------------------------------------
    * Create Prepared Directory and Load Original Image
    --------------------------------------------------------------------"""
    if mkdir(prep_dir):
        pass
    else:
        pass

    if mkdir(train_dir):
            # make all the train images
            print("Creating Directories and Loading Data")
            X, y = rawread(raw_data, raw_labels)

            print("Full Images")
            print(f'X shape {X.shape}')
            print(f'y shape {y.shape}')

            save_np(X, f'{train_dir}/full-img.npy' )
            save_np(y, f'{train_dir}/full-label.npy')

    else:
        try:
            X = np.load(f'{train_dir}/full-img.npy')
            y = np.load(f'{train_dir}/full-label.npy')
        except Exception as e:
            print(e)
            print("NP failed to load full images")
            print("Reverting to raw read")
            X, y = rawread(raw_data, raw_labels)
            save_np(X, f'{train_dir}/full-img.npy' )
            save_np(y, f'{train_dir}/full-label.npy')
        

    print(f'X shape {X.shape}')
    print(f'y shape {y.shape}')


    if mkdir(crop_dir):
        # now we have to crop X and y
        subimages, sublabels = create_sub_images(X, y, X.shape[1], X.shape[2])

    else:
        try:
            subimages, sublabels = create_sub_images(X, y, X.shape[1], X.shape[2])

        except:
            subimages, sublabels = create_sub_images(X, y, X.shape[1], X.shape[2])

    for img, label in zip(subimages, sublabels):
        
        fig, axs = plt.subplots(1, 2, sharey=True)
        axs[0].imshow(img[3,:,:])
        axs[1].imshow(label)
        plt.show()
            # attempt to load the saved cropped arrays

            # pass
            # if mkdir(crop_dir):
            #     # crop all the images
            #     pass
            #     if mkdir(orig_dir):
            #         # save the cropped originals in here
            #         pass
            #         if mkdir(osampled_dir):
            #             # over sample each sub image and save it here
            #             pass
            #         else:
            #             # load the oversampled images
            #             pass
            #     else:
            #         # load the original images

            #         pass


    # here we need tos tore the oversampled cropped images


def prepare_data(subpath):
    root_path = f"data/{subpath}/"
    reference_data_root = f"{root_path}data_bcgw/"
    raw_data_root = f"{root_path}data_img/"

    create_paths(root_path)

#     target = {
#             "conifer" : "CONIFER.bin",
#             "ccut" : "CCUTBL.bin",
#             "water": "WATER.bin",
#             "broadleaf" : "BROADLEAF.bin",
#             "shrub" : "SHRUB.bin",
#             "mixed" : "MIXED.bin",
#             "herb" : "HERB.bin",
#             "exposed" : "EXPOSED.bin",
#             "river" : "Rivers.bin",
#             # "road" : "ROADS.bin",
#             # "vri" : "vri_s3_objid2.tif_proj.bin",
#         }


#     data_path = os.path.join(root_path, "prep")
#     data_path_visuals = os.path.join(data_path, "visuals")
#     data_path_training = os.path.join(data_path, "train")
#     mkdir(data_path_visuals)
#     mkdir(data_path_training)

#     data_path_training_raw = os.path.join(data_path_training, "raw")
#     data_path_training_scaled = os.path.join(data_path_training, "scaled")

#     mkdir(data_path_training_scaled)
#     mkdir(data_path_training_raw)

#     cols, rows, bands, X = read_binary(
#         f'{raw_data_root}S2A.bin', to_string=False)

#     X_train = X.reshape((rows * cols, bands))
#     np.save(f'{data_path_training_raw}/raw', X_train)
#     np.save(f'{data_path_training_scaled}/scaled', StandardScaler().fit_transform(X_train))

prepare_data('full')