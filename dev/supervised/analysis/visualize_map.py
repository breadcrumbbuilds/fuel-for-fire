import os
import re
import sys
sys.path.append(os.curdir)
from Utils.Misc import *
import numpy as np


single = False
dirty_path = sys.argv[1]
if len(sys.argv) > 2:
    single = sys.argv[2] == 's'
try:
    if single:
        raise Exception
    path, file = dirty_path.split('/data')
    path += '/data'
    proba_predictions = None
    for x in range(5):
        filename = re.sub("-\d", f'-{x}', file)
        if proba_predictions is None:
            print("Initialize proba predictions")
            proba_predictions = load_np(f'{path}/{filename}')
        else:
            print("concat")
            proba_predictions = np.concatenate((proba_predictions, load_np(f'{path}/{filename}')))
    if "rgb" in dirty_path:
        plt.imshow(proba_predictions.reshape((4835,3402, 3)))
    else:
        plt.imshow(proba_predictions.reshape((4835,1701)), cmap='gray')

    plt.title("Initial (Unseeded) RF Prediction Probability")
    plt.suptitle("K models produce a prediction for that models test sub-image")
    plt.tight_layout()
    plt.show()
except:
    print("Assuming single image passed")
    proba_predictions = load_np(dirty_path)
    print(proba_predictions)
    x = np.unique(proba_predictions, return_counts=True)
    if "val" in filename:
        plt.imshow(proba_predictions.reshape((4835//5,3402//2)), cmap='gray')
        plt.show()
    else:
        plt.imshow(proba_predictions.reshape((4835//5,3402//2)), cmap='gray')
        plt.show()

# y_subbed_list = create_sub_imgs(probability_map, fold_length)