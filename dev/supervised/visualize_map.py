import os
import sys
sys.path.append(os.curdir)
from Utils.Misc import *
import numpy as np


path = "/home/brad/Projects/research/fuel-for-fire/outs/KFold/Seeded/run__2020_08_01-11_04_40/data"
# for x in range(5):
#     arr = load_np(f'{path}/water_proba-prediction-{x}.npy')
#     print(arr)

proba_predictions = None
for x in range(5):
    filename = f'conifer_initial_proba-prediction-{x}.npy'
    if proba_predictions is None:
        print("Initialize proba predictions")
        proba_predictions = load_np(f'{path}/{filename}')

    else:
        print("concat")
        proba_predictions = np.concatenate((proba_predictions, load_np(f'{path}/{filename}')))
plt.imshow(proba_predictions.reshape((4835,3402)), cmap='gray')
plt.show()


# y_subbed_list = create_sub_imgs(probability_map, fold_length)