import os
import sys
sys.path.append(os.curdir)
from Utils.Misc import *
import numpy as np

path = "/home/brad/Projects/research/fuel-for-fire/outs/KFold/Seeded/run__2020_08_04-12_32_19/data"
fn = f"water_seeded-95percentile_proba-prediction"
proba_predictions = None


single = True


if single:
    proba_predictions = load_np(f'{path}/{fn}-1.npy')
else:
    for x in range(5):
        filename = f'{fn}-{x}.npy'
        if proba_predictions is None:
            print("Initialize proba predictions")
            proba_predictions = load_np(f'{path}/{filename}')

        else:
            print("concat")
            proba_predictions = np.concatenate((proba_predictions, load_np(f'{path}/{filename}')))
plt.hist(proba_predictions, bins=10)
plt.show()

