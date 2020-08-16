import os
import re
import sys
sys.path.append(os.curdir)
from Utils.Misc import *
import numpy as np

single = False
dirty_path = sys.argv[1]
target = sys.argv[2]
percentile = sys.argv[3]
type= sys.argv[4]
pattern = f'val_{target}_seeded-{percentile}percentile_{type}-prediction_xxx.npy'
if single:
    proba_predictions = load_np(dirty_path)
else:
    proba_predictions = None
    for x in range(5):
        filename = pattern.replace('xxx', str(x))
        if proba_predictions is None:
            print("Initialize proba predictions")
            proba_predictions = load_np(f'{dirty_path}/{filename}')

        else:
            print("concat")
            proba_predictions = np.concatenate((proba_predictions, load_np(f'{dirty_path}/{filename}')))
max = 0
for x in proba_predictions:
    if x > max:
        max = x

print(max)

try:
    proba_predictions = np.round(proba_predictions, decimals=2)
except:
    pass
plt.hist(proba_predictions, bins=10)
plt.title("Initial RF Prediction Probability")
plt.suptitle("Probability rounded to .x")
plt.show()
