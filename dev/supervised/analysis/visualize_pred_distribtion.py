import os
import re
import sys
sys.path.append(os.curdir)
from Utils.Misc import *
import numpy as np

single = False
dirty_path = sys.argv[1]

if single:
    proba_predictions = load_np(dirty_path)
else:
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

proba_predictions = np.round(proba_predictions, decimals=1)
plt.hist(proba_predictions, bins=10)
plt.title("Initial RF Prediction Probability")
plt.suptitle("Probability rounded to .x")
plt.show()
