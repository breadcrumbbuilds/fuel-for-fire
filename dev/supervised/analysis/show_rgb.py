import os
import sys
sys.path.append(os.curdir)
from Utils.Misc import *
import matplotlib.pyplot as pyplot



plt.imshow(load_np("./outs/KFold/Seeded/run__2020_08_14-15_51_40/data/water-85-percentile_map_3.npy").reshape(967, 1701), cmap='gray')

plt.show()