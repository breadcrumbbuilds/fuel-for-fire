import sys
import os
sys.path.append(os.curdir)
from Utils.Misc import read_binary, bsq_to_scikit


cols, rows, bands, data = read_binary('data/update-2020-09/stack_v2.bin', to_string=False)
print(cols, rows, bands, data.shape)


