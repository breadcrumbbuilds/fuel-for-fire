import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from Utils.Misc import read_binary


xs, xl, xb, X = read_binary('data/data_img/output4_selectS2.bin', to_string=False)


X = X.reshape((xs,xl,xb))
print(X.shape)
