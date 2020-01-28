import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.Misc import *
from Utils.Data import Data

if __name__ == "__main__":

    data = Data("data", "data_img", "data_bcgw")
    print(data.S2.bands)
    print(data.S2.lines)
    print(data.S2.samples)
    print(data.L8.bands)
    print(data.L8.lines)
    print(data.L8.samples)
    plt.imshow(data.S2.Data[:,:,1:4])
    plt.show()