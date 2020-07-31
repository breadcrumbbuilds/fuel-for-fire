import copy
import math
import numpy as np

from Utils.targets import get_bcgw_targets


def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy =  np.array(X)
    y_copy = np.array(y)

    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:,:-1]
        y_copy = data[:,-1].astype(int)

    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i : i + batch_size, :], y_copy[i : i + batch_size])


@staticmethod
def mean_center_normalize(X):
    mean_vals = np.mean(X, axis=0)
    std_val = np.std(X)

    return mean_vals, std_val

@staticmethod
def init_data():
    return rb("data", "data_img", "data_bcgw", get_bcgw_targets())


def rescale(arr, two_percent=True):
    arr_min = arr.min()
    arr_max = arr.max()
    scaled = (arr - arr_min) / (arr_max - arr_min)

    if two_percent:
        # 2%-linear stretch transformation for hi-contrast vis
        values = copy.deepcopy(scaled)
        values = values.reshape(np.prod(values.shape))
        values = values.tolist()
        values.sort()
        npx = len(values)  # number of pixels
        if values[-1] < values[0]:
            print('error: failed to sort')
            sys.exit(1)
        v_min = values[int(math.floor(float(npx)*0.02))]
        v_max = values[int(math.floor(float(npx)*0.98))]
        scaled -= v_min
        rng = v_max - v_min
        if rng > 0.:
            scaled /= (v_max - v_min)

    return scaled