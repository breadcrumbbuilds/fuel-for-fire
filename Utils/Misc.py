''' basic functions and functions for manipulating ENVI binary files'''
import os
import sys
import copy
import math
import struct
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import time

def err(msg):
    print('Error: ' + msg)
    sys.exit(1)


def run(cmd):
    a = os.system(cmd)
    if a != 0:
        err("command failed: " + cmd)


def exist(f):
    return os.path.exists(f)


def hdr_fn(bin_fn):
    # return filename for header file, given name for bin file
    hfn = bin_fn[:-4] + '.hdr'
    if not exist(hfn):
        hfn2 = bin_fn + '.hdr'
        if not exist(hfn2):
            err("didn't find header file at: " + hfn + " or: " + hfn2)
        return hfn2
    return hfn


def read_hdr(hdr):
    samples, lines, bands = 0, 0, 0
    for line in open(hdr).readlines():
        line = line.strip()
        words = line.split('=')
        if len(words) == 2:
            f, g = words[0].strip(), words[1].strip()
            if f == 'samples':
                samples = g
            if f == 'lines':
                lines = g
            if f == 'bands':
                bands = g
    return samples, lines, bands


# use numpy to read floating-point data, 4 byte / float, byte-order 0
def read_float(fn):
    print("+r", fn)
    return np.fromfile(fn, '<f4')


def wopen(fn):
    f = open(fn, "wb")
    if not f:
        err("failed to open file for writing: " + fn)
    print("+w", fn)
    return f


def read_binary(fn, to_string=True):
    hdr = hdr_fn(fn)

    # read header and print parameters
    samples, lines, bands = read_hdr(hdr)
    print("\tsamples", samples, "lines", lines, "bands", bands)

    data = read_float(fn)
    if not to_string:
        samples = int(samples)
        lines = int(lines)
        bands = int(bands)
    return samples, lines, bands, data


def write_binary(np_ndarray, fn):
    of = wopen(fn)
    np_ndarray.tofile(of, '', '<f4')
    of.close()


def write_hdr(hfn, samples, lines, bands):
    lines = ['ENVI',
             'samples = ' + str(samples),
             'lines = ' + str(lines),
             'bands = ' + str(bands),
             'header offset = 0',
             'file type = ENVI Standard',
             'data type = 4',
             'interleave = bsq',
             'byte order = 0']
    open(hfn, 'wb').write('\n'.join(lines).encode())


def hist(data):
    # counts of each data instance
    count = {}
    for d in data:
        count[d] = 1 if d not in count else count[d] + 1
    return count


def parfor(my_function, my_inputs):
    # evaluate a function in parallel, and collect the results
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(my_function, my_inputs)
    return(result)


def convert_y_to_binary(target, y, cols, rows):
    ones = np.ones((cols * rows))
    vals = np.sort(np.unique(y))
    # create an array populate with the false value
    t = ones * vals[len(vals) - 1]
    if target == 'water':
        y = np.not_equal(y, t)
    else:
        y = np.logical_and(y, t)
    return y

def get_working_directories(path):
    """ Create the working directories for the data and results of this run """
    root_of_output = mkdir(os.path.join(os.curdir, 'outs'))
    for dir in path.split("/"):
        root_of_output = mkdir(os.path.join(root_of_output, dir))
    root_of_output = mkdir(get_run_logdir(root_of_output))
    return  mkdir(os.path.join(root_of_output, 'data')), mkdir(os.path.join(root_of_output, 'results'))


def mkdir(path):
    """ Create a directory if it doesn't exist"""
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'+w {path}')
    else:
        print(f'{path} exists')
    return path


def get_run_logdir(root_logdir):
    """ Create a unique directory for a specific run from system time """
    run_id = time.strftime("run__%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def rescale(arr, two_percent=True):
    """ Rescale arr to 0 - 1, conditionally add linear stretch to arr """
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


def save_np(data, filename):
    """ Saves data using numpy to filename, logs the save to console """
    np.save(filename, data)
    print(f'+w {filename}.npy')

def load_np(filename):
    print(f"+r {filename}")
    return np.load(filename)
