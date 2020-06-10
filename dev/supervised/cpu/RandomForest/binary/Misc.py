''' basic functions and functions for manipulating ENVI binary files'''
import os
import sys
import copy
import math
import struct
import numpy as np
import os.path as path
import matplotlib.pyplot as plt


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
