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

def save_np(arr, path):
    np.save(path, arr)
    print(f'+w {path}')

def load_np(path):
    print(f'+r {path}')
    np.load(path)


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
    """ Binary read. First check if there is a npy file (this should be in
        scikit format). if not, read the raw binary"""

    hdr = hdr_fn(fn)

    # read header and print parameters
    samples, lines, bands = read_hdr(hdr)
    if not to_string:
        samples = int(samples)
        lines = int(lines)
        bands = int(bands)

    print("\tsamples", samples, "lines", lines, "bands", bands)
    np_fn = fn.replace('.bin', '.npy')
    if os.path.exists(np_fn):
        data = load_np(np_fn)
    else:
        data = read_float(fn)
        print(data.shape)
        data = bsq_to_scikit(samples, lines, bands, data)
        save_np(data, np_fn.replace('.npy', ''))
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
    """ Create the working directories for the data, results and models of this
        run.

        To use with analysis directories following training, use this scheme for
        the path variable:
        {algorithm}/{modification/description of algorithm}/{key parameter}/{sampling type}

        example:
        data_dir, results_dir, models_dir =
            get_working_directories(RandmomForest/Stumps/1000-trees/unsersample)


    """
    root_of_output = mkdir(os.path.join(os.curdir, 'outs'))
    for dir in path.split("/"):
        root_of_output = mkdir(os.path.join(root_of_output, dir))
    root_of_output = mkdir(get_run_logdir(root_of_output))
    return  mkdir(os.path.join(root_of_output, 'data')), mkdir(os.path.join(root_of_output, 'results')), mkdir(os.path.join(root_of_output, 'models'))


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



def split_train_val(data, shape):
    """ Splits X into 5 sub images of equal size and return the sub images in a list """
    train = list()
    val = list()
    for x in range(5):
        for y in range(2):
            x_start = x * shape[0]
            x_end = (x+1) * shape[0]
            y_start = y * shape[1]
            y_end = (y+1) * shape[1]
            if len(data.shape) > 2:
                if y == 0:
                    train.append(data[ x_start:x_end , y_start : y_end, :])
                else:
                    val.append(data[x_start:x_end, y_start:y_end, :])
            else:
                if y == 0:
                    train.append(data[x_start:x_end , y_start : y_end])
                else:
                    val.append(data[x_start:x_end, y_start:y_end])
    return train, val


def save_subimg_maps(y_subbed_list, sub_img_shape, data_output_directory, target, filename):
    print("Saving sub image maps")
    for x, sub_img in enumerate(y_subbed_list):
        save_np(sub_img, os.path.join(data_output_directory, f"{target}-{filename}_{x}"))


def save_rgb(subimgs, sub_img_shape, output_directory, name):
    """ Saves each subimgs RGB interpretation to output_directory """
    print("Creating RGB sub images")
    sub_imgs = list()
    temp = np.zeros((sub_img_shape[0], sub_img_shape[1], 3))
    rgb_stretched = np.zeros((sub_img_shape[0], sub_img_shape[1], 3))
    full_img = None
    for x, data in enumerate(subimgs):
        rgb = np.zeros((sub_img_shape[0], sub_img_shape[1], 3))
        for i in range(0,3):
            rgb[:,:, i] = data[:,:, 4 - i]
        if full_img is None:
            full_img = rgb
        else:
            full_img = np.concatenate((full_img, rgb))
    for i in range(0,3):
        full_img[:,:,i] = rescale(full_img[:,:,i], two_percent=False)
    save_np(full_img, os.path.join(output_directory, f"rgb_{name}_image-twopercentstretch"))
    print()


def save_rgb_subbed(subimgs, sub_img_shape, output_directory, name):
    """ Saves each subimgs RGB interpretation to output_directory """
    print("Creating RGB sub images")
    sub_imgs = list()
    temp = np.zeros((sub_img_shape[0], sub_img_shape[1], 3))
    rgb_stretched = np.zeros((sub_img_shape[0], sub_img_shape[1], 3))
    full_img = None
    for x, data in enumerate(subimgs):
        rgb = np.zeros((sub_img_shape[0], sub_img_shape[1], 3))
        for i in range(0,3):
            rgb[:,:, i] = data[:,:, 4 - i]
        if full_img is None:
            full_img = rgb
        else:
            full_img = np.concatenate((full_img, rgb))
    for i in range(0,3):
        full_img[:,:,i] = rescale(full_img[:,:,i], two_percent=False)
    save_np(full_img, os.path.join(output_directory, f"rgb_{name}_image-twopercentstretch"))
    print()


def save_rgb(img, shape, output_directory, name):
    temp = np.zeros((shape[0], shape[1], 3))
    for i in range(0,3):
        temp[:,:, i] = img[:,:, 4 - i]
    for i in range(0,3):
        temp[:,:,i] = rescale(temp[:,:,i], two_percent=False)
    save_np(temp, os.path.join(output_directory, f"rgb_{name}_twopercentstretch"))
    print()


def bsq_to_scikit(ncol, nrow, nband, d):
    # convert image to a format expected by sgd / scikit learn
    print("Converting bsq to Sklearn Format")
    npx = nrow * ncol # number of pixels

    # convert the image data to a numpy array of format expected by sgd
    img_np = np.zeros((npx, nband))
    for i in range(0, nrow):
        ii = i * ncol
        for j in range(0, ncol):
            for k in range(0, nband):
                # don't mess up the indexing
                img_np[ii + j, k] = d[(k * npx) + ii + j]
    return(img_np)