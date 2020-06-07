from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import time
# User imports
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Utils.Misc import read_binary
from Utils.Helper import *

root_path = "data/full/prepared/train"
reference_data_root = f"{root_path}data_bcgw/"
raw_data_root = f"{root_path}data_img/"

def main():
    n_skip = 51
    n_est = 50
    target = {
        "conifer" : "CONIFER.bin",
        "ccut" : "CCUTBL.bin",
        "water": "WATER.bin",
        "broadleaf" : "BROADLEAF.bin",
        "shrub" : "SHRUB.bin",
        "mixed" : "MIXED.bin",
        "herb" : "HERB.bin",
        "exposed" : "EXPOSED.bin",
        "river" : "Rivers.bin",
        # "road" : "ROADS.bin",
        # "vri" : "vri_s3_objid2.tif_proj.bin",
    }
    classes = ["unlabelled"]
    keys = list(target.keys())
    for key in keys:
        classes.append(key)
    outdir = os.path.join(os.curdir,'outs')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdir = os.path.join(outdir, 'RandomForest')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdir = os.path.join(outdir, 'KFold')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdir = get_run_logdir(outdir)
    os.mkdir(outdir)

    datadir = f'{root_path}'


    X = np.load(f'{datadir}/full-img.npy')
    y = np.load(f'{datadir}/full-label.npy')

    X_train = n_th(X, n_skip)
    y_train = n_th(y, n_skip)

    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')


    rf = RandomForestClassifier(n_estimators=n_est, oob_score=True, verbose=1, n_jobs=-1) # could crash on warn and increase # of estimators

    rf.fit(X_train, y_train)

    predict = rf.predict(X_train)
    correct_classifications = np.logical_and(predict, y_train)
    df = np.sum(np.logical_and(predict, y_train))
    npx_t = math.floor(X_train.shape[0] / n_skip)
    acc = 100. * ((npx_t - abs(df)) / npx_t)
    print("train%", acc)
    predict2 = rf.predict(X) # print("set(predict2)", set(predict2))
    df =  np.sum(np.logical_and(predict2, y))
    acc = 100. * ((X.shape[0] - abs(df)) / X.shape[0])
    print("all  %", acc)

def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run__%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

def n_th(img, n): # take every n-th data point (data in scikit-learn expected format)
    print(len(img.shape))
    npx = img.shape[0]
    npx2 = int(math.floor(npx / n))
    if len(img.shape) == 1:
        nband = 1
        result = np.zeros(npx2)
        for i in range(0, npx, n):
            ip = int(math.floor(i/n))
            if npx2 == ip
                break # index error
            result[ip] = img[i]
    else:
        nband = img.shape[1]
        print(npx, nband)
        result = np.zeros((npx2, nband))
        for i in range(0, npx, n):
            ip = int(math.floor(i/n))
            for k in range(0, nband):
                if npx2 == ip:
                    break
                result[ip, k] = img[i, k]
    return result  # kindof a flaky sampling procedure but it's fairly effective!


if __name__ == "__main__":

   main()
