import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.curdir) # so python can find Utils
from Utils.Misc import *

def main():

    dir = sys.argv[1]
    target = sys.argv[2]
    percentile = sys.argv[3]
    rgb_pattern = f'rgb_side_subimage_xxx-twopercentstretch.npy'
    orig_map_pattern = f'target_side_map_xxx.npy'
    orig_pred_pattern = f'target_initial_proba-prediction_xxx.npy'
    val_pred_pattern = f'val_target_initial_proba-prediction_model-modnum_xxx.npy'
    val_map_pattern = f'target_{percentile}-percentile_map_xxx.npy'
    seeded_pred_pattern = f'val_target_seeded-{percentile}percentile_proba-prediction_model-modnum_xxx.npy'

    figure, axes = plt.subplots(3, 3, sharex=True, figsize=(20,10))

    """ The initial model's images """
    rgb_train = read_sub_imgs(dir, rgb_pattern, 'training')
    axes[0][0].set_title("Training RGB")
    axes[0][0].imshow(rgb_train)

    orig_map = read_sub_imgs(dir, orig_map_pattern, 'training', target='water')
    axes[0][1].set_title("Training Orig Reference")
    axes[0][1].imshow(orig_map, cmap='gray')

    initial_pred = read_sub_imgs(dir, orig_pred_pattern, orig_pred_pattern, target='water')
    axes[0][2].set_title("Initial Model Test Prediciton")
    axes[0][2].imshow(initial_pred, cmap='gray')


    """ Building the Map for the Seeded model """
    model_num = '1'
    rgb_val = read_sub_imgs(dir, rgb_pattern, 'validation')
    axes[1][0].set_title("Validation RGB")
    axes[1][0].imshow(rgb_val)

    orig_map = read_sub_imgs(dir, orig_map_pattern, 'validation', target='water')
    axes[1][1].set_title("Validation Orig Reference")
    axes[1][1].imshow(orig_map, cmap='gray')

    val_pred = read_sub_imgs(dir, val_pred_pattern, 'validation', target='water', model_num=model_num).reshape(4835, 3402//2)
    axes[1][2].set_title(f"Initial Model Validation Prediction")
    axes[1][2].imshow(val_pred, cmap='gray')

    axes[2][0].set_title(f"Training Orig Reference")
    axes[2][0].imshow(orig_map, cmap='gray')

    seeded_map = read_sub_imgs(dir, val_map_pattern, 'validation', target='water', model_num=model_num).reshape(4835, 3402//2)
    axes[2][1].set_title(f"{percentile}th Percentile Reference")
    axes[2][1].imshow(seeded_map, cmap='gray')

    val_pred = read_sub_imgs(dir, seeded_pred_pattern, 'validation', target='water', model_num=model_num).reshape(4835, 3402//2)
    axes[2][2].set_title(f"{percentile}th` Percentile Model Prediction")
    axes[2][2].imshow(val_pred, cmap='gray')
    # model_num = '0'
    # axes[1][1].set_title(f"Validation Model {model_num} Prediction")
    # axes[1][1].imshow(val_pred)

    plt.tight_layout()
    plt.show()


def read_sub_imgs(dir, pattern, side, target=None, model_num=None):
    full_img = None
    for x in range(5):
        x = str(x)
        if target is None:
            sub_img = load_np(os.path.join(dir,pattern.replace('side', side).replace('xxx', x) ))
        else:
            if model_num is None:
                sub_img = load_np(os.path.join(dir,pattern.replace('side', side).replace('xxx', x).replace('target', target) ))
            else:
                sub_img = load_np(os.path.join(dir,pattern.replace('side', side).replace('xxx', x).replace('target', target).replace('modnum', model_num) ))
        if full_img is None:
            full_img = sub_img
        else:
            full_img = np.concatenate((full_img, sub_img))
    return full_img




if __name__ == "__main__":

   main()