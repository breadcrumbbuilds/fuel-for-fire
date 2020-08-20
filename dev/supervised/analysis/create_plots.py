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
    rgb_pattern = f'rgb_side_image-twopercentstretch.npy'
    orig_map_pattern = f'target-side_map_xxx.npy'
    orig_pred_pattern = f'val_target_initial_proba-prediction_xxx.npy'
    val_pred_pattern = f'val_target_initial_proba-prediction_xxx.npy'
    val_map_pattern = f'target-{percentile}-percentile_map_xxx.npy'
    seeded_pred_pattern = f'val_target_seeded-{percentile}percentile_type-prediction_xxx.npy'

    figure, axes = plt.subplots(2, 4, sharex=True, figsize=(20,10))

    """ Top row, RGB and Original Maps """

    try:
        orig_map = read_sub_imgs(dir, orig_map_pattern, 'training', target=target)
        axes[0][0].set_title("Training Orig Reference")
        axes[0][0].imshow(orig_map, cmap='gray',  vmin=0, vmax=1)

        rgb_train = read_full_img(dir, rgb_pattern, 'training')
        axes[0][1].set_title("Training RGB")
        axes[0][1].imshow(rgb_train)

        rgb_val = read_full_img(dir, rgb_pattern, 'validation')
        axes[0][2].set_title("Val RGB")
        axes[0][2].imshow(rgb_val)

        val_map = read_sub_imgs(dir, orig_map_pattern, 'validation', target=target)
        axes[0][3].set_title("Val Orig Reference")
        axes[0][3].imshow(val_map, cmap='gray',  vmin=0, vmax=1)

        """ Second Row, Right to Left, Initial Pred, The Map """

        val_pred = read_sub_imgs(dir, val_pred_pattern, 'validation', target=target).reshape(4835, 3402//2)
        axes[1][3].set_title(f"Initial Model Val Prediction")
        axes[1][3].imshow(val_pred, cmap='gray', vmin=0, vmax=1)

        seeded_map = read_sub_imgs(dir, val_map_pattern, 'validation', target=target).reshape(4835, 3402//2)
        axes[1][2].set_title(f"{percentile}th Percentile Reference")
        axes[1][2].imshow(seeded_map, cmap='gray', vmin=0, vmax=1)

        seeded_pred_proba = read_sub_imgs(dir, seeded_pred_pattern, 'validation', target=target, type='proba').reshape(4835, 3402//2)
        axes[1][1].set_title(f"{percentile}th Percentile Prediction")
        axes[1][1].imshow(seeded_pred_proba, cmap='gray',  vmin=0, vmax=1)

        seeded_pred_class = read_sub_imgs(dir, seeded_pred_pattern, 'validation', target=target, type='class').reshape(4835, 3402//2)
        axes[1][0].set_title(f"{percentile}th Percentile Prediction")
        axes[1][0].imshow(seeded_pred_class, cmap='gray',  vmin=0, vmax=1)


    except Exception as e:
        # Let's just show whatever has worked so far
        print(e)

    plt.tight_layout()
    plt.show()


def read_sub_imgs(dir, pattern, side, target=None, type=None):
    full_img = None
    for x in range(5):
        x = str(x)
        if target is None:
            sub_img = load_np(os.path.join(dir,pattern.replace('side', side).replace('xxx', x) ))
        else:
            if type is None:
                sub_img = load_np(os.path.join(dir,pattern.replace('side', side).replace('xxx', x).replace('target', target) ))
            else:
                sub_img = load_np(os.path.join(dir,pattern.replace('side', side).replace('xxx', x).replace('target', target).replace('type', type) ))
        if full_img is None:
            full_img = sub_img
        else:
            full_img = np.concatenate((full_img, sub_img))
    return full_img


def read_full_img(dir, pattern, side, target=None, model_num=None):
    """ This only reads the RGB image """
    full_img = load_np(os.path.join(dir, pattern.replace("side", side )))
    return full_img


if __name__ == "__main__":

   main()