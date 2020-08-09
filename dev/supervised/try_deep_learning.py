import sys
import os
from tensorflow import keras
sys.path.append(os.curdir)
from Utils.Misc import *

def main():
	root = "data/full/"
	train_root_path = f"{root}/prepared/train/"
	reference_data_root = f"{root}data_bcgw/"
	raw_data_root = f"{root}data_img/"
	data_output_directory, results_output_directory, models_output_directory = get_working_directories("KFold/DeepLearning")

	X = np.load(f'{train_root_path}/full-img.npy')

	sub_img_shape = (4835//5,3402)
	fold_length = X.shape[0] // 5

	model = create_model()


def create_model():
	model = keras.Sequential(
    	[
        keras.layers.Dense(2, activation="relu", name="layer1"),
        keras.layers.Dense(3, activation="relu", name="layer2"),
        keras.layers.Dense(4, name="layer3"),
    	]
	)


if __name__ == "__main__":
	main()
