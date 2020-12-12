import sys
import os
sys.path.append(os.curdir)


def read_update_202009(path: str):
    """Read the update-2020-09 dataset.
        test asdfasdfasdfasd
    Args:

    path - directory containing the .bin and .hdr files.

    write_rgb (optional): write the rgb of the raw data.

    Returns:

    X - numpy array of data

    y = numpy array of labels

    """
    cols, rows, bands, data = read_binary(path, to_string=False)
    X = data[:,:11]
    y = data[:,11:] # make this a dict with labels attached

    return X, y


def torgb(path, data_r, shape):
    l, s, b = shape
    path += 'rgb'

    if exist(f'{path}.npy'):
        return load_np(f'{path}.npy')
    print('RGB not found. Creating RGB image')
    rgb = np.zeros((l, s, 3))

    for i in range(0, 3):
        rgb[:, :, i] = data_r[:, 4 - i].reshape((l, s))
    for i in range(0, 3):
        rgb[:, :, i] = rescale(rgb[:, :, i])

    save_np(rgb, path)
    plt.imshow(rgb)
    plt.title(path.split('/')[2])
    plt.savefig(path)
    plt.clf()
    return rgb