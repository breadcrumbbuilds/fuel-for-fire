
"""
*   Attempting to dynamically load data to a keras model
*
*
*
*
"""

class DataGenerator:


    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 11), n_channels=1, n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __data_generation(self, list_IDs_temp):
        'Generate data containing batch_size samples'
        # Init
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.load(f'data/prepared/train/cropped/{i}-data.npy')
            y[i] = np.load(f'data/prepared/train/cropped/{i}-label.npy')

        return X, keras.utils.to_categorical(y, num_class=self.n_classes)


    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        indices = self.indeces[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indices]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)