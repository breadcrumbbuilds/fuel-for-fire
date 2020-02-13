import numpy as np


class Helper():


    @staticmethod
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