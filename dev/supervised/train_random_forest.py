from Utils.Helper import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import multiprocessing


def main():

 ## TUNING

    number_of_estimators = 5
    # 23 features here, reduce the number of features each tree will see
    max_features = 17

    ## DATA WRANGLE
    data = Helper.init_data()
    print()

    for key in data.Target.keys():
        class_to_train = key
        X = data.Combined.Data()
        y = data.Target[class_to_train].Binary
        print("X Shape:", X.shape)
        print("y Shape:", y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25  )

        # n jobs uses all the available cores - turn to 1 so can parallelize the visual
        forest = RandomForestClassifier(n_estimators=number_of_estimators,
                                        random_state=2,
                                        max_features=max_features,
                                        n_jobs=1)

        forest.fit(X_train, y_train)


        ### Generate a representation of the RandomForest's ability
        ### flatten the image so we can enumerate
        rgb = data.S2.rgb.reshape(data.S2.lines * data.S2.samples, 3)
        cpus = multiprocessing.cpu_count()
        print(cpus)
        pool = multiprocessing.Pool(processes=cpus)

        """
                FANCY FRENCH CUISINE - One hot plate of python please
        """
        arr = [pool.apply(comparePixel, args=(forest, pixel, X[idx,:].reshape(1,23), y[idx], rgb[idx,:])) for idx, pixel in enumerate(rgb[:,0])]
        arr = np.asarray(arr)
        arr = arr.reshape(401, 410, 3)
        pool.close()

        plt.imshow(arr)
        plt.savefig("RandomForest_" + key)
    #makeRep(forest, X, y, rgb)
def comparePixel(clf, pixel, features, reality, rgb):

    result = clf.predict(features)
    if result and reality:
            # True Positive
        return [0,1,0]
    elif result and not reality:
        # False Positive
        return [1,0,0]
    elif not result and reality:
        # False Negative
        return [1,0.5,0]
    elif not result and not reality:
        # True Negative, leave the pixel
        return rgb
    else:
        print("error")

if __name__ == "__main__":

   main()