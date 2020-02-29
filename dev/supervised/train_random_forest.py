from Utils.Helper import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import multiprocessing
import color
import vector


"""
    Paralellized program to create RandomForests for all 9 classes

"""
def run(key):
    # colours = [color.hsv_to_rgb(vector( i / (next_label - 1), 1., 1.)) for i in range(0, next_label)]

    n_estimators = 5
    n_features = 17
    data = Helper.init_data()
    X = data.Combined.Data()
    y = data.Target[key].Binary
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25  )

    # n jobs uses all the available cores - turn to 1 so can parallelize the visual
    forest = RandomForestClassifier(n_estimators=n_estimators,
                                    random_state=2,
                                    max_features=n_features,
                                    n_jobs=1)

    forest.fit(X_train, y_train)

    predictions = forest.predict(X)
    trainScore = forest.score(X_train, y_train)
    testScore = forest.score(X_test, y_test)
    print(key)
    print('Train: %.3f' % trainScore)
    print('Test: %.3f' % testScore)
    print()

    for idx, pixel in enumerate(zip(predictions, y)):
        if pixel[0] == True and pixel[1] == True:
            # True Positive

            pass
        elif pixel[0] and not pixel[1]:
            # False Positive
            pass
        elif not pixel[0] and pixel[1]:
            # False Negative
            pass
        elif not pixel[0] and not pixel[1]:
            # True Negative
            pass
        else:
            raise Exception("There was a problem predicting the pixel", idx)

    return forest#, visualization, trainScore, testScore, confMatrix,
def main():

 ## TUNING
    data = Helper.init_data()
    # rgb = data.S2.rgb.reshape(data.S2.lines * data.S2.samples, 3)

    cpus = multiprocessing.cpu_count()
    print('Found %s threads' % cpus)
    pool = multiprocessing.Pool(processes=cpus)
    print()
    print(data.Target.keys())
    # arr = [pool.apply(comparePixel, args=(forest, pixel, X[idx,:].reshape(1,23), y[idx], rgb[idx,:])) for idx, pixel in enumerate(rgb[:,0])]
    result = pool.map(run, data.Target.keys())


        # arr = np.asarray(arr)
        # arr = arr.reshape(401, 410, 3)
        # pool.close()

        # plt.imshow(arr)
        # plt.savefig("RandomForest_" + key)
    #makeRep(forest, X, y, rgb)
# def comparePixel(clf, pixel, features, reality, rgb):

#     result = clf.predict(features)
#     if result and reality:
#             # True Positive
#         return [0,1,0]
#     elif result and not reality:
#         # False Positive
#         return [1,0,0]
#     elif not result and reality:
#         # False Negative
#         return [1,0.5,0]
#     elif not result and not reality:
#         # True Negative, leave the pixel
#         return rgb
#     else:
#         print("error")

if __name__ == "__main__":

   main()