from Utils.Helper import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import multiprocessing
import pickle
import time

n_estimators = 5
n_features = 8
test_size =.25
file = 'RandForest/%s_%s_%s.png' % ('water', n_estimators, n_features)
data = Helper.init_data()
X = data.S2.Data()
y = (data.Target['river'].Binary | data.Target['water'].Binary) & data.Target['conifer'].Binary

ex = Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)

plt.title("Herb OR Shrub OR Broadleaf AND Conifer")
plt.imshow(y.reshape(data.S2.lines, data.S2.samples), cmap='gray')



plt.savefig("outs/herbshrubbroadleaf-and-conifer")


plt.show()
unique_elements, counts_elements = np.unique(y, return_counts=True)
for c in counts_elements:
    print(c/len(y))

for z in zip(unique_elements, counts_elements):
    print(z)