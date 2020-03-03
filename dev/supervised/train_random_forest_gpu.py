from Utils.Helper import *
import tensorflow as tf

from tensorflow.python.ops import resources
from sklearn.model_selection import train_test_split
# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

data = Helper.init_data()
X = data.S2.Data()
y = data.Target['water'].Binary
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
feat_labels = [str(x) for x in range(X.shape[1])]
feat_labels = np.asarray(feat_labels)

# Parameters
num_steps = 500 # Total steps to train
batch_size = 1024 # The number of samples per batch
num_classes = len(data.Targets) # The 10 digits
num_features = len(X[:1]) # Each image is 28x28 pixels
num_trees = 10
max_nodes = 1000

print(num_steps)
print(batch_size)
print(num_classes)
print(num_features)
print(num_trees)
print(max_nodes)

# # Input and Target data
# X = tf.placeholder(tf.float32, shape=[None, num_features])
# # For random forest, labels must be integers (the class id)
# Y = tf.placeholder(tf.int32, shape=[None])

# # Random Forest Parameters
# hparams = tensor_forest.ForestHParams(num_classes=num_classes,
#                                       num_features=num_features,
#                                       num_trees=num_trees,
#                                       max_nodes=max_nodes).fill()

# # Build the Random Forest
# forest_graph = tensor_forest.RandomForestGraphs(hparams)
# # Get training graph and loss
# train_op = forest_graph.training_graph(X, Y)
# loss_op = forest_graph.training_loss(X, Y)

# # Measure the accuracy
# infer_op, _, _ = forest_graph.inference_graph(X)
# correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
# accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# # Initialize the variables (i.e. assign their default value) and forest resources
# init_vars = tf.group(tf.global_variables_initializer(),
#     resources.initialize_resources(resources.shared_resources()))

# # Start TensorFlow session
# sess = tf.Session()

# # Run the initializer
# sess.run(init_vars)

# # Training
# for i in range(1, num_steps + 1):
#     # Prepare Data
#     # Get the next batch of MNIST data (only images are needed, not labels)
#     batch_x, batch_y = mnist.train.next_batch(batch_size)
#     _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
#     if i % 50 == 0 or i == 1:
#         acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
#         print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

# # Test Model
# test_x, test_y = mnist.test.images, mnist.test.labels
# print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))