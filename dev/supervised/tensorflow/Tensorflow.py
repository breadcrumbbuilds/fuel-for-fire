import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import numpy as np
from Utils.Helper import Helper
tf.disable_v2_behavior()

class TFLinreg(object):
    def __init__(self, x_dim, learning_rate=0.01,
            random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()

        ## build the model
        with self.g.as_default():
            ## set graph level random seed
            tf.set_random_seed(random_seed)

            self.build()

            self.init_op = tf.global_variables_initializer()

    def build(self):
        ## define the placeholders for inputs
        self.X = tf.placeholder(dtype=tf.float32,
                shape=(None, self.x_dim),
                name='x_input')
        self.y = tf.placeholder(dtype=tf.float32,
                shape=(None),
                name='y_input')

        print(self.X)
        print(self.y)
        ## define weight matrix and bias vector
        w = tf.Variable(tf.zeros(shape=(1)),
                name='weight')
        b = tf.Variable(tf.zeros(shape=(1)),
                name='bias')

        print(w)
        print(b)

        self.z_net = tf.squeeze(w*self.X + b,
                name='z_net')

        print(self.z_net)

        sqr_errors = tf.square(self.y - self.z_net,
                name='sqr_errors')
        print(sqr_errors)

        self.mean_cost = tf.reduce_mean(sqr_errors,
                name='mean_cost')

        optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate,
                name='GradientDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)

    @staticmethod
    def train_linreg(sess, model, X_train, y_train, num_epochs=100):
        ## initialize all varaibles W and b
        sess.run(model.init_op)

        training_costs = []

        for i in range(num_epochs):
            _, cost = sess.run([model.optimizer, model.mean_cost],
                    feed_dict={model.X:X_train,
                        model.y:y_train})
            training_costs.append(cost)

        return training_costs

    @staticmethod
    def predict_linreg(sess, model, X_test):
        y_pred = sess.run(model.z_net, feed_dict={model.X:X_test})
        return y_pred

class LayersMultiLayerPerceptron(object):
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.g = tf.Graph()
        self.learning_rate = learning_rate
        self.n_classes = n_classes

        with self.g.as_default():
            tf.set_random_seed(123)
            self.tf_x = tf.placeholder(dtype=tf.float32,
                                       shape=(None, n_features),
                                       name='tf_x')
            self.tf_y = tf.placeholder(dtype=tf.int32,
                                       shape=None, name='tf_y')
            self.y_onehot = tf.one_hot(indices=self.tf_y, depth=self.n_classes)
            self.h1 = tf.layers.dense(inputs=self.tf_x, units=50, activation=tf.tanh,
                                      name='hidden_layer1')
            self.h2 = tf.layers.dense(inputs=self.h1, units=50,
                                      activation=tf.tanh,
                                      name='hidden_layer2')
            self.logits = tf.layers.dense(inputs=self.h2,
                                          units=n_classes,
                                          activation=None,
                                          name='output_layer')
            self.predictions = {
                'classes' : tf.argmax(self.logits, axis=1,
                                      name='predicted_classes'),
                'probabilites' : tf.nn.softmax(self.logits,
                                               name='softmax_tensor')
            }

            self.build()
            self.init_op = tf.global_variables_initializer()

    def build(self):
        self.cost = tf.losses.softmax_cross_entropy(onehot_labels=self.y_onehot, logits=self.logits)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(loss=self.cost)

    @staticmethod
    def train_mlp(sess, model, X_train, y_train, num_epochs=100):
        sess.run(model.init_op)

        for epoch in range(num_epochs):
            training_costs = []

            batch_generator = HelperFunction.create_batch_generator(
                X_train, y_train, batch_size=256
            )

            for batch_X, batch_y in batch_generator:
                feed = {model.tf_x : batch_X, model.tf_y : batch_y}
                _, batch_cost = sess.run([model.train_op, model.cost], feed_dict=feed)
                training_costs.append(batch_cost)

            print('-- Epoch %2d '
                  'Avg Training Loss: %4f' % (
                      epoch+1, np.mean(training_costs)
                  ))
        return training_costs

    @staticmethod
    def predict_mlp(sess, model, X_test):
        feed = {model.tf_x : X_test}
        y_pred = sess.run(model.predictions['classes'],
                          feed_dict=feed)
        return y_pred

class LayersMultiLayerPerceptron2_50(object):
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.g = tf.Graph()
        self.learning_rate = learning_rate
        self.n_classes = n_classes

        with self.g.as_default():
            tf.set_random_seed(123)
            self.tf_x = tf.placeholder(dtype=tf.float32,
                                       shape=(None, n_features),
                                       name='tf_x')

            self.tf_y = tf.placeholder(dtype=tf.int32,
                                       shape=None, name='tf_y')

            self.y_onehot = tf.one_hot(indices=self.tf_y, depth=self.n_classes)

            self.h1 = tf.layers.dense(inputs=self.tf_x, units=50, activation=tf.tanh,
                                      name='hidden_layer1')

            self.h2 = tf.layers.dense(inputs=self.h1, units=50,
                                      activation=tf.tanh,
                                      name='hidden_layer2')

            self.logits = tf.layers.dense(inputs=self.h2,
                                          units=n_classes,
                                          activation=None,
                                          name='output_layer')

            self.predictions = {
                'classes' : tf.argmax(self.logits, axis=1,
                                      name='predicted_classes'),
                'probabilites' : tf.nn.softmax(self.logits,
                                               name='softmax_tensor')
            }

            self.__build()
            self.init_op = tf.global_variables_initializer()

    def __build(self):
        self.cost = tf.losses.softmax_cross_entropy(onehot_labels=self.y_onehot, logits=self.logits)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(loss=self.cost)

    @staticmethod
    def train_mlp(sess, model, X_train, y_train, num_epochs=100):
        sess.run(model.init_op)

        for epoch in range(num_epochs):
            training_costs = []

            batch_generator = Helper.create_batch_generator(
                X_train, y_train, batch_size=128
            )

            for batch_X, batch_y in batch_generator:
                feed = {model.tf_x : batch_X, model.tf_y : batch_y}
                _, batch_cost = sess.run([model.train_op, model.cost], feed_dict=feed)
                training_costs.append(batch_cost)

            print('-- Epoch %2d '
                  'Avg Training Loss: %4f' % (
                      epoch+1, np.mean(training_costs)
                  ))
        return training_costs

    @staticmethod
    def predict_mlp(sess, model, X_test):
        feed = {model.tf_x : X_test}
        y_pred = sess.run(model.predictions['classes'],
                          feed_dict=feed)
        return y_pred

class LayersMultiLayerPerceptron3_128(object):
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.g = tf.Graph()
        self.learning_rate = learning_rate
        self.n_classes = n_classes

        with self.g.as_default():
            tf.set_random_seed(123)
            self.tf_x = tf.placeholder(dtype=tf.float32,
                                       shape=(None, n_features),
                                       name='tf_x')
            self.tf_y = tf.placeholder(dtype=tf.int32,
                                       shape=None, name='tf_y')
            self.y_onehot = tf.one_hot(indices=self.tf_y, depth=self.n_classes)
            self.h1 = tf.layers.dense(inputs=self.tf_x, units=32, activation=tf.tanh,
                                      name='hidden_layer1')
            self.h2 = tf.layers.dense(inputs=self.h1, units=64,
                                      activation=tf.tanh,
                                      name='hidden_layer2')
            self.h3 = tf.layers.dense(inputs=self.h2, units=128,
                                      activation=tf.tanh,
                                        name="hidden_layer3")
            self.h4 = tf.layers.dense(inputs=self.h3, units=256,
                                      activation=tf.tanh,
                                        name="hidden_layer4")
            self.h5 = tf.layers.dense(inputs=self.h4, units=128,
                                      activation=tf.tanh,
                                        name="hidden_layer5")
            self.h6 = tf.layers.dense(inputs=self.h5, units=64,
                                      activation=tf.tanh,
                                        name="hidden_layer6")
            self.logits = tf.layers.dense(inputs=self.h6,
                                          units=9,
                                          activation=None,
                                          name='output_layer')
            self.predictions = {
                'classes' : tf.argmax(self.logits, axis=1,
                                      name='predicted_classes'),
                'probabilites' : tf.nn.softmax(self.logits,
                                               name='softmax_tensor')
            }

            self.build()
            self.init_op = tf.global_variables_initializer()

    def build(self):
        self.cost = tf.losses.softmax_cross_entropy(onehot_labels=self.y_onehot, logits=self.logits)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(loss=self.cost)

    @staticmethod
    def train_mlp(sess, model, X_train, y_train, num_epochs=100, batch_size=64):
        sess.run(model.init_op)

        for epoch in range(num_epochs):
            training_costs = []

            batch_generator = HelperFunction.create_batch_generator(
                X_train, y_train, batch_size=batch_size
            )

            for batch_X, batch_y in batch_generator:
                feed = {model.tf_x : batch_X, model.tf_y : batch_y}
                _, batch_cost = sess.run([model.train_op, model.cost], feed_dict=feed)
                training_costs.append(batch_cost)

            print('-- Epoch %2d '
                  'Avg Training Loss: %4f' % (
                      epoch+1, np.mean(training_costs)
                  ))
        return training_costs

    @staticmethod
    def predict_mlp(sess, model, X_test):
        feed = {model.tf_x : X_test}
        y_pred = sess.run(model.predictions['classes'],
                          feed_dict=feed)
        return y_pred

class LayersMultiLayerPerceptron9_512(object):
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.g = tf.Graph()
        self.learning_rate = learning_rate
        self.n_classes = n_classes

        with self.g.as_default():
            tf.set_random_seed(123)
            self.tf_x = tf.placeholder(dtype=tf.float32,
                                       shape=(None, n_features),
                                       name='tf_x')
            self.tf_y = tf.placeholder(dtype=tf.int32,
                                       shape=None, name='tf_y')
            self.y_onehot = tf.one_hot(indices=self.tf_y, depth=self.n_classes)
            self.h1 = tf.layers.dense(inputs=self.tf_x, units=32, activation=tf.tanh,
                                      name='hidden_layer1')
            self.h2 = tf.layers.dense(inputs=self.h1, units=64,
                                      activation=tf.tanh,
                                      name='hidden_layer2')
            self.h3 = tf.layers.dense(inputs=self.h2, units=128,
                                      activation=tf.tanh,
                                        name="hidden_layer3")
            self.h4 = tf.layers.dense(inputs=self.h3, units=256,
                                      activation=tf.tanh,
                                        name="hidden_layer4")
            self.h5 = tf.layers.dense(inputs=self.h4, units=512,
                                      activation=tf.tanh,
                                        name="hidden_layer5")
            self.h6 = tf.layers.dense(inputs=self.h5, units=256,
                                      activation=tf.tanh,
                                        name="hidden_layer6")
            self.h7 = tf.layers.dense(inputs=self.h6, units=128,
                                      activation=tf.tanh,
                                        name="hidden_layer7")
            self.h8 = tf.layers.dense(inputs=self.h7, units=64,
                                      activation=tf.tanh,
                                        name="hidden_layer8")
            self.h9 = tf.layers.dense(inputs=self.h8, units=32,
                                      activation=tf.tanh,
                                        name="hidden_layer9")
            self.logits = tf.layers.dense(inputs=self.h9,
                                          units=self.n_classes,
                                          activation=None,
                                          name='output_layer')
            self.predictions = {
                'classes' : tf.argmax(self.logits, axis=1,
                                      name='predicted_classes'),
                'probabilites' : tf.nn.softmax(self.logits,
                                               name='softmax_tensor')
            }

            self.build()
            self.init_op = tf.global_variables_initializer()

    def build(self):
        self.cost = tf.losses.softmax_cross_entropy(onehot_labels=self.y_onehot, logits=self.logits)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(loss=self.cost)

    @staticmethod
    def train_mlp(sess, model, X_train, y_train, num_epochs=100, batch_size=1024):
        sess.run(model.init_op)

        for epoch in range(num_epochs):
            training_costs = []

            batch_generator = HelperFunction.create_batch_generator(
                X_train, y_train, batch_size=batch_size
            )

            for batch_X, batch_y in batch_generator:
                feed = {model.tf_x : batch_X, model.tf_y : batch_y}
                _, batch_cost = sess.run([model.train_op, model.cost], feed_dict=feed)
                training_costs.append(batch_cost)

            print('-- Epoch %2d '
                  'Avg Training Loss: %4f' % (
                      epoch+1, np.mean(training_costs)
                  ))
        return training_costs

    @staticmethod
    def predict_mlp(sess, model, X_test):
        feed = {model.tf_x : X_test}
        y_pred = sess.run(model.predictions['classes'],
                          feed_dict=feed)
        return y_pred