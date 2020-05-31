import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# housing = fetch_california_housing()
# data = housing.data
#
# scaler = StandardScaler()
# scaler.fit(data)
# data = scaler.transform(data)
# m, n = data.shape
# housing_data_plus_bias = np.c_[np.ones((m, 1)), data]
# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name=" X ")
# y = tf.constant(housing.target.reshape(- 1, 1), dtype=tf.float32, name=" y")
# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(XT, X)), XT), y)


# @tf.function
# def func(theta, X, y, n_epochs):
#     for epoch in range(n_epochs):
#         # with tf.GradientTape() as tape:
#         #     tape.watch(theta)
#         y_pred = tf.matmul(X, theta, name="predictions")
#         error = y_pred - y
#         mse = tf.reduce_mean(tf.square(error), name="mse")
#
#         gradients = 2 / m * tf.matmul(tf.transpose(X), error)
#         # gradients = tape.gradient(mse, theta)
#         theta = theta - learning_rate * gradients
#
#         # if epoch % 100 == 0:
#             # print("Эпоха", epoch, "MSE = ", mse.numpy())
#
#     return theta
#
#
# log_dir = "../datasets/my_log_dir"
# writer = tf.summary.create_file_writer(log_dir)
#
# n_epochs = 100
# learning_rate = 0.01
# # optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# theta = tf.Variable(tf.random.uniform([n + 1, 1], -1.0, 1.0), name="theta")
#
# tf.summary.trace_on(graph=True, profiler=True)
# theta = func(theta, X, y, n_epochs)
#
# with writer.as_default():
#     tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=log_dir)
#
# print(theta.numpy())

mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = X[:6000], X[6000:12000], y[:6000], y[6000:12000]
# print(X_train.shape)

# def input(dataset):
#     return dataset.images, dataset.labels.astype(np.int32)
#
#
# feature_columns = [tf.feature_column.numeric_column('x', shape=[784, ])]
#
# classifier = tf.estimator.DNNClassifier(
#     feature_columns=feature_columns,
#     hidden_units=[256, 32],
#     optimizer=tf.optimizers.Adam(1e-4),
#     n_classes=10,
#     dropout=0.1,
#     model_dir="./tmp/mnist_model"
# )
#
# train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
#     x={'x': X_train},
#     y=y_train.astype(int),
#     num_epochs=None,
#     batch_size=50,
#     shuffle=True
# )
#
# classifier.train(input_fn=train_input_fn, steps=100000)

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10


def neuron_layer(X_train, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X_train.shape[1])
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        init = tf.random.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        X = tf.constant(X_train, dtype=tf.float32, name="X")
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z


with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X_train, n_hidden1, name="hiddenl",
                           activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs")
    # hidden1 = tf.keras.layers.Dense(n_hidden1, name="hiddenl",
    #                                 activation='relu')
    # hidden2 = tf.keras.layers.Dense(n_hidden2, name="hidden2",
    #                                 activation='relu')
    # logits = tf.keras.layers.Dense(n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train.astype(int), logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    training_op = optimizer.minimize(loss)
