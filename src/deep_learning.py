import numpy as np


# def w_sum(a,b):
#     assert(len(a) == len(b))
#     output = 0
#     for i in range(len(a)):
#         output += (a[i] * b[i])
#     return output
#
#
# def vect_mat_mul(vect, matrix):
#     assert(len(vect) == len(matrix))
#     output = [0,0,0]
#     for i in range(len(vect)):
#         output[i] = w_sum(vect, matrix[i])
#     return output
#
#
# def neural_network(input, weights):
#     pred = vect_mat_mul(input, weights)
#     return pred
#
#
# weights = [[0.1, 0.1, -0.3],
#            [0.1, 0.2, 0.0],
#            [0.0, 1.3, 0.1]]
#
# toes = [8.5, 9.5, 9.9, 9.0]
# wlrec = [0.65, 0.8, 0.8, 0.9]
# nfans = [1.2, 1.3, 0.5, 1.0]
#
# input = [toes[0], wlrec[0], nfans[0]]
# pred = neural_network(input, weights)

# weight = 0.5
# goal_pred = 0.8
# input = 0.5
#
# for iteration in range(20):
#     pred = input * weight
#     error = (pred - goal_pred) ** 2
#     direction_and_amount = (pred - goal_pred) * input
#     weight = weight - direction_and_amount
#     print("Error:" + str(error) + " Prediction:" + str(pred))

# print(pred)

# np.random.seed(1)
#
# def relu(x):
#     return (x > 0) * x
#
# def relu2deriv(output):
#     return output > 0
#
#
# streetlights = np.array([[1, 0, 1],
#                          [0, 1, 1],
#                          [0, 0, 1],
#                          [1, 1, 1]])
# walk_vs_stop = np.array([[1, 1, 0, 0]]).T
# alpha = 0.2
# hidden_size = 4
# weights_0_1 = 2*np.random.random((3, hidden_size)) - 1
# weights_1_2 = 2*np.random.random((hidden_size, 1)) - 1
#
# for iteration in range(60):
#     layer_2_error = 0
#     for i in range(len(streetlights)):
#         layer_0 = streetlights[i:i+1]
#         layer_1 = relu(np.dot(layer_0, weights_0_1))
#         layer_2 = np.dot(layer_1, weights_1_2)
#         layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)
#         layer_2_delta = (layer_2 - walk_vs_stop[i:i+1])
#         layer_1_delta = layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_1)
#         weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
#         weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)
#     if iteration % 10 == 9:
#         print("Error:" + str(layer_2_error))

import sys
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[0:1000].reshape(1000, 28*28) / 255, y_train[0:1000])

one_hot_labels = np.zeros((len(labels), 10))
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels
test_images = x_test.reshape(len(x_test), 28*28) / 255
test_labels = np.zeros((len(y_test), 10))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1
np.random.seed(1)
np.seterr(all='raise', over='ignore')


def relu(x):
    # x = np.nan_to_num(x)
    return (x >= 0) * x


relu2deriv = lambda x: x > 0
alpha, iterations, hidden_size, pixels_per_image, num_labels = (0.005, 350, 40, 784, 10)
weights_0_1 = 0.2*np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    error, correct_cnt = (0.0, 0)
    for i in range(len(images)):
        layer_0 = images[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        # dropout_mask = np.random.randint(2, size=layer_1.shape)
        # layer_1 *= dropout_mask * 2
        layer_2 = np.dot(layer_1, weights_1_2)
        error += np.sum((labels[i:i+1] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))
        layer_2_delta = (labels[i:i+1] - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
        # layer_1_delta *= dropout_mask
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
sys.stdout.write(" I:" + str(j) +
                 " Error:" + str(error / float(len(images)))[0:5] +
                 " Correct:" + str(correct_cnt / float(len(images))))
