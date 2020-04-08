import sys

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
# import sys
# from keras.datasets import mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# images, labels = (x_train[0:1000].reshape(1000, 28*28) / 255, y_train[0:1000])
#
# one_hot_labels = np.zeros((len(labels), 10))
# for i, l in enumerate(labels):
#     one_hot_labels[i][l] = 1
# labels = one_hot_labels
# test_images = x_test.reshape(len(x_test), 28*28) / 255
# test_labels = np.zeros((len(y_test), 10))
# for i, l in enumerate(y_test):
#     test_labels[i][l] = 1
# np.random.seed(1)
# np.seterr(all='raise', over='ignore')
#
#
# def relu(x):
#     # x = np.nan_to_num(x)
#     return (x >= 0) * x
#
#
# relu2deriv = lambda x: x > 0
# alpha, iterations, hidden_size, pixels_per_image, num_labels = (0.005, 350, 40, 784, 10)
# weights_0_1 = 0.2*np.random.random((pixels_per_image, hidden_size)) - 0.1
# weights_1_2 = 0.2*np.random.random((hidden_size, num_labels)) - 0.1
#
# for j in range(iterations):
#     error, correct_cnt = (0.0, 0)
#     for i in range(len(images)):
#         layer_0 = images[i:i+1]
#         layer_1 = relu(np.dot(layer_0, weights_0_1))
#         # dropout_mask = np.random.randint(2, size=layer_1.shape)
#         # layer_1 *= dropout_mask * 2
#         layer_2 = np.dot(layer_1, weights_1_2)
#         error += np.sum((labels[i:i+1] - layer_2) ** 2)
#         correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))
#         layer_2_delta = (labels[i:i+1] - layer_2)
#         layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
#         # layer_1_delta *= dropout_mask
#         weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
#         weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
# sys.stdout.write(" I:" + str(j) +
#                  " Error:" + str(error / float(len(images)))[0:5] +
#                  " Correct:" + str(correct_cnt / float(len(images))))
# def tanh(x):
#     return np.tanh(x)
#
#
# def tanh2deriv(output):
#     return 1 - (output ** 2)
#
#
# def softmax(x):
#     temp = np.exp(x)
#     return temp / np.sum(temp, axis=1, keepdims=True)
#
#
# alpha, iterations = (2, 300)
# pixels_per_image, num_labels = (784, 10)
# batch_size = 128
# input_rows = 28
# input_cols = 28
# kernel_rows = 3
# kernel_cols = 3
# num_kernels = 16222
# hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels
# kernels = 0.02*np.random.random((kernel_rows*kernel_cols, num_kernels))-0.01
# weights_1_2 = 0.2*np.random.random((hidden_size, num_labels)) - 0.1
#
#
# def get_image_section(layer, row_from, row_to, col_from, col_to):
#     section = layer[:, row_from:row_to, col_from:col_to]
#     return section.reshape(-1, 1, row_to-row_from, col_to-col_from)
#
#
# for j in range(iterations):
#     correct_cnt = 0
#     for i in range(int(len(images) / batch_size)):
#         batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
#         layer_0 = images[batch_start:batch_end]
#         layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
#         # layer_0.shape
#         sects = list()
#         for row_start in range(layer_0.shape[1] - kernel_rows):
#             for col_start in range(layer_0.shape[2] - kernel_cols):
#                 sect = get_image_section(layer_0,
#                                          row_start,
#                                          row_start + kernel_rows,
#                                          col_start,
#                                          col_start + kernel_cols)
#                 sects.append(sect)
#
#         expanded_input = np.concatenate(sects, axis=1)
#         es = expanded_input.shape
#         flattened_input = expanded_input.reshape(es[0] * es[1], -1)
#         kernel_output = flattened_input.dot(kernels)
#         layer_1 = tanh(kernel_output.reshape(es[0], -1))
#         dropout_mask = np.random.randint(2, size=layer_1.shape)
#         layer_1 *= dropout_mask * 2
#         layer_2 = softmax(np.dot(layer_1, weights_1_2))
#         for k in range(batch_size):
#             labelset = labels[batch_start + k:batch_start + k + 1]
#             _inc = int(np.argmax(layer_2[k:k + 1]) == np.argmax(labelset))
#             correct_cnt += _inc
#
#         layer_2_delta = (labels[batch_start:batch_end] - layer_2) / (batch_size * layer_2.shape[0])
#         layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)
#         layer_1_delta *= dropout_mask
#         weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
#         lld_reshape = layer_1_delta.reshape(kernel_output.shape)
#         k_update = flattened_input.T.dot(lld_reshape)
#         kernels -= alpha * k_update
#
#     test_correct_cnt = 0
#     for i in range(len(test_images)):
#         layer_0 = test_images[i:i + 1]
#         layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
#         # layer_0.shape
#         sects = list()
#         for row_start in range(layer_0.shape[1] - kernel_rows):
#             for col_start in range(layer_0.shape[2] - kernel_cols):
#                 sect = get_image_section(layer_0,
#                                          row_start,
#                                          row_start + kernel_rows,
#                                          col_start,
#                                          col_start + kernel_cols)
#                 sects.append(sect)
#
#         expanded_input = np.concatenate(sects, axis=1)
#         es = expanded_input.shape
#         flattened_input = expanded_input.reshape(es[0] * es[1], -1)
#         kernel_output = flattened_input.dot(kernels)
#         layer_1 = tanh(kernel_output.reshape(es[0], -1))
#         layer_2 = np.dot(layer_1, weights_1_2)
#         test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))
#         if j % 1 == 0:
#             sys.stdout.write("I:" + str(j) +
#                              "Test-Acc:" + str(test_correct_cnt / float(len(test_images))) +
#                              "Train-Acc:" + str(correct_cnt/float(len(images))))

# f = open('../data/reviews.txt')
# raw_reviews = f.readlines()
# f.close()
#
# f = open('../data/labels.txt')
# raw_labels = f.readlines()
# f.close()
#
# tokens = list(map(lambda x:set(x.split(" ")), raw_reviews))
# vocab = set()
# for sent in tokens:
#     for word in sent:
#         if len(word)>0:
#             vocab.add(word)
# vocab = list(vocab)
#
# word2index = {}
# for i, word in enumerate(vocab):
#     word2index[word] = i
#
# input_dataset = list()
# for sent in tokens:
#     sent_indices = list()
#     for word in sent:
#         try:
#             sent_indices.append(word2index[word])
#         except:
#             ""
#     input_dataset.append(list(set(sent_indices)))
#
# target_dataset = list()
# for label in raw_labels:
#     if label == 'positive\n':
#         target_dataset.append(1)
#     else:
#         target_dataset.append(0)
#
# np.random.seed(1)
#
# from collections import Counter
# import math
# import random
#
#
# def sigmoid(x):
#     return 1/(1 + np.exp(-x))
#
#
# alpha, iterations = (0.01, 2)
# hidden_size = 100
# weights_0_1 = 0.2*np.random.random((len(vocab), hidden_size)) - 0.1
# weights_1_2 = 0.2*np.random.random((hidden_size, 1)) - 0.1
# correct, total = (0, 0)
#
# for iter in range(iterations):
#     for i in range(len(input_dataset)-1000):
#         x, y = (input_dataset[i], target_dataset[i])
#         layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
#         layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
#         layer_2_delta = layer_2 - y
#         layer_1_delta = layer_2_delta. dot(weights_1_2.T)
#         weights_0_1[x] -= layer_1_delta * alpha
#         weights_1_2 -= np.outer(layer_1, layer_2_delta) * alpha
#         if np.abs(layer_2_delta) < 0.5:
#             correct += 1
#         total += 1
#         if i % 10 == 9:
#             progress = str(i / float(len(input_dataset)))
#             sys.stdout.write('\rlter:' + str(iter)
#                              + ' Progress:' + progress[2:4]
#                              + '.' + progress[4:6]
#                              + '% Training Accuracy:'
#                              + str(correct/float(total)) + '%')
#     print()
#
# correct, total = (0, 0)
# for i in range(len(input_dataset)-1000, len(input_dataset)):
#     x = input_dataset[i]
#     y = target_dataset[i]
#     layer_l = sigmoid(np.sum(weights_0_1[x], axis=0))
#     layer_2 = sigmoid(np.dot(layer_l, weights_1_2))
#     if np.abs(layer_2 - y) < 0.5:
#         correct += 1
#     total += 1
#
# print("Test Accuracy:" + str(correct / float(total)))
#
# norms = np.sum(weights_0_1 * weights_0_1, axis=1)
# norms.resize(norms.shape[0], 1)
# normed_weights = weights_0_1 * norms
#
#
# def make_sent_vect(words):
#     indices = list(map(lambda x: word2index[x], filter(lambda x: x in word2index, words)))
#     return np.mean(normed_weights[indices], axis=0)
#
#
# reviews2vectors = list()
# for review in tokens:
#     reviews2vectors.append(make_sent_vect(review))
# reviews2vectors = np.array(reviews2vectors)
#
#
# def most_similar_reviews(review):
#     v = make_sent_vect(review)
#     scores = Counter()
#     for i, val in enumerate(reviews2vectors.dot(v)):
#         scores[i] = val
#     most_similar = list()
#     for idx, score in scores.most_common(3):
#         most_similar.append(raw_reviews[idx][0:40])
#     return most_similar
#
#
# print(most_similar_reviews(['boring', 'awful']))

# concatenated = list()
# input_dataset = list()
#
# for sent in tokens:
#     sent_indices = list()
#     for word in sent:
#         try:
#             sent_indices.append(word2index[word])
#             concatenated.append(word2index[word])
#         except:
#             ""
#     input_dataset.append(sent_indices)
# concatenated = np.array(concatenated)
#
# random.shuffle(input_dataset)
# alpha, iterations = (0.05, 2)
# hidden_size, window, negative = (50, 2, 5)
# weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
# weights_1_2 = np.random.rand(len(vocab), hidden_size) * 0.2
# layer_2_target = np.zeros(negative + 1)
# layer_2_target[0] = 1
#
#
# def similar(target='beautiful'):
#     target_index = word2index[target]
#     scores = Counter()
#     for word, index in word2index.items():
#         raw_difference = weights_0_1[index] - (weights_0_1[target_index])
#         squared_difference = raw_difference * raw_difference
#         scores[word] = -math.sqrt(sum(squared_difference))
#     return scores.most_common(10)
#
#
# for rev_i, review in enumerate(input_dataset * iterations):
#     for target_i in range(len(review)):
#         target_samples = [review[target_i]] + list(concatenated[(np.random.rand(negative) *
#                                                                  len(concatenated)).astype('int').tolist()])
#         left_context = review[max(0, target_i - window):target_i]
#         right_context = review[target_i+1:min(len(review), target_i + window)]
#         layer_1 = np.mean(weights_0_1[left_context + right_context], axis=0)
#         layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T))
#         layer_2_delta = layer_2 - layer_2_target
#         layer_1_delta = layer_2_delta.dot(weights_1_2[target_samples])
#         weights_0_1[left_context + right_context] -= layer_1_delta * alpha
#         weights_1_2[target_samples] -= np.outer(layer_2_delta, layer_1) * alpha
#
#     if rev_i % 250 == 0:
#        sys.stdout.write('\rProgress:' + str(rev_i / float(len(input_dataset) * iterations)))
#
# print(similar('terrible'))


# word_vects = dict()
# word_vects['yankees'] = np.array([[0., 0., 0.]])
# word_vects['bears'] = np.array([[0., 0., 0.]])
# word_vects['braves'] = np.array([[0., 0., 0.]])
# word_vects['red'] = np.array([[0., 0., 0.]])
# word_vects['sox'] = np.array([[0., 0., 0.]])
# word_vects['lose'] = np.array([[0., 0., 0.]])
# word_vects['defeat'] = np.array([[0., 0., 0.]])
# word_vects['beat'] = np.array([[0., 0., 0.]])
# word_vects['tie'] = np.array([[0., 0., 0.]])
#
# sent2output = np.random.rand(3, len(word_vects))
# identity = np.eye(3)
#
# layer_0 = word_vects['red']
# layer_1 = layer_0.dot(identity) + word_vects['sox']
# layer_2 = layer_1.dot(identity) + word_vects['defeat']
# pred = softmax(layer_2.dot(sent2output))
# print(pred)
#
# y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
# pred_delta = pred - y
# layer_2_delta = pred_delta.dot(sent2output.T)
# defeat_delta = layer_2_delta * 1
# layer_1_delta = layer_2_delta.dot(identity.T)
# sox_delta = layer_1_delta * 1
# layer_0_delta = layer_1_delta.dot(identity.T)
# alpha = 0.01
# word_vects['red'] -= layer_0_delta * alpha
# word_vects['sox'] -= sox_delta * alpha
# word_vects['defeat'] -= defeat_delta * alpha
# identity -= np.outer(layer_0, layer_1_delta) * alpha
# identity -= np.outer(layer_1, layer_2_delta) * alpha
# sent2output -= np.outer(layer_2, pred_delta) * alpha

import sys, random, math
from collections import Counter

f = open('../datasets/tasksv11/en/qa1_single-supporting-fact_train.txt', 'r')
raw = f.readlines()
f.close()
# tokens = list()
# for line in raw[0:1000]:
#     tokens.append(line.lower().replace("\n", "").split(" ")[1:])
#
# # print(tokens[0:3])
#
# vocab = set()
# for sent in tokens:
#     for word in sent:
#         vocab.add(word)
#
# vocab = list(vocab)
# word2index = {}
# for i, word in enumerate(vocab):
#     word2index[word] = i
#
#
# def words2indices(sentence):
#     idx = list()
#     for word in sentence:
#         idx.append(word2index[word])
#     return idx
#
#
# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)
#
#
# np.random.seed(1)
# embed_size = 10
# embed = (np.random.rand(len(vocab), embed_size) - 0.5) * 0.1
# recurrent = np.eye(embed_size)
# start = np.zeros(embed_size)
# decoder = (np.random.rand(embed_size, len(vocab)) - 0.5) * 0.1
# one_hot = np.eye(len(vocab))
#
#
# def predict(sent):
#     layers = list()
#     layer = dict()
#     layer['hidden'] = start
#     layers.append(layer)
#     loss = 0
#
#     for target_i in range(len(sent)):
#         layer = dict()
#         layer['pred'] = softmax(layers[-1]['hidden'].dot(decoder))
#         loss += -np.log(layer['pred'][sent[target_i]])
#         layer['hidden'] = layers[-1]['hidden'].dot(recurrent) + embed[sent[target_i]]
#         layers.append(layer)
#     return layers, loss
#
#
# for iter in range(30000):
#     alpha = 0.001
#     sent = words2indices(tokens[iter % len(tokens)][1:])
#     layers, loss = predict(sent)
#     for layer_idx in reversed(range(len(layers))):
#         layer = layers[layer_idx]
#         target = sent[layer_idx-1]
#         if layer_idx > 0:
#             layer['output_delta'] = layer['pred'] - one_hot[target]
#             new_hidden_delta = layer['output_delta'].dot(decoder.transpose())
#
#             if layer_idx == len(layers)-1:
#                 layer['hidden_delta'] = new_hidden_delta
#             else:
#                 layer['hidden_delta'] = new_hidden_delta + layers[layer_idx + 1]['hidden_delta']\
#                     .dot(recurrent.transpose())
#         else:
#             layer['hidden_delta'] = layers[layer_idx + 1]['hidden_delta'].dot(recurrent.transpose())
#
#     start -= layers[0]['hidden_delta'] * alpha / float(len(sent))
#     for layer_idx, layer in enumerate(layers[1:]):
#         decoder -= np.outer(layers[layer_idx]['hidden'], layer['output_delta']) * alpha / float(len(sent))
#         embed_idx = sent[layer_idx]
#         embed[embed_idx] -= layers[layer_idx]['hidden_delta'] * alpha / float(len(sent))
#         recurrent -= np.outer(layers[layer_idx]['hidden'], layer['hidden_delta']) * alpha / float(len(sent))
#
#     if iter % 1000 == 0:
#         print("Perplexity:" + str(np.exp(loss/len(sent))))
#
#
# sent_index = 4
# l, _ = predict(words2indices(tokens[sent_index]))
# print(tokens[sent_index])
#
# for i, each_layer in enumerate(l[1:-1]):
#     input = tokens[sent_index][i]
#     true = tokens[sent_index][i+1]
#     pred = vocab[each_layer['pred'].argmax()]
#     print("Prev Input:" + input + (' ' * (12 - len(input))) +
#           "True:" + true + (" " * (15 - len(true))) + "Pred:" + pred)
from sklearn.impute import SimpleImputer


# class Tensor(object):
#     def __init__(self, data,
#                  autograd=False,
#                  creators=None,
#                  creation_op=None,
#                  id=None):
#         self.data = np.array(data)
#         self.creators = creators
#         self.creation_op = creation_op
#         self.grad = None
#         self.index_select_indices = None
#         self.autograd = autograd
#         self.children = {}
#         if id is None:
#             id = np.random.randint(0, 100000)
#             self.id = id
#         if creators is not None:
#             for c in creators:
#                 if self.id not in c.children:
#                     c.children[self.id] = 1
#                 else:
#                     c.children[self.id] += 1
#
#     def sigmoid(self):
#         if self.autograd:
#             return Tensor(1 / (1 + np.exp(-self.data)),
#                           autograd=True,
#                           creators=[self],
#                           creation_op="sigmoid")
#         return Tensor(1 / (1 + np.exp(-self.data)))
#
#     def tanh(self):
#         if self.autograd:
#             return Tensor(np.tanh(self.data),
#                           autograd=True,
#                           creators=[self],
#                           creation_op="tanh")
#         return Tensor(np.tanh(self.data))
#
#     def index_select(self, indices):
#         if self.autograd:
#             new = Tensor(self.data[indices.data],
#                          autograd=True,
#                          creators=[self],
#                          creation_op="index_select")
#             new.index_select_indices = indices
#             return new
#         return Tensor(self.data[indices.data])
#
#     def all_children_grads_accounted_for(self):
#         for id, ent in self.children.items():
#             if ent != 0:
#                 return False
#         return True
#
#     def backward(self, grad=None, grad_origin=None):
#         if self.autograd:
#             if grad is None:
#                 grad = Tensor(np.ones_like(self.data))
#             if grad_origin is not None:
#                 if self.children[grad_origin.id] == 0:
#                     raise Exception("cannot backprop more than once")
#                 else:
#                     self.children[grad_origin.id] -= 1
#
#         if self.grad is None:
#             self.grad = grad
#         else:
#             self.grad += grad
#
#         if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):
#             if self.creation_op == "add":
#                 self.creators[0].backward(self.grad, self)
#                 self.creators[1].backward(self.grad, self)
#
#             if self.creation_op == "neg":
#                 self.creators[0].backward(self.grad.__neg__())
#
#             if self.creation_op == "sub":
#                 new = Tensor(self.grad.data)
#                 self.creators[0].backward(new, self)
#                 new = Tensor(self.grad.__neg__().data)
#                 self.creators[1].backward(new, self)
#
#             if self.creation_op == "mul":
#                 new = self.grad * self.creators[1]
#                 self.creators[0].backward(new, self)
#                 new = self.grad * self.creators[0]
#                 self.creators[1].backward(new, self)
#
#             if self.creation_op == "mm":
#                 act = self.creators[0]
#                 weights = self.creators[1]
#                 new = self.grad.mm(weights.transpose())
#                 act.backward(new)
#                 new = self.grad.transpose().mm(act).transpose()
#                 weights.backward(new)
#
#             if self.creation_op == "transpose":
#                 self.creators[0].backward(self.grad.transpose())
#
#             if self.creation_op == "sigmoid":
#                 ones = Tensor(np.ones_like(self.grad.data))
#                 self.creators[0].backward(self.grad * (self * (ones - self)))
#
#             if self.creation_op == "tanh":
#                 ones = Tensor(np.ones_like(self.grad.data))
#                 self.creators[0].backward(self.grad * (ones - (self * self)))
#
#             if self.creation_op == "index_select":
#                 new_grad = np.zeros_like(self.creators[0].data)
#                 indices_ = self.index_select_indices.data.flatten()
#                 grad_ = grad.data.reshape(len(indices_), -1)
#                 for i in range(len(indices_)):
#                     new_grad[indices_[i]] += grad_[i]
#                 self.creators[0].backward(Tensor(new_grad))
#
#             if self.creation_op == "cross_entropy":
#                 dx = self.softmax_output - self.target_dist
#                 self.creators[0].backward(Tensor(dx))
#
#             if "sum" in self.creation_op:
#                 dim = int(self.creation_op.split("_")[1])
#                 ds = self.creators[0].data.shape[dim]
#                 self.creators[0].backward(self.grad.expand(dim, ds))
#
#             if "expand" in self.creation_op:
#                 dim = int(self.creation_op.split("_")[1])
#                 self.creators[0].backward(self.grad.sum(dim))
#
#     def __add__(self, other):
#         if self.autograd and other.autograd:
#             return Tensor(self.data + other.data, autograd=True, creators=[self, other], creation_op="add")
#         return Tensor(self.data + other.data)
#
#     def __neg__(self):
#         if self.autograd:
#             return Tensor(self.data * -1,
#                           autograd=True,
#                           creators=[self],
#                           creation_op="neg")
#         return Tensor(self.data * -1)
#
#     def __sub__(self, other):
#         if self.autograd and other.autograd:
#             return Tensor(self.data - other.data, autograd=True, creators=[self, other], creation_op="sub")
#         return Tensor(self.data - other.data)
#
#     def __mul__(self, other):
#         if self.autograd and other.autograd:
#             return Tensor(self.data * other.data,
#                           autograd=True,
#                           creators=[self, other],
#                           creation_op="mul")
#
#         return Tensor(self.data * other.data)
#
#     def sum(self, dim):
#         if self.autograd:
#             return Tensor(self.data.sum(dim),
#                           autograd=True,
#                           creators=[self],
#                           creation_op="sum_" + str(dim))
#
#         return Tensor(self.data.sum(dim))
#
#     def expand(self, dim, copies):
#         trans_cmd = list(range(0, len(self.data.shape)))
#         trans_cmd.insert(dim, len(self.data.shape))
#         new_shape = list(self.data.shape) + [copies]
#         new_data = self.data.repeat(copies).reshape(new_shape)
#         new_data = new_data.transpose(trans_cmd)
#         if self.autograd:
#             return Tensor(new_data,
#                           autograd=True,
#                           creators=[self],
#                           creation_op="expand_" + str(dim))
#         return Tensor(new_data)
#
#     def transpose(self):
#         if self.autograd:
#             return Tensor(self.data.transpose(),
#                           autograd=True,
#                           creators=[self],
#                           creation_op="transpose")
#
#         return Tensor(self.data.transpose())
#
#     def mm(self, x):
#         if self.autograd:
#             return Tensor(self.data.dot(x.data),
#                           autograd=True,
#                           creators=[self, x],
#                           creation_op="mm")
#
#         return Tensor(self.data.dot(x.data))
#
#     def cross_entropy(self, target_indices):
#         temp = np.exp(self.data)
#
#         softmax_output = temp / np.sum(temp,
#                                        axis=len(self.data.shape) - 1,
#                                        keepdims=True)
#         t = target_indices.data.flatten()
#         p = softmax_output.reshape(len(t), -1)
#         target_dist = np.eye(p.shape[1])[t]
#         loss = -(np.log(p) * (target_dist)).sum(1).mean()
#         if self.autograd:
#             out = Tensor(loss,
#                          autograd=True,
#                          creators=[self],
#                          creation_op="cross_entropy")
#             out.softmax_output = softmax_output
#             out.target_dist = target_dist
#             return out
#         return Tensor(loss)
#
#     def __repr__(self):
#         return str(self.data.__repr__())
#
#     def __str__(self):
#         return str(self.data.__str__())
#
#
# class SGD(object):
#     def __init__(self, parameters, alpha=0.1):
#         self.parameters = parameters
#         self.alpha = alpha
#
#     def zero(self):
#         for p in self.parameters:
#             p.grad.data *= 0
#
#     def step(self, zero=True):
#         for p in self.parameters:
#             p.data -= p.grad.data * self.alpha
#             if zero:
#                 p.grad.data *= 0
#
#
# class Layer(object):
#     def __init__(self):
#         self.parameters = list()
#
#     def get_parameters(self):
#         return self.parameters
#
#
# class Linear(Layer):
#     def __init__(self, n_inputs, n_outputs):
#         super().__init__()
#         W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
#         self.weight = Tensor(W, autograd=True)
#         self.bias = Tensor(np.zeros(n_outputs), autograd=True)
#         self.parameters.append(self.weight)
#         self.parameters.append(self.bias)
#
#     def forward(self, input):
#         return input.mm(self.weight) + self.bias.expand(0, len(input.data))
#
#
# class Sequential(Layer):
#     def __init__(self, layers=list()):
#         super().__init__()
#         self.layers = layers
#
#     def add(self, layer):
#         self.layers.append(layer)
#
#     def forward(self, input):
#         for layer in self.layers:
#             input = layer.forward(input)
#         return input
#
#     def get_parameters(self):
#         params = list()
#         for l in self.layers:
#             params += l.get_parameters()
#         return params
#
#
# class MSELoss(Layer):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, pred, target):
#         return ((pred - target) * (pred - target)).sum(0)
#
#
# class Tanh(Layer):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input):
#         return input.tanh()
#
#
# class Sigmoid(Layer):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input):
#         return input.sigmoid()
#
#
# class Embedding(Layer):
#     def __init__(self, vocab_size, dim):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.dim = dim
#         weight = (np.random.rand(vocab_size, dim) - 0.5) / dim
#         self.weight = Tensor(weight, autograd=True)
#         self.parameters.append(self.weight)
#
#     def forward(self, input):
#         return self.weight.index_select(input)
#
#
# class CrossEntropyLoss(object):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input, target):
#         return input.cross_entropy(target)
#
#
# class RNNCell(Layer):
#     def __init__(self, n_inputs, n_hidden, n_output, activation='sigmoid'):
#         super().__init__()
#         self.n_inputs = n_inputs
#         self.n_hidden = n_hidden
#         self.n_output = n_output
#         if activation == 'sigmoid':
#             self.activation = Sigmoid()
#         elif activation == 'tanh':
#             self.activation == Tanh()
#         else:
#             raise Exception("Non-linearity not found")
#         self.w_ih = Linear(n_inputs, n_hidden)
#         self.w_hh = Linear(n_hidden, n_hidden)
#         self.w_ho = Linear(n_hidden, n_output)
#         self.parameters += self.w_ih.get_parameters()
#         self.parameters += self.w_hh.get_parameters()
#         self.parameters += self.w_ho.get_parameters()
#
#     def forward(self, input, hidden):
#         from_prev_hidden = self.w_hh.forward(hidden)
#         combined = self.w_ih.forward(input) + from_prev_hidden
#         new_hidden = self.activation.forward(combined)
#         output = self.w_ho.forward(new_hidden)
#         return output, new_hidden
#
#     def init_hidden(self, batch_size=1):
#         return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)


# np.random.seed(0)
#
# data = Tensor(np.array([1, 2, 1, 2]), autograd=True)
# target = Tensor(np.array([0, 1, 0, 1]), autograd=True)
#
# model = Sequential([Embedding(3, 3), Tanh(), Linear(3, 4)])
# criterion = CrossEntropyLoss()
# optim = SGD(parameters=model.get_parameters(), alpha=0.1)
#
# for i in range(10):
#     pred = model.forward(data)
#     loss = criterion.forward(pred, target)
#     loss.backward(Tensor(np.ones_like(loss.data)))
#     optim.step()
#
#     print(loss)

# tokens = list()
# for line in raw[0:1000]:
#     tokens.append(line.lower().replace("\n","").split(" ")[1:])
#
# new_tokens = list()
# for line in tokens:
#     new_tokens.append(['-'] * (6 - len(line)) + line)
# tokens = new_tokens
#
# vocab = set()
# for sent in tokens:
#     for word in sent:
#         vocab.add(word)
#
# vocab = list(vocab)
#
# word2index = {}
# for i, word in enumerate(vocab):
#     word2index[word] = i
#
#
# def words2indices(sentence):
#     idx = list()
#     for word in sentence:
#         idx.append(word2index[word])
#     return idx
#
#
# indices = list()
# for line in tokens:
#     idx = list()
#     for w in line:
#         idx.append(word2index[w])
#     indices.append(idx)
#
# data = np.array(indices)
#
# embed = Embedding(vocab_size=len(vocab), dim=16)
# model = RNNCell(n_inputs=16, n_hidden=16, n_output=len(vocab))
# criterion = CrossEntropyLoss()
# params = model.get_parameters() + embed.get_parameters()
# optim = SGD(parameters=params, alpha=0.05)
#
# for iter in range(1000):
#     batch_size = 100
#     total_loss = 0
#     hidden = model.init_hidden(batch_size=batch_size)
#
#     for t in range(5):
#         input = Tensor(data[0:batch_size, t], autograd=True)
#         rnn_input = embed.forward(input=input)
#         output, hidden = model.forward(input=rnn_input, hidden=hidden)
#
#     target = Tensor(data[0:batch_size, t+1], autograd=True)
#     loss = criterion.forward(output, target)
#     loss.backward()
#     optim.step()
#     total_loss += loss.data
#
#     if iter % 200 == 0:
#         p_correct = (target.data == np.argmax(output.data,axis=1)).mean()
#         print_loss = total_loss / (len(data)/batch_size)
#         print("Loss:", print_loss,"% Correct", p_correct)
#
# batch_size = 1
# hidden = model.init_hidden(batch_size=batch_size)
#
# for t in range(5):
#     input = Tensor(data[0:batch_size, t], autograd=True)
#     rnn_input = embed.forward(input=input)
#     output, hidden = model.forward(input=rnn_input, hidden=hidden)
#
# target = Tensor(data[0:batch_size, t+1], autograd=True)
# loss = criterion.forward(output, target)
#
# ctx = ""
# for idx in data[0:batch_size][0][0:-1]:
#     ctx += vocab[idx] + " "
#
# print("Context:", ctx)
# print("Pred:", vocab[output.data.argmax()])

# np.random.seed(0)
# f = open("../datasets/shaker")
# raw = f.read()
# f.close()
#
# vocab = list(set(raw))
# word2index = {}
# for i, word in enumerate(vocab):
#     word2index[word] = i
# indices = np.array(list(map(lambda x: word2index[x], raw)))
#
# embed = Embedding(vocab_size=len(vocab), dim=512)
# model = RNNCell(n_inputs=512, n_hidden=512, n_output=len(vocab))
# criterion = CrossEntropyLoss()
# optim = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.05)
#
# batch_size = 32
# bptt = 16
# n_batches = int((indices.shape[0] / batch_size))
#
# trimmed_indices = indices[:n_batches*batch_size]
# batched_indices = trimmed_indices.reshape(batch_size, n_batches)
# batched_indices = batched_indices.transpose()
# input_batched_indices = batched_indices[0:-1]
# target_batched_indices = batched_indices[1:]
# n_bptt = int(((n_batches-1) / bptt))
# input_batches = input_batched_indices[:n_bptt*bptt]
# input_batches = input_batches.reshape(n_bptt,bptt,batch_size)
# target_batches = target_batched_indices[:n_bptt*bptt]
# target_batches = target_batches.reshape(n_bptt, bptt, batch_size)
#
#
# def generate_sample(n=30, init_char=' '):
#     s = ""
#     hidden = model.init_hidden(batch_size=1)
#     input = Tensor(np.array([word2index[init_char]]))
#     for i in range(n):
#         rnn_input = embed.forward(input)
#         output, hidden = model.forward(input=rnn_input, hidden=hidden)
#         output.data *= 10
#         temp_dist = output.softmax() #???
#         temp_dist /= temp_dist.sum() #???
#         m = (temp_dist > np.random.rand()).argmax()
#         c = vocab[m]
#         input = Tensor(np.array([m]))
#         s += c
#     return s
#
#
# def train(iterations=100):
#     for iter in range(iterations):
#         total_loss = 0
#         n_loss = 0
#         hidden = model.init_hidden(batch_size=batch_size)
#         for batch_i in range(len(input_batches)):
#             hidden = Tensor(hidden.data, autograd=True)
#             loss = None
#             losses = list()
#             for t in range(bptt):
#                 input = Tensor(input_batches[batch_i][t], autograd=True)
#                 rnn_input = embed.forward(input=input)
#                 output, hidden = model.forward(input=rnn_input, hidden=hidden)
#                 target = Tensor(target_batches[batch_i][t], autograd=True)
#                 batch_loss = criterion.forward(output, target)
#                 losses.append(batch_loss)
#                 if t == 0:
#                     loss = batch_loss
#                 else:
#                     loss = loss + batch_loss
#             for loss in losses:
#                 ""
#             loss.backward()
#             optim.step()
#             total_loss += loss.data
#             log = "\r Iter:" + str(iter)
#             log += " - Batch "+str(batch_i+1)+"/"+str(len(input_batches))
#             log += " - Loss:" + str(np.exp(total_loss / (batch_i+1)))
#             if batch_i == 0:
#                 log += " - " + generate_sample(70, '\n').replace("\n", " ")
#             if batch_i % 10 == 0 or batch_i-1 == len(input_batches):
#                 sys.stdout.write(log)
#         optim.alpha *= 0.99
#         print()
#
# train()
# print(generate_sample(n=2000, init_char='\n'))

import numpy as np
from collections import Counter
import random
import sys
import codecs

np.random.seed(12345)
with codecs.open('../datasets/spam.txt', "r", encoding='utf-8', errors='ignore') as f:
    raw = f.readlines()

vocab, spam, ham = (set(["<unk>"]), list(), list())
for row in raw:
    spam.append(set(row[:-2].split(" ")))

for word in spam[-1]:
    vocab.add(word)

with codecs.open('../datasets/ham.txt', "r", encoding='utf-8', errors='ignore') as f:
    raw = f.readlines()

for row in raw:
    ham.append(set(row[:-2].split(" ")))
    for word in ham[-1]:
        vocab.add(word)

vocab, w2i = (list(vocab), {})
for i, w in enumerate(vocab):
    w2i[w] = i


def to_indices(input, l=500):
    indices = list()
    for line in input:
        if len(line) < l:
            line = list(line) + ["<unk>"] * (l - len(line))
            idxs = list()
            for word in line:
                idxs.append(w2i[word])
            indices.append(idxs)
    return indices
