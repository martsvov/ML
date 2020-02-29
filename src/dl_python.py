import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
import tensorflow_core
import keras.backend.tensorflow_backend as ktf

# from tensorflow_core.python import keras
# print(keras.__version__)
# import keras
# print(keras.__version__)
# import tensorflow
# print(tensorflow.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# from tensorflow_core.python.client import device_lib
# print(device_lib.list_local_devices())
# print(tf.test.is_gpu_available())
# print(tf.test.is_built_with_cuda())

# (train_images_l, train_labels), (test_images, test_labels) = mnist.load_data()
#
# network = models.Sequential()
# network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.Dense(10, activation='softmax'))
#
# network.compile(optimizer='rmsprop',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
#
# train_images = train_images_l.reshape((60000, 28 * 28))
# train_images = train_images.astype('float32') / 255
# test_images = test_images.reshape((10000, 28 * 28))
# test_images = test_images.astype('float32') / 255
#
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
#
# network.fit(train_images, train_labels, epochs=5, batch_size=128)
# test_loss, test_acc = network.evaluate(test_images, test_labels)
# print('test_acc:', test_acc)
#
# # digit = train_images_l[4]
# # plt.imshow(digit, cmap=matplotlib.cm.binary)
# # plt.show()

# from keras.datasets import imdb
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# word_index = imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
        return results


# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)
#
# y_train = np.asarray(train_labels).astype('float32')
# y_test = np.asarray(test_labels).astype('float32')
#
# model = models.Sequential()
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=4, batch_size=512)
# results = model.evaluate(x_test, y_test)
# print(results)

from keras.datasets import reuters
from keras.utils.np_utils import to_categorical

# (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)
#
# y_train = to_categorical(train_labels)
# y_test = to_categorical(test_labels)
#
# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(46, activation='softmax'))
#
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# x_val = x_train[:1000]
# partial_x_train = x_train[1000:]
#
# y_val = y_train[:1000]
# partial_y_train = y_train[1000:]

# history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,
#                     validation_data=(x_val, y_val))
#
# history_dict = history.history

# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# epochs = range(1, len(loss_values) + 1)
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))
# results = model.evaluate(x_test, y_test)
#
# predictions = model.predict(x_test)
# print(np.sum(predictions[0]))

from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)


average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# print(average_mae_history)

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
