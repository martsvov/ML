import codecs
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras import layers
from keras.optimizers import RMSprop


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# token_index = {}
# for sample in samples:
#     for word in sample.split():
#         if word not in token_index:
#             token_index[word] = len(token_index) + 1
#
# max_length = 10
# results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))
#
# for i, sample in enumerate(samples):
#     for j, word in list(enumerate(sample.split()))[:max_length]:
#         index = token_index.get(word)
#         results[i, j, index] = 1.
#
# print(results)

# tokenizer = Tokenizer(num_words=10)
# tokenizer.fit_on_texts(samples)
#
# sequences = tokenizer.texts_to_sequences(samples)
# one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
# word_index = tokenizer.word_index
#
# print('Found %s unique tokens.' % len(word_index))
# print(one_hot_results)

# dimensionality = 1000
# max_length = 10
# results = np.zeros((len(samples), max_length, dimensionality))
#
# for i, sample in enumerate(samples):
#     for j, word in list(enumerate(sample.split()))[:max_length]:
#         index = abs(hash(word)) % dimensionality
#         results[i, j, index] = 1.
#
# print(results)

# embedding_layer = Embedding(1000, 64)

# max_features = 10000
# maxlen = 20
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
#
# model = Sequential()
# model.add(Embedding(10000, 8, input_length=maxlen))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
#
# history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# imdb_dir = '../../datasets/aclImdb'
# train_dir = os.path.join(imdb_dir, 'train')
#
# labels = []
# texts = []
# for label_type in ['neg', 'pos']:
#     dir_name = os.path.join(train_dir, label_type)
#     for fname in os.listdir(dir_name):
#         if fname[-4:] == '.txt':
#             f = codecs.open(os.path.join(dir_name, fname), encoding='utf-8', mode='r')
#             texts.append(f.read())
#             f.close()
#             if label_type == 'neg':
#                 labels.append(0)
#             else:
#                 labels.append(1)
#
# maxlen = 100
# training_samples = 200
# validation_samples = 10000
# max_words = 10000
#
# tokenizer = Tokenizer(num_words=max_words)
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
# word_index = tokenizer.word_index
#
# print('Found %s unique tokens.' % len(word_index))
#
# data = pad_sequences(sequences, maxlen=maxlen)
# labels = np.asarray(labels)
#
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)
#
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
#
# x_train = data[:training_samples]
# y_train = labels[:training_samples]
# x_val = data[training_samples: training_samples + validation_samples]
# y_val = labels[training_samples: training_samples + validation_samples]
#
# glove_dir = '../../datasets/glove.6B'
# embeddings_index = {}
# f = codecs.open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8', mode='r')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
#
# print('Found %s word vectors.' % len(embeddings_index))
#
# embedding_dim = 100
# embedding_matrix = np.zeros((max_words, embedding_dim))
# for word, i in word_index.items():
#     if i < max_words:
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
#
# model = Sequential()
# model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# model.layers[0].set_weights([embedding_matrix])
# model.layers[0].trainable = False
#
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])
# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=32,
#                     validation_data=(x_val, y_val))
# model.save_weights('pre_trained_glove_model.h5')
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
#
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# test_dir = os.path.join(imdb_dir, 'test')
#
# labels = []
# texts = []
#
# for label_type in ['neg', 'pos']:
#     dir_name = os.path.join(test_dir, label_type)
#     for fname in sorted(os.listdir(dir_name)):
#         if fname[-4:] == '.txt':
#             f = open(os.path.join(dir_name, fname))
#     texts.append(f.read())
#     f.close()
#     if label_type == 'neg':
#         labels.append(0)
#     else:
#         labels.append(1)
#
# sequences = tokenizer.texts_to_sequences(texts)
# x_test = pad_sequences(sequences, maxlen=maxlen)
# y_test = np.asarray(labels)

# model = Sequential()
# model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])

# model.load_weights('pre_trained_glove_model.h5')
# model.evaluate(x_test, y_test)

# timesteps = 100
# input_features = 32
# output_features = 64
# inputs = np.random.random((timesteps, input_features))
# state_t = np.zeros((output_features,))
#
# W = np.random.random((output_features, input_features))
# U = np.random.random((output_features, output_features))
# b = np.random.random((output_features,))
#
# successive_outputs = []
#
# for input_t in inputs:
#     output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
#     successive_outputs.append(output_t)
#     state_t = output_t
#
# final_output_sequence = np.concatenate(successive_outputs, axis=0)

# max_features = 10000
# sequencemax_features = 10000
# maxlen = 500
# batch_size = 32
#
# print('Loading data...')
#
# (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
#
# print(len(input_train), 'train sequences')
# print(len(input_test), 'test sequences')
# print('Pad sequences (samples x time)')
#
# input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
# input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
#
# print('input_train shape:', input_train.shape)
# print('input_test shape:', input_test.shape)
#
# model = Sequential()
# model.add(Embedding(max_features, 32))
# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])
# history = model.fit(input_train, y_train,
#                     epochs=10,
#                     batch_size=128,
#                     validation_split=0.2)
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

data_dir = '../../datasets/jena_climate_2009_2016'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# temp = float_data[:, 1]
# # plt.plot(range(len(temp)), temp)
# plt.plot(range(1440), temp[:1440])
# plt.show()

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
        i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000, shuffle=True,
                      step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, step=step,
                    batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None, step=step,
                     batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size


# def evaluate_naive_method():
#     batch_maes = []
#     for step in range(val_steps):
#         samples, targets = next(val_gen)
#         preds = samples[:, -1, 1]
#         mae = np.mean(np.abs(preds - targets))
#         batch_maes.append(mae)
#     print(np.mean(batch_maes))
#
# evaluate_naive_method()


# model = Sequential()
# model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=20,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

# model = Sequential()
# model.add(layers.GRU(32,
#                      dropout=0.2,
#                      recurrent_dropout=0.2,
#                      input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=40,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

# model = Sequential()
# model.add(layers.GRU(32,
#                      dropout=0.1,
#                      recurrent_dropout=0.5,
#                      return_sequences=True,
#                      input_shape=(None, float_data.shape[-1])))
# model.add(layers.GRU(64, activation='relu',
#                      dropout=0.1,
#                      recurrent_dropout=0.5))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=40,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
