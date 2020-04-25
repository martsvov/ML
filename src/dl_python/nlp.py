import codecs
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys
import tensorflow as tf
import keras
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
from keras.models import Input, Model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

# data_dir = '../../datasets/jena_climate_2009_2016'
# fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
# f = open(fname)
# data = f.read()
# f.close()
# lines = data.split('\n')
# header = lines[0].split(',')
# lines = lines[1:]
#
# float_data = np.zeros((len(lines), len(header) - 1))
# for i, line in enumerate(lines):
#     values = [float(x) for x in line.split(',')[1:]]
#     float_data[i, :] = values

# temp = float_data[:, 1]
# # plt.plot(range(len(temp)), temp)
# plt.plot(range(1440), temp[:1440])
# plt.show()

# mean = float_data[:200000].mean(axis=0)
# float_data -= mean
# std = float_data[:200000].std(axis=0)
# float_data /= std
#
#
# def generator(data, lookback, delay, min_index, max_index,
#               shuffle=False, batch_size=128, step=6):
#     if max_index is None:
#         max_index = len(data) - delay - 1
#     i = min_index + lookback
#     while 1:
#         if shuffle:
#             rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
#         else:
#             if i + batch_size >= max_index:
#                 i = min_index + lookback
#             rows = np.arange(i, min(i + batch_size, max_index))
#         i += len(rows)
#         samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
#         targets = np.zeros((len(rows),))
#         for j, row in enumerate(rows):
#             indices = range(rows[j] - lookback, rows[j], step)
#             samples[j] = data[indices]
#             targets[j] = data[rows[j] + delay][1]
#
#         yield samples, targets
#
#
# lookback = 1440
# step = 6
# delay = 144
# batch_size = 128
#
# train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000, shuffle=True,
#                       step=step, batch_size=batch_size)
# val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, step=step,
#                     batch_size=batch_size)
# test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None, step=step,
#                      batch_size=batch_size)
#
# val_steps = (300000 - 200001 - lookback) // batch_size
# test_steps = (len(float_data) - 300001 - lookback) // batch_size


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

# model = Sequential()
# model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=20,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

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

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# max_features = 10000
# maxlen = 500
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
#
# # x_train = [x[::-1] for x in x_train]
# # x_test = [x[::-1] for x in x_test]
#
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# model = Sequential()
# model.add(layers.Embedding(max_features, 32))
# model.add(layers.Bidirectional(layers.LSTM(32)))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=128,
#                     validation_split=0.2)

# model = Sequential()
# model.add(layers.Embedding(max_features, 128, input_length=maxlen))
# model.add(layers.Conv1D(32, 7, activation='relu'))
# model.add(layers.MaxPooling1D(5))
# model.add(layers.Conv1D(32, 7, activation='relu'))
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(lr=1e-4),
#               loss='binary_crossentropy',
#               metrics=['acc'])
# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=128,
#                     validation_split=0.2)

# model = Sequential()
# model.add(layers.Conv1D(32, 5, activation='relu',
#                         input_shape=(None, float_data.shape[-1])))
# model.add(layers.MaxPooling1D(3))
# model.add(layers.Conv1D(32, 5, activation='relu'))
# model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
# model.add(layers.Dense(1))
# model.summary()
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=20,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)


# input_tensor = Input(shape=(64,))
# x = layers.Dense(32, activation='relu')(input_tensor)
# x = layers.Dense(32, activation='relu')(x)
# output_tensor = layers.Dense(10, activation='softmax')(x)
# model = Model(input_tensor, output_tensor)
#
# print(model.summary())

# text_vocabulary_size = 10000
# question_vocabulary_size = 10000
# answer_vocabulary_size = 500
#
# text_input = Input(shape=(None,), dtype='int32', name='text')
# embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
# encoded_text = layers.LSTM(32)(embedded_text)
#
# question_input = Input(shape=(None,), dtype='int32', name='question')
# embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
# encoded_question = layers.LSTM(16)(embedded_question)
#
# concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
# answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)
#
# model = Model([text_input, question_input], answer)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
#
#
# num_samples = 1000
# max_length = 100
#
# text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
# question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
#
# answers = np.zeros(shape=(num_samples, answer_vocabulary_size))
# indices = np.random.randint(0, answer_vocabulary_size, size=num_samples)
#
# for i, x in enumerate(answers):
#     x[indices[i]] = 1
#
# model.fit([text, question], answers, epochs=10, batch_size=128)
# model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)

# vocabulary_size = 50000
# num_income_groups = 10
# posts_input = Input(shape=(None,), dtype='int32', name='posts')
# embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
# x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
# x = layers.MaxPooling1D(5)(x)
# x = layers.Conv1D(256, 5, activation='relu')(x)
# x = layers.Conv1D(256, 5, activation='relu')(x)
# x = layers.MaxPooling1D(5)(x)
# x = layers.Conv1D(256, 5, activation='relu')(x)
# x = layers.Conv1D(256, 5, activation='relu')(x)
# x = layers.GlobalMaxPooling1D()(x)
# x = layers.Dense(128, activation='relu')(x)
#
# age_prediction = layers.Dense(1, name='age')(x)
# income_prediction = layers.Dense(num_income_groups,
#                                  activation='softmax',
#                                  name='income')(x)
# gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
# model = Model(posts_input,
#               [age_prediction, income_prediction, gender_prediction])
#
# model.compile(optimizer='rmsprop',
#               loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
# model.compile(optimizer='rmsprop',
#               loss={'age': 'mse',
#                     'income': 'categorical_crossentropy',
#                     'gender': 'binary_crossentropy'})
# model.compile(optimizer='rmsprop',
#               loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
#               loss_weights=[0.25, 1., 10.])
#
# branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)
# branch_b = layers.Conv2D(128, 1, activation='relu')(x)
# branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
# branch_c = layers.AveragePooling2D(3, strides=2)(x)
# branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
# branch_d = layers.Conv2D(128, 1, activation='relu')(x)
# branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
# branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
# output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
#
# lstm = layers.LSTM(32)
# left_input = Input(shape=(None, 128))
# left_output = lstm(left_input)
# right_input = Input(shape=(None, 128))
# right_output = lstm(right_input)
#
# merged = layers.concatenate([left_output, right_output], axis=-1)
# predictions = layers.Dense(1, activation='sigmoid')(merged)
#
# model = Model([left_input, right_input], predictions)
# model.fit([left_data, right_data], targets)

# max_features = 2000
# max_len = 500
#
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# x_train = sequence.pad_sequences(x_train, maxlen=max_len)
# x_test = sequence.pad_sequences(x_test, maxlen=max_len)
#
# model = Sequential()
# model.add(layers.Embedding(max_features, 128,
#                            input_length=max_len,
#                            name='embed'))
# model.add(layers.Conv1D(32, 7, activation='relu'))
# model.add(layers.MaxPooling1D(5))
# model.add(layers.Conv1D(32, 7, activation='relu'))
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(1))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])
#
# callbacks = [
#     keras.callbacks.TensorBoard(
#         log_dir='..\..\datasets\my_log_dir',
#         histogram_freq=1,
#         embeddings_freq=1,
#     )
# ]
# history = model.fit(x_train, y_train,
#                     epochs=20,
#                     batch_size=128,
#                     validation_split=0.2,
#                     callbacks=callbacks)

# path = keras.utils.get_file(
#     'nietzsche.txt',
#     origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
#
# text = open(path).read().lower()
# # print('Corpus length:', len(text))
#
# maxlen = 60
# step = 3
# sentences = []
# next_chars = []
#
# for i in range(0, len(text) - maxlen, step):
#     sentences.append(text[i: i + maxlen])
#     next_chars.append(text[i + maxlen])
#
# print('Number of sequences:', len(sentences))
#
# chars = sorted(list(set(text)))
# print('Unique characters:', len(chars))
# char_indices = dict((char, chars.index(char)) for char in chars)
# print('Vectorization...')
#
# x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
# y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         x[i, t, char_indices[char]] = 1
#         y[i, char_indices[next_chars[i]]] = 1
#
# model = keras.models.Sequential()
# model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
# model.add(layers.Dense(len(chars), activation='softmax'))
#
# optimizer = keras.optimizers.RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)
#
#
# def sample(preds, temperature=1.0):
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
#
# for epoch in range(1, 60):
#     print('epoch', epoch)
#     model.fit(x, y, batch_size=128, epochs=1)
#     start_index = random.randint(0, len(text) - maxlen - 1)
#     generated_text = text[start_index: start_index + maxlen]
#     print('--- Generating with seed: "' + generated_text + '"')
#
#     for temperature in [0.2, 0.5, 1.0, 1.2]:
#         print('------ temperature:', temperature)
#         sys.stdout.write(generated_text)
#         for i in range(400):
#             sampled = np.zeros((1, maxlen, len(chars)))
#             for t, char in enumerate(generated_text):
#                 sampled[0, t, char_indices[char]] = 1.
#
#             preds = model.predict(sampled, verbose=0)[0]
#             next_index = sample(preds, temperature)
#             next_char = chars[next_index]
#
#             generated_text += next_char
#             generated_text = generated_text[1:]
#
#             sys.stdout.write(next_char)

# from keras.applications import inception_v3
# from keras import backend as K
#
# K.set_learning_phase(0)
# model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
#
# layer_contributions = {'mixed2': 0.2, 'mixed3': 3., 'mixed4': 2., 'mixed5': 1.5, }
#
# layer_dict = dict([(layer.name, layer) for layer in model.layers])
# loss = K.variable(0.)
# for layer_name in layer_contributions:
#     coeff = layer_contributions[layer_name]
#     activation = layer_dict[layer_name].output
#     scaling = K.prod(K.cast(K.shape(activation), 'float32'))
#     loss = loss + coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling
#
# dream = model.input
# grads = K.gradients(loss, dream)[0]
# grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
# outputs = [loss, grads]
# fetch_loss_and_grads = K.function([dream], outputs)
#
#
# def eval_loss_and_grads(x):
#     outs = fetch_loss_and_grads([x])
#     loss_value = outs[0]
#     grad_values = outs[1]
#     return loss_value, grad_values
#
#
# def gradient_ascent(x, iterations, step, max_loss=None):
#     for i in range(iterations):
#         loss_value, grad_values = eval_loss_and_grads(x)
#         if max_loss is not None and loss_value > max_loss:
#             break
#         print('...Loss value at', i, ':', loss_value)
#         x += step * grad_values
#     return x
#
#
# import scipy
# from keras.preprocessing import image
#
#
# def resize_img(img, size):
#     img = np.copy(img)
#     factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
#     return scipy.ndimage.zoom(img, factors, order=1)
#
#
# def save_img(img, fname):
#     pil_img = deprocess_image(np.copy(img))
#     scipy.misc.imsave(fname, pil_img)
#
#
# def preprocess_image(image_path):
#     img = image.load_img(image_path)
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = inception_v3.preprocess_input(img)
#     return img
#
#
# def deprocess_image(x):
#     if K.image_data_format() == 'channels_first':
#         x = x.reshape((3, x.shape[2], x.shape[3]))
#         x = x.transpose((1, 2, 0))
#     else:
#         x = x.reshape((x.shape[1], x.shape[2], 3))
#     x /= 2.
#     x += 0.5
#     x *= 255.
#     x = np.clip(x, 0, 255).astype('uint8')
#     return x
#
#
# step = 0.01
# num_octave = 3
# octave_scale = 1.4
# iterations = 20
# max_loss = 10.
#
# base_image_path = '../../datasets/foto.jpg'
# img = preprocess_image(base_image_path)
# original_shape = img.shape[1:3]
# successive_shapes = [original_shape]
#
# for i in range(1, num_octave):
#     shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
#     successive_shapes.append(shape)
#
# successive_shapes = successive_shapes[::-1]
# original_img = np.copy(img)
# shrunk_original_img = resize_img(img, successive_shapes[0])
#
# for shape in successive_shapes:
#     print('Processing image shape', shape)
#     img = resize_img(img, shape)
#     img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)
#     upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
#     same_size_original = resize_img(original_img, shape)
#     lost_detail = same_size_original - upscaled_shrunk_original_img
#     img += lost_detail
#     shrunk_original_img = resize_img(original_img, shape)
#     save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
#
# save_img(img, fname='final_dream.png')

# from keras.preprocessing.image import load_img, img_to_array
#
# target_image_path = 'D:\\Project\\ML\\datasets\\foto.jpg'
# style_reference_image_path = 'D:\\Project\\ML\\datasets\\vangog.jpg'
# width, height = load_img(target_image_path).size
# img_height = 400
# img_width = int(width * img_height / height)
#
# from keras.applications import vgg19
#
#
# def preprocess_image(image_path):
#     img = load_img(image_path, target_size=(img_height, img_width))
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = vgg19.preprocess_input(img)
#     return img
#
#
# def deprocess_image(x):
#     x[:, :, 0] += 103.939
#     x[:, :, 1] += 116.779
#     x[:, :, 2] += 123.68
#     x = x[:, :, ::-1]
#     x = np.clip(x, 0, 255).astype('uint8')
#     return x
#
#
# from keras import backend as K
#
# target_image = K.constant(preprocess_image(target_image_path))
# style_reference_image = K.constant(preprocess_image(style_reference_image_path))
# combination_image = K.placeholder((1, img_height, img_width, 3))
# input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)
# model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
# print('Model loaded.')
#
#
# def content_loss(base, combination):
#     return K.sum(K.square(combination - base))
#
#
# def gram_matrix(x):
#     features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
#     gram = K.dot(features, K.transpose(features))
#     return gram
#
#
# def style_loss(style, combination):
#     S = gram_matrix(style)
#     C = gram_matrix(combination)
#     channels = 3
#     size = img_height * img_width
#     return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
#
#
# def total_variation_loss(x):
#     a = K.square(
#         x[:, :img_height - 1, :img_width - 1, :] -
#         x[:, 1:, :img_width - 1, :])
#     b = K.square(
#         x[:, :img_height - 1, :img_width - 1, :] -
#         x[:, :img_height - 1, 1:, :])
#     return K.sum(K.pow(a + b, 1.25))
#
#
# outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# content_layer = 'block5_conv2'
# style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
# total_variation_weight = 1e-4
# style_weight = 1.
# content_weight = 0.025
# loss = K.variable(0.)
# layer_features = outputs_dict[content_layer]
# target_image_features = layer_features[0, :, :, :]
# combination_features = layer_features[2, :, :, :]
# loss = loss + content_weight * content_loss(target_image_features, combination_features)
#
# for layer_name in style_layers:
#     layer_features = outputs_dict[layer_name]
#     style_reference_features = layer_features[1, :, :, :]
#     combination_features = layer_features[2, :, :, :]
#     sl = style_loss(style_reference_features, combination_features)
#     loss = loss + (style_weight / len(style_layers)) * sl
# loss = loss + total_variation_weight * total_variation_loss(combination_image)
#
# grads = K.gradients(loss, combination_image)[0]
# fetch_loss_and_grads = K.function([combination_image], [loss, grads])
#
#
# class Evaluator(object):
#     def __init__(self):
#         self.loss_value = None
#         self.grads_values = None
#
#     def loss(self, x):
#         assert self.loss_value is None
#         x = x.reshape((1, img_height, img_width, 3))
#         outs = fetch_loss_and_grads([x])
#         loss_value = outs[0]
#         grad_values = outs[1].flatten().astype('float64')
#         self.loss_value = loss_value
#         self.grad_values = grad_values
#         return self.loss_value
#
#     def grads(self, x):
#         assert self.loss_value is not None
#         grad_values = np.copy(self.grad_values)
#         self.loss_value = None
#         self.grad_values = None
#         return grad_values
#
#
# evaluator = Evaluator()
#
# from scipy.optimize import fmin_l_bfgs_b
# from scipy.misc import imsave
# import time
#
# result_prefix = 'my_result'
# iterations = 20
# x = preprocess_image(target_image_path)
# x = x.flatten()
#
# for i in range(iterations):
#     print('Start of iteration', i)
#     start_time = time.time()
#     x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
#     print('Current loss value:', min_val)
#     img = x.copy().reshape((img_height, img_width, 3))
#     img = deprocess_image(img)
#     fname = result_prefix + '_at_iteration_%d.png' % i
#     imsave(fname, img)
#     print('Image saved as', fname)
#     end_time = time.time()
#     print('Iteration %d completed in %ds' % (i, end_time - start_time))
