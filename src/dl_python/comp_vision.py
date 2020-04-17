import os
import shutil
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
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)


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


# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((len(sequences), dimension))
#     for i, sequence in enumerate(sequences):
#         results[i, sequence] = 1
#         return results


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

# from keras.datasets import boston_housing
# (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
#
# mean = train_data.mean(axis=0)
# train_data -= mean
# std = train_data.std(axis=0)
# train_data /= std
# test_data -= mean
# test_data /= std
#
#
# def build_model():
#     model = models.Sequential()
#     model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(1))
#     model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
#     return model
#
#
# k = 4
# num_val_samples = len(train_data) // k
# num_epochs = 100
# all_mae_histories = []
#
# for i in range(k):
#     print('processing fold #', i)
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
#     partial_train_data = np.concatenate([train_data[:i * num_val_samples],
#                                          train_data[(i + 1) * num_val_samples:]], axis=0)
#     partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
#                                             train_targets[(i + 1) * num_val_samples:]], axis=0)
#     model = build_model()
#     history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
#                         epochs=num_epochs, batch_size=1, verbose=0)
#     mae_history = history.history['val_mae']
#     all_mae_histories.append(mae_history)
#
#
# average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# # print(average_mae_history)
#
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train_images = train_images.reshape((60000, 28, 28, 1))
# train_images = train_images.astype('float32') / 255
# test_images = test_images.reshape((10000, 28, 28, 1))
# test_images = test_images.astype('float32') / 255
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=5, batch_size=64)
# test_loss, test_acc = model.evaluate(test_images, test_labels)
#
# print(test_acc)

# original_dataset_dir = '../datasets/dogs-vs-cats/train'
base_dir = '../datasets/dogs-vs-cats_small'
# # os.mkdir(base_dir)
#
train_dir = os.path.join(base_dir, 'train')
# # os.mkdir(train_dir)
#
validation_dir = os.path.join(base_dir, 'validation')
# # os.mkdir(validation_dir)
#
test_dir = os.path.join(base_dir, 'test')
# # os.mkdir(test_dir)
#
train_cats_dir = os.path.join(train_dir, 'cats')
# # os.mkdir(train_cats_dir)
#
train_dogs_dir = os.path.join(train_dir, 'dogs')
# # os.mkdir(train_dogs_dir)
#
validation_cats_dir = os.path.join(validation_dir, 'cats')
# # os.mkdir(validation_cats_dir)
#
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# # os.mkdir(validation_dogs_dir)
#
test_cats_dir = os.path.join(test_dir, 'cats')
# # os.mkdir(test_cats_dir)
#
test_dogs_dir = os.path.join(test_dir, 'dogs')
# # os.mkdir(test_dogs_dir)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
# print('total training cat images:', len(os.listdir(train_cats_dir)))
# print('total training dog images:', len(os.listdir(train_dogs_dir)))
# print('total validation cat images:', len(os.listdir(validation_cats_dir)))
# print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
# print('total test cat images:', len(os.listdir(test_cats_dir)))
# print('total test dog images:', len(os.listdir(test_dogs_dir)))

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import load_model

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(train_dir,
#                                                     target_size=(150, 150),
#                                                     batch_size=20,
#                                                     class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(validation_dir,
#                                                         target_size=(150, 150),
#                                                         batch_size=20,
#                                                         class_mode='binary')
#
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=30,
#     validation_data=validation_generator,
#     validation_steps=50)
#
# model.save('cats_and_dogs_small_1.h5')

# datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')
#
# fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
# img_path = fnames[3]
# img = image.load_img(img_path, target_size=(150, 150))
# x = image.img_to_array(img)
# x = x.reshape((1,) + x.shape)
#
# i = 0
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i += 1
#     if i % 4 == 0:
#         break
#
# plt.show()

# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='binary')
#
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=50)
#
# model.save('cats_and_dogs_small_2.h5')


# from keras.applications import VGG16
# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#
# datagen = ImageDataGenerator(rescale=1./255)
# batch_size = 20
#
#
# def extract_features(directory, sample_count):
#     features = np.zeros(shape=(sample_count, 4, 4, 512))
#     labels = np.zeros(shape=(sample_count))
#     generator = datagen.flow_from_directory(
#         directory,
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')
#     i = 0
#     for inputs_batch, labels_batch in generator:
#         features_batch = conv_base.predict(inputs_batch)
#         features[i * batch_size: (i + 1) * batch_size] = features_batch
#         labels[i * batch_size: (i + 1) * batch_size] = labels_batch
#         i += 1
#         if i * batch_size >= sample_count:
#             break
#     return features, labels
#
#
# train_features, train_labels = extract_features(train_dir, 2000)
# validation_features, validation_labels = extract_features(validation_dir, 1000)
# test_features, test_labels = extract_features(test_dir, 1000)
#
# train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
# validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
# test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
#
# model = models.Sequential()
# model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
#               loss='binary_crossentropy',
#               metrics=['acc'])
# history = model.fit(train_features, train_labels,
#                     epochs=30,
#                     batch_size=20,
#                     validation_data=(validation_features, validation_labels))
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
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# model = load_model('../datasets/cats_and_dogs_small_2.h5')
#
# img_path = '../datasets/dogs-vs-cats_small/test/cats/cat.1700.jpg'
# img = image.load_img(img_path, target_size=(150, 150))
# img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor /= 255

# print(img_tensor.shape)
# plt.imshow(img_tensor[0])
# plt.show()

# layer_outputs = [layer.output for layer in model.layers[:8]]
# activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(img_tensor)
#
# # first_layer_activation = activations[0]
# # plt.matshow(first_layer_activation[0, :, :, 2], cmap='viridis')
# # plt.matshow(first_layer_activation[0, :, :, 10], cmap='viridis')
#
# layer_names = []
# for layer in model.layers[:8]:
#     layer_names.append(layer.name)
#
# images_per_row = 16
# for layer_name, layer_activation in zip(layer_names, activations):
#     n_features = layer_activation.shape[-1]
#     size = layer_activation.shape[1]
#     n_cols = n_features // images_per_row
#     display_grid = np.zeros((size * n_cols, images_per_row * size))
#
#     for col in range(n_cols):
#         for row in range(images_per_row):
#             channel_image = layer_activation[0, :, :, col * images_per_row + row]
#             channel_image -= channel_image.mean()
#             channel_image /= channel_image.std()
#             channel_image *= 64
#             channel_image += 128
#             channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#             display_grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = channel_image
#
#     scale = 1. / size
#     plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
#     plt.title(layer_name)
#     plt.grid(False)
#     plt.imshow(display_grid, aspect='auto', cmap='viridis')
#
# plt.show()

from keras.applications import VGG16
from keras import backend as K

# model = VGG16(weights='imagenet', include_top=False)
#
#
# def deprocess_image(x):
#     x -= x.mean()
#     x /= (x.std() + 1e-5)
#     x *= 0.1
#     x += 0.5
#     x = np.clip(x, 0, 1)
#     x *= 255
#     x = np.clip(x, 0, 255).astype('uint8')
#     return x
#
#
# def generate_pattern(layer_name, filter_index, size=150):
#     layer_output = model.get_layer(layer_name).output
#     loss = K.mean(layer_output[:, :, :, filter_index])
#
#     grads = K.gradients(loss, model.input)[0]
#     grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#
#     iterate = K.function([model.input], [loss, grads])
#     input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
#
#     step = 1.
#     for i in range(40):
#         loss_value, grads_value = iterate([input_img_data])
#         input_img_data += grads_value * step
#
#     img = input_img_data[0]
#     return deprocess_image(img)
#
#
# layer_name = 'block3_conv1'
# size = 64
# margin = 5
# results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
#
# for i in range(8):
#     for j in range(8):
#         filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
#         horizontal_start = i * size + i * margin
#         horizontal_end = horizontal_start + size
#         vertical_start = j * size + j * margin
#         vertical_end = vertical_start + size
#         results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
#
# plt.figure(figsize=(20, 20))
# plt.imshow(results)
# plt.show()

from keras.applications.vgg16 import preprocess_input, decode_predictions
import cv2

model = VGG16(weights='imagenet')

img_path = '../datasets/creative_commons_elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
print(np.argmax(preds[0]))

african_elephant_output = model.output[:, 386]
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# plt.matshow(heatmap)

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('../datasets/elephant_cam.jpg', superimposed_img)
