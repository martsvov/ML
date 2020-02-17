import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

mnist = fetch_openml('mnist_784')
Х, у = mnist["data"], mnist["target"]

some_digit = Х[36000]
some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

X_train, X_test, y_train, y_test = Х[:60000], Х[60000:], у[:60000], у[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train = y_train.astype(int)
y_train_9 = (y_train == 9)
y_test_9 = (y_test == 9)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_9)

res = sgd_clf.predict([some_digit])

print(res)
