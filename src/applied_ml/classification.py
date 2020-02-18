import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,\
    precision_recall_curve, roc_curve, roc_auc_score


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

# skfolds = StratifiedKFold(n_splits=3, random_state=42)
# for train_index, test_index in skfolds.split(X_train, y_train_9):
#     clone_clf = clone(sgd_clf)
#     X_train_folds = X_train[train_index]
#     y_train_folds = y_train_9[train_index]
#     X_test_fold = X_train[test_index]
#     y_test_fold = y_train_9[test_index]
#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     n_correct = sum(y_pred == y_test_fold)
#     print(n_correct/len(y_pred))
cvs = cross_val_score(sgd_clf, X_train, y_train_9, cv=3, scoring="accuracy")

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_9, cv=3)
# res = sgd_clf.predict(X_train)
# cm = confusion_matrix(y_train_9, y_train_pred)
# print(precision_score(y_train_9, y_train_pred))
# print(recall_score(y_train_9, y_train_pred))
# print(f1_score(y_train_9, y_train_pred))

y_scores = cross_val_predict(sgd_clf, X_train, y_train_9, cv=3, method="decision_function")

# def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#     plt.plot(thresholds, precisions[:-1], "b--", label="Точность")
#     plt.plot(thresholds, recalls[:-1], "g-", label="Полнота")
#     plt.xlabel("Пopoг")
#     plt.legend(loc="center left")
#     plt.ylim([0, 1])
#
# precisions, recalls, thresholds = precision_recall_curve(y_train_9, y_scores)
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)


# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.axis([0, 1, 0, 1])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#
# fpr, tpr, thresholds = roc_curve(y_train_9, y_scores)
# plot_roc_curve(fpr, tpr)

print(roc_auc_score(y_train_9, y_scores))

# plt.show()
print()
