import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, \
    precision_recall_curve, roc_curve, roc_auc_score

# mnist = fetch_openml('mnist_784')
# Х, у = mnist["data"], mnist["target"]

# some_digit = Х[36000]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

# X_train, X_test, y_train, y_test = Х[:60000], Х[60000:], у[:60000], у[60000:]
# shuffle_index = np.random.permutation(60000)
# X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
#
# y_train = y_train.astype(int)
# y_train_9 = (y_train == 9)
# y_test_9 = (y_test == 9)

# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_9)

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
# cvs = cross_val_score(sgd_clf, X_train, y_train_9, cv=3, scoring="accuracy")

# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_9, cv=3)
# res = sgd_clf.predict(X_train)
# cm = confusion_matrix(y_train_9, y_train_pred)
# print(precision_score(y_train_9, y_train_pred))
# print(recall_score(y_train_9, y_train_pred))
# print(f1_score(y_train_9, y_train_pred))

# y_scores = cross_val_predict(sgd_clf, X_train, y_train_9, cv=3, method="decision_function")

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
#
# fpr, tpr, thresholds = roc_curve(y_train_9, y_scores)
# plot_roc_curve(fpr, tpr)
# print(roc_auc_score(y_train_9, y_scores))

# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train)
# some_digit_scores = sgd_clf.decision_function(X_train)
# print(some_digit_scores)

# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)
#
# X_b = np.c_[np.ones((100, 1)), X]
# theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
#
# X_new = np.array([[0], [2]])
# X_new_b = np.c_[np.ones((2, 1)), X_new]
#
# y_predict = X_new_b.dot(theta_best)
#
# plt.plot(X_new, y_predict, "r-")
# plt.plot(X, y, "b.")
# plt.axis([0, 2, 0, 15])
# plt.show()

# eta = 0.1
# n_iterations = 1000
# m = 100
# theta = np.random.randn(2, 1)
#
# for iteration in range(n_iterations):
#     gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
#     theta = theta - eta * gradients
#
# print(theta)

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# iris = datasets.load_iris()
# X = iris["data"][:, 3:]
# y = (iris["target"] == 2).astype(np.int)
#
# log_reg = LogisticRegression()
# log_reg.fit(X, y)
#
# X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# y_proba = log_reg.predict_proba(X_new)
# plt.plot(X_new, y_proba[:, 1], "g-")
# plt.plot(X_new, y_proba[:, 0], "b--")
#
# plt.show()

from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# iris = datasets.load_iris()
# X = iris["data"][:, (2, 3)]
# y = (iris["target"] == 2).astype(np.float64)
# svm_clf = Pipeline([
#     ("scaler", StandardScaler()),
#     ("linear_svc", LinearSVC(C=1, loss="hinge")),
# ])
# svm_clf.fit(X, y)
# print(svm_clf.predict([[5.5, 1.7]]))

# rbf_kernel_svm_clf = Pipeline([
#     ("scaler", StandardScaler()),
#     ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
# ])
# rbf_kernel_svm_clf.fit(X, y)

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz
#
# iris = datasets.load_iris()
# X = iris.data[:, 2:]
# y = iris.target
# tree_clf = DecisionTreeClassifier(max_depth=2)
# tree_clf.fit(X, y)
#
# export_graphviz(
#     tree_clf,
#     out_file="iris_tree.dot",
#     feature_names=iris.feature_names[2:],
#     class_names=iris.target_names,
#     rounded=True,
#     filled=True)
#
# print(tree_clf.predict_proba([[5, 1.5]]))
# print(tree_clf.predict([[5, 1.5]]))

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X, y = make_moons(n_samples=10000)

n = 8000
X_train, X_test, y_train, y_test = X[:n], X[n:], y[:n], y[n:]

# log_clf = LogisticRegression()
# rnd_clf = RandomForestClassifier()
# svm_clf = SVC()
# voting_clf = VotingClassifier(
#     estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
#     voting='hard')
# voting_clf.fit(X_train, y_train)
#
# for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# bag_clf = BaggingClassifier(
#     DecisionTreeClassifier(), n_estimators=500,
#     max_samples=100, bootstrap=True, n_jobs=-1, oob_score=True)
# bag_clf.fit(X_train, y_train)
# y_pred = bag_clf.predict(X_test)
#
# print(y_pred)
# print(bag_clf.oob_score_)

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
                           algorithm="SAMME.R", learning_rate=0.5)
ada_clf.fit(X_train, y_train)
