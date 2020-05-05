import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve, learning_curve

seaborn.set()

# xfit = np.linspace(-1, 11)
# Xfit = xfit[:, np.newaxis]


# def PolynomialRegression(degree=2, **kwargs):
#     return make_pipeline(PolynomialFeatures(degree),
#                          LinearRegression(**kwargs))
#
#
# def make_data(N, err=1.0, rseed=1):
#     rng = np.random.RandomState(rseed)
#     X = rng.rand(N, 1) ** 2
#     y = 10 - 1. / (X.ravel() + 0.1)
#     if err > 0:
#         y += err * rng.randn(N)
#     return X, y
#
#
# X, y = make_data(40)
# X_test = np.linspace(-0.1, 1.1, 500)[:, None]
# plt.scatter(X.ravel(), y, color='black')
# axis = plt.axis()
# for degree in [1, 3, 5]:
#     y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
#     plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
#
# plt.xlim(-0.1, 1.0)
# plt.ylim(-2, 12)
# plt.legend(loc='best')

# degree = np.arange(0, 21)
# train_score, val_score = validation_curve(PolynomialRegression(), X, y,
#                                           'polynomialfeatures__degree',
#                                           degree, cv=7)
# plt.plot(degree, np.median(train_score, 1), color='blue',
#          label='training score')
# plt.plot(degree, np.median(val_score, 1), color='red',
#          label='validation score')
# plt.legend(loc='best')
# plt.ylim(0, 1)
# plt.xlabel('degree')
# plt.ylabel('score')

# fig, ax = plt.subplots(1, 2, figsize=(16, 6))
# fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
# for i, degree in enumerate([2, 9]):
#     N, train_lc, val_lc = learning_curve(PolynomialRegression(degree),
#                                          X, y, cv=7,
#                                          train_sizes=np.linspace(0.3, 1, 25))
#     ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
#     ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
#     ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1],
#                  color='gray',
#                  linestyle='dashed')
#     ax[i].set_ylim(0, 1)
#     ax[i].set_xlim(N[0], N[-1])
#     ax[i].set_xlabel('training size')  # Размерность обучения
#     ax[i].set_ylabel('score')
#     ax[i].set_title('degree = {0}'.format(degree), size=14)
#     ax[i].legend(loc='best')

# x = np.array([1, 2, 3, 4, 5])
# y = np.array([4, 2, 1, 3, 7])
# X = x[:, np.newaxis]
# poly = PolynomialFeatures(degree=3, include_bias=False)
# X2 = poly.fit_transform(X)
#
# model = LinearRegression().fit(X2, y)
# yfit = model.predict(X2)
# plt.scatter(x, y)
# plt.plot(x, yfit)

from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB

# X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
#
# model = GaussianNB()
# model.fit(X, y)
#
# rng = np.random.RandomState(0)
# Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
# ynew = model.predict(Xnew)
#
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
# lim = plt.axis()
# plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu',
#             alpha=0.1)
# plt.axis(lim)
# plt.show()

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# data = fetch_20newsgroups()
# categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space',
#               'comp.graphics']
# train = fetch_20newsgroups(subset='train', categories=categories)
# test = fetch_20newsgroups(subset='test', categories=categories)
#
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# model.fit(train.data, train.target)
# labels = model.predict(test.data)
# from sklearn.metrics import confusion_matrix
#
# mat = confusion_matrix(test.target, labels)
# seaborn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=train.target_names, yticklabels=train.target_names)
# plt.xlabel('true label')
# plt.ylabel('predicted label')

# rng = np.random.RandomState(1)
# x = 10 * rng.rand(50)
# y = 2 * x - 5 + rng.randn(50)
#
# model = LinearRegression(fit_intercept=True)
# model.fit(x[:, np.newaxis], y)
# xfit = np.linspace(0, 10, 1000)
# yfit = model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y)
# plt.plot(xfit, yfit)

# poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)
# poly_model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
# yfit = poly_model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y)
# plt.plot(xfit, yfit)

from sklearn.base import BaseEstimator, TransformerMixin


class GaussianFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)


gauss_model = make_pipeline(GaussianFeatures(20),
                            LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)


# yfit = gauss_model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y)
# plt.plot(xfit, yfit)
# plt.xlim(0, 10)


def basis_plot(model, title=None):
    fig, ax = plt.subplots(2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
    if title:
        ax[0].set_title(title)
    ax[1].plot(model.steps[0][1].centers_,
               model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location',
              ylabel='coefficient',
              xlim=(0, 10))


model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)

from sklearn.linear_model import Ridge
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
basis_plot(model, title='Ridge Regression')

from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
basis_plot(model, title='Lasso Regression')

plt.show()
