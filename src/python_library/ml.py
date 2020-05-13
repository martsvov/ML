from _ast import Continue

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
# rng = np.random.RandomState(1)
# x = 10 * rng.rand(50)
# y = np.sin(x) + 0.1 * rng.randn(50)
# poly_model.fit(x[:, np.newaxis], y)
# xfit = np.linspace(0, 10, 1000)
# yfit = poly_model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y)
# plt.plot(xfit, yfit)

from sklearn.base import BaseEstimator, TransformerMixin

# class GaussianFeatures(BaseEstimator, TransformerMixin):
#     def __init__(self, N, width_factor=2.0):
#         self.N = N
#         self.width_factor = width_factor
#
#     @staticmethod
#     def _gauss_basis(x, y, width, axis=None):
#         arg = (x - y) / width
#         return np.exp(-0.5 * np.sum(arg ** 2, axis))
#
#     def fit(self, X, y=None):
#         self.centers_ = np.linspace(X.min(), X.max(), self.N)
#         self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
#         return self
#
#     def transform(self, X):
#         return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
#                                  self.width_, axis=1)
#
#
# gauss_model = make_pipeline(GaussianFeatures(20),
#                             LinearRegression())
# gauss_model.fit(x[:, np.newaxis], y)


# yfit = gauss_model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y)
# plt.plot(xfit, yfit)
# plt.xlim(0, 10)


# def basis_plot(model, title=None):
#     fig, ax = plt.subplots(2, sharex=True)
#     model.fit(x[:, np.newaxis], y)
#     ax[0].scatter(x, y)
#     ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
#     ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
#     if title:
#         ax[0].set_title(title)
#     ax[1].plot(model.steps[0][1].centers_,
#                model.steps[1][1].coef_)
#     ax[1].set(xlabel='basis location',
#               ylabel='coefficient',
#               xlim=(0, 10))
#
#
# model = make_pipeline(GaussianFeatures(30), LinearRegression())
# basis_plot(model)
#
# from sklearn.linear_model import Ridge
# model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
# basis_plot(model, title='Ridge Regression')
#
# from sklearn.linear_model import Lasso
# model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
# basis_plot(model, title='Lasso Regression')

from sklearn.datasets.samples_generator import make_blobs

# X, y = make_blobs(n_samples=50, centers=2,
#                   random_state=0, cluster_std=0.60)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');

from sklearn.svm import SVC

# model = SVC(kernel='linear', C=1E10)
# model.fit(X, y)
#
#
# def plot_svc_decision_function(model, ax=None, plot_support=True):
#     if ax is None:
#         ax = plt.gca()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#
#     x = np.linspace(xlim[0], xlim[1], 30)
#     y = np.linspace(ylim[0], ylim[1], 30)
#     Y, X = np.meshgrid(y, x)
#     xy = np.vstack([X.ravel(), Y.ravel()]).T
#     P = model.decision_function(xy).reshape(X.shape)
#
#     ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5,
#                linestyles=['--', '-', '--'])
#
#     if plot_support:
#         ax.scatter(model.support_vectors_[:, 0],
#                    model.support_vectors_[:, 1],
#                    s=300, linewidth=1, facecolors='none');
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plot_svc_decision_function(model)

from sklearn.datasets.samples_generator import make_circles

# X, y = make_circles(100, factor=.1, noise=.1)
# clf = SVC(kernel='rbf', C=1E6)
# clf.fit(X, y)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plot_svc_decision_function(clf)
# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
#             s=300, lw=1, facecolors='none')

from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person=60)
# # fig, ax = plt.subplots(3, 5)
# # for i, axi in enumerate(ax.flat):
# #     axi.imshow(faces.images[i], cmap='bone')
# #     axi.set(xticks=[], yticks=[],
# #             xlabel=faces.target_names[faces.target[i]])
#
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# from sklearn.pipeline import make_pipeline
#
# pca = PCA(n_components=150, whiten=True, random_state=42)
# svc = SVC(kernel='rbf', class_weight='balanced')
# model = make_pipeline(pca, svc)
#
# from sklearn.model_selection import train_test_split
#
# Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data,
#                                                 faces.target,
#                                                 random_state=42)
#
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {'svc__C': [1, 5, 10, 50],
#               'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
# grid = GridSearchCV(model, param_grid)
# grid.fit(Xtrain, ytrain)
# print(grid.best_params_)
#
# model = grid.best_estimator_
# yfit = model.predict(Xtest)

# fig, ax = plt.subplots(4, 6)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
#     axi.set(xticks=[], yticks=[])
#     axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
#                    color='black' if yfit[i] == ytest[i] else 'red')
# fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)

from sklearn.metrics import confusion_matrix

# mat = confusion_matrix(ytest, yfit)
# seaborn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=faces.target_names,
#             yticklabels=faces.target_names)
# plt.xlabel('true label')
# plt.ylabel('predicted label')

# X, y = make_blobs(n_samples=300, centers=4,
#                   random_state=0, cluster_std=1.0)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')
#
# from sklearn.tree import DecisionTreeClassifier
#
# # tree = DecisionTreeClassifier().fit(X, y)
#
#
# def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
#     ax = ax or plt.gca()
#     ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
#                clim=(y.min(), y.max()), zorder=3)
#     ax.axis('tight')
#     ax.axis('off')
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#
#     model.fit(X, y)
#     xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
#                          np.linspace(*ylim, num=200))
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
#
#     n_classes = len(np.unique(y))
#     contours = ax.contourf(xx, yy, Z, alpha=0.3,
#                            levels=np.arange(n_classes + 1) - 0.5,
#                            cmap=cmap, clim=(y.min(), y.max()),
#                            zorder=1)
#     ax.set(xlim=xlim, ylim=ylim)
#
#
# visualize_classifier(DecisionTreeClassifier(), X, y)
#
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import BaggingClassifier
#
# tree = DecisionTreeClassifier()
# bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8,
#                         random_state=1)
# bag.fit(X, y)
# visualize_classifier(bag, X, y)

# rng = np.random.RandomState(42)
# x = 10 * rng.rand(200)
#
#
# def model(x, sigma=0.3):
#     fast_oscillation = np.sin(5 * x)
#     slow_oscillation = np.sin(0.5 * x)
#     noise = sigma * rng.randn(len(x))
#     return slow_oscillation + fast_oscillation + noise
#
#
# from sklearn.ensemble import RandomForestRegressor
#
# y = model(x)
# forest = RandomForestRegressor(200)
# forest.fit(x[:, None], y)
# xfit = np.linspace(0, 10, 1000)
# yfit = forest.predict(xfit[:, None])
# ytrue = model(xfit, sigma=0)
# plt.errorbar(x, y, 0.3, fmt='o', alpha=0.5)
# plt.plot(xfit, yfit, '-r')
# plt.plot(xfit, ytrue, '-k', alpha=0.5)

from sklearn.decomposition import PCA

# rng = np.random.RandomState(1)
# X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
# plt.scatter(X[:, 0], X[:, 1])
# plt.axis('equal')

# pca = PCA(n_components=2)
# pca.fit(X)


# def draw_vector(v0, v1, ax=None):
#     ax = ax or plt.gca()
#     arrowprops = dict(arrowstyle='->',
#                       linewidth=2,
#                       shrinkA=0, shrinkB=0)
#     ax.annotate('', v1, v0, arrowprops=arrowprops)
#
#
# plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v)
# plt.axis('equal')

# pca = PCA(n_components=1)
# pca.fit(X)
# X_pca = pca.transform(X)
# X_new = pca.inverse_transform(X_pca)
# plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
# plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
# plt.axis('equal')

from sklearn.datasets import load_digits

# digits = load_digits()
# pca = PCA(2)
# projected = pca.fit_transform(digits.data)
# plt.scatter(projected[:, 0], projected[:, 1],
#             c=digits.target, edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('Spectral', 10))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar()

# pca = PCA().fit(digits.data)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')

# pca = PCA(150, svd_solver='randomized')
# pca.fit(faces.data)
# fig, axes = plt.subplots(3, 8, figsize=(9, 4),
#                          subplot_kw={'xticks': [], 'yticks': []},
#                          gridspec_kw=dict(hspace=0.1, wspace=0.1))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')

# def make_hello(N=1000, rseed=42):
#     fig, ax = plt.subplots(figsize=(4, 1))
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     ax.axis('off')
#     ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold',
#             size=85)
#     fig.savefig('hello.png')
#     plt.close(fig)
#
#     from matplotlib.image import imread
#     data = imread('hello.png')[::-1, :, 0].T
#     rng = np.random.RandomState(rseed)
#     X = rng.rand(4 * N, 2)
#     i, j = (X * data.shape).astype(int).T
#     mask = (data[i, j] < 1)
#     X = X[mask]
#     X[:, 0] *= (data.shape[0] / data.shape[1])
#     X = X[:N]
#     return X[np.argsort(X[:, 0])]
#

# X = make_hello(1000)
# colorize = dict(c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
# plt.scatter(X[:, 0], X[:, 1], **colorize)
# plt.axis('equal')

from sklearn.metrics import pairwise_distances

# D = pairwise_distances(X)
# plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
# plt.colorbar()

# def random_projection(X, dimension=3, rseed=42):
#     assert dimension >= X.shape[1]
#     rng = np.random.RandomState(rseed)
#     C = rng.randn(dimension, dimension)
#     e, V = np.linalg.eigh(np.dot(C, C.T))
#     return np.dot(X, V[:X.shape[1]])
#
#
# X3 = random_projection(X, 3)

from mpl_toolkits import mplot3d
from sklearn.manifold import MDS

# ax = plt.axes(projection='3d')
# ax.scatter3D(X3[:, 0], X3[:, 1], X3[:, 2],
#              **colorize)
# ax.view_init(azim=70, elev=50)

# model = MDS(n_components=2, random_state=1)
# out3 = model.fit_transform(X3)
# plt.scatter(out3[:, 0], out3[:, 1], **colorize)
# plt.axis('equal')

# def make_hello_s_curve(X):
#     t = (X[:, 0] - 2) * 0.75 * np.pi
#     x = np.sin(t)
#     y = X[:, 1]
#     z = np.sign(t) * (np.cos(t) - 1)
#     return np.vstack((x, y, z)).T
#
#
# XS = make_hello_s_curve(X)

# ax = plt.axes(projection='3d')
# ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2],
#              **colorize)

from sklearn.manifold import LocallyLinearEmbedding

# model = LocallyLinearEmbedding(n_neighbors=100, n_components=2, method='modified',
#                                eigen_solver='dense')
# out = model.fit_transform(XS)
# fig, ax = plt.subplots()
# ax.scatter(out[:, 0], out[:, 1], **colorize)
# ax.set_ylim(0.15, -0.15)

from sklearn.manifold import Isomap
from matplotlib import offsetbox

# model = Isomap(n_components=2)
# proj = model.fit_transform(faces.data)
#
#
# def plot_components(data, model, images=None, ax=None, thumb_frac=0.05, cmap='gray'):
#     ax = ax or plt.gca()
#     proj = model.fit_transform(data)
#     ax.plot(proj[:, 0], proj[:, 1], '.k')
#
#     if images is not None:
#         proj_max = proj.max(0)
#         proj_min = proj.min(0)
#         min_dist_2 = (thumb_frac * max(proj_max - proj_min)) ** 2
#         shown_images = np.array([2 * proj.max(0)])
#         for i in range(data.shape[0]):
#             dist = np.sum((proj[i] - shown_images) ** 2, 1)
#             if np.min(dist) < min_dist_2:
#                 Continue
#             shown_images = np.vstack([shown_images, proj[i]])
#             imagebox = offsetbox.AnnotationBbox(
#                 offsetbox.OffsetImage(images[i], cmap=cmap),
#                 proj[i])
#             ax.add_artist(imagebox)
#
#
# fig, ax = plt.subplots(figsize=(10, 10))
# plot_components(faces.data,
#                 model=Isomap(n_components=2),
#                 images=faces.images[:, ::2, ::2])

# X, y_true = make_blobs(n_samples=300, centers=4,
#                        cluster_std=0.60, random_state=0)
# plt.scatter(X[:, 0], X[:, 1], s=50)

from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters=4)
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)

# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

from sklearn.metrics import pairwise_distances_argmin

# def find_clusters(X, n_clusters, rseed=2):
#     rng = np.random.RandomState(rseed)
#     i = rng.permutation(X.shape[0])[:n_clusters]
#     centers = X[i]
#     while True:
#         labels = pairwise_distances_argmin(X, centers)
#         new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
#         if np.all(centers == new_centers):
#             break
#         centers = new_centers
#
#     return centers, labels
#
#
# centers, labels = find_clusters(X, 4)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

# X, y = make_moons(200, noise=.05, random_state=0)
# model = SpectralClustering(n_clusters=2,
#                            affinity='nearest_neighbors',
#                            assign_labels='kmeans')
# labels = model.fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels,
#             s=50, cmap='viridis')

from sklearn.datasets import load_sample_image

# china = load_sample_image("china.jpg")
# ax = plt.axes(xticks=[], yticks=[])
# ax.imshow(china)

# data = china / 255.0
# data = data.reshape(427 * 640, 3)
#
#
# def plot_pixels(data, title, colors=None, N=10000):
#     if colors is None:
#         colors = data
#     rng = np.random.RandomState(0)
#     i = rng.permutation(data.shape[0])[:N]
#     colors = colors[i]
#     R, G, B = data[i].T
#     fig, ax = plt.subplots(1, 2, figsize=(16, 6))
#
#     ax[0].scatter(R, G, color=colors, marker='.')
#     ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))
#     ax[1].scatter(R, B, color=colors, marker='.')
#     ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
#     fig.suptitle(title, size=20)
#
#
# # plot_pixels(data, title='Input color space: 16 million possible colors')
#
# from sklearn.cluster import MiniBatchKMeans
#
# kmeans = MiniBatchKMeans(16)
# kmeans.fit(data)
# new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
# # plot_pixels(data, colors=new_colors,
# #             title="Reduced color space: 16 colors")
#
# china_recolored = new_colors.reshape(china.shape)
# fig, ax = plt.subplots(1, 2, figsize=(16, 6),
#                        subplot_kw=dict(xticks=[], yticks=[]))
# fig.subplots_adjust(wspace=0.05)
# ax[0].imshow(china)
# ax[0].set_title('Original Image', size=16)
# ax[1].imshow(china_recolored)
# ax[1].set_title('16-color Image', size=16)

from sklearn.mixture import GaussianMixture

# X, y_true = make_blobs(n_samples=400, centers=4,
#                        cluster_std=0.60, random_state=0)
# X = X[:, ::-1]
#
# gmm = GaussianMixture(n_components=4).fit(X)
# labels = gmm.predict(X)
# # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
#
# probs = gmm.predict_proba(X)
# size = 50 * probs.max(1) ** 2
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size)

from sklearn.datasets import make_moons
from matplotlib.patches import Ellipse


# def draw_ellipse(position, covariance, ax=None, **kwargs):
#     ax = ax or plt.gca()
#     if covariance.shape == (2, 2):
#         U, s, Vt = np.linalg.svd(covariance)
#         angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
#         width, height = 2 * np.sqrt(s)
#     else:
#         angle = 0
#         width, height = 2 * np.sqrt(covariance)
#
#     for nsig in range(1, 4):
#         ax.add_patch(Ellipse(position, nsig * width, nsig * height,
#                              angle, **kwargs))
#
#
# def plot_gmm(gmm, X, label=True, ax=None):
#     ax = ax or plt.gca()
#     labels = gmm.fit(X).predict(X)
#     if label:
#         ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
#     else:
#         ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
#     ax.axis('equal')
#     w_factor = 0.2 / gmm.weights_.max()
#     for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
#         draw_ellipse(pos, covar, alpha=w * w_factor)
#
#
# Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
# plt.scatter(Xmoon[:, 0], Xmoon[:, 1])
#
# gmm16 = GaussianMixture(n_components=16, covariance_type='full', random_state=0)
# plot_gmm(gmm16, Xmoon, label=False)

def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x


# x = make_data(1000)
# hist = plt.hist(x, bins=30, normed=True)
# density, bins, patches = hist
# widths = bins[1:] - bins[:-1]
# print((density * widths).sum())

x = make_data(20)
# x_d = np.linspace(-4, 8, 2000)
# density = sum((abs(xi - x_d) < 0.5) for xi in x)
# plt.fill_between(x_d, density, alpha=0.5)
# plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
# plt.axis([-4, 8, -0.2, 8])
#
# from scipy.stats import norm
#
x_d = np.linspace(-4, 8, 1000)
# density = sum(norm(xi).pdf(x_d) for xi in x)
# plt.fill_between(x_d, density, alpha=0.5)
# plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
# plt.axis([-4, 8, -0.2, 5])

from sklearn.neighbors import KernelDensity

# kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
# kde.fit(x[:, None])
#
# logprob = kde.score_samples(x_d[:, None])
# plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
# plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
# plt.ylim(-0.02, 0.22)

from sklearn.datasets import fetch_species_distributions

# data = fetch_species_distributions()
# latlon = np.vstack([data.train['dd lat'],
#                     data.train['dd long']]).T
# species = np.array([d.decode('ascii').startswith('micro')
#                     for d in data.train['species']], dtype='int')

from skimage import data, color, feature
import skimage.data

image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualize=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')
ax[1].imshow(hog_vis)
ax[1].set_title('visualization of HOG features')

plt.show()
