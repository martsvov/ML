import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# class Perceptron(object):
#     def __init__(self, eta=0.01, n_iter=10):
#         self.eta = eta
#         self.n_iter = n_iter
#
#     def fit(self, X, y):
#         self.w_ = np.zeros(1 + X.shape[1])
#         self.errors_ = []
#
#         for _ in range(self.n_iter):
#             errors = 0
#             for xi, target in zip(X, y):
#                 update = self.eta * (target - self.predict(xi))
#                 self.w_[1:] += update * xi
#                 self.w_[0] += update
#                 errors += int(update != 0.0)
#             self.errors_.append(errors)
#         return self
#
#     def net_input(self, X):
#         return np.dot(X, self.w_[1:]) + self.w_[0]
#
#     def predict(self, X):
#         return np.where(self.net_input(X) >= 0.0, 1, - 1)
#
#
# def plot_decision_regions(X, y, classifier, resolution=0.02):
#     markers = ('s', 'x', 'o', '^', 'v')
#     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])
#
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                            np.arange(x2_min, x2_max, resolution))
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
#     plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
#
#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1],
#                     alpha=0.8, c=cmap(idx),
#                     marker=markers[idx], label=cl)
#
#
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# df = pd.read_csv(url, header=None)
#
# y = df.iloc[0:100, 4].values
# y = np.where(y == 'Iris-setosa', -1, 1)
# X = df.iloc[0:100, [0, 2]].values
# plt.scatter(X[:50, 0], X[:50, 1],
#             color='red', marker='o', label='щетинистый')
# plt.scatter(X[50:100, 0], X[50:100, 1],
#             color='blue', marker='x', label='разноцветный')
# plt.xlabel('длина чашелистика')
# plt.ylabel('длина лепестка')
# plt.legend(loc='upper left')
# plt.show()

# ppn = Perceptron(eta=0.1, n_iter=10)
# ppn.fit(X, y)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Эпохи')
# plt.ylabel('Число случаев ошибочной классификации')
# plt.show()

# plot_decision_regions(X, y, classifier=ppn)
# plt.xlabel('длина чашелистика [см]')
# plt.ylabel('длина лепестка [см]')
# plt.legend(loc='upper left')
# plt.show()

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
# df_wine = pd.read_csv(url, header=None)
#
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# sc = StandardScaler()
# X_train_std = sc.fit_transform(X_train)
# X_test_std = sc.transform(X_test)
#
# cov_mat = np.cov(X_train_std.T)
# eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
#
# eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
#                for i in range(len(eigen_vals))]
# eigen_pairs.sort(reverse=True)
#
# # print(eigen_pairs)
#
# mean_vecs = []
# for label in range(1, 4):
#     mean_vecs.append(np.mean(
#         X_train_std[y_train == label], axis=0))
#     print(label, mean_vecs[label - 1])
#
# d = 13
# S_W = np.zeros((d, d))
# for label, mv in zip(range(1, 4), mean_vecs):
#     class_scatter = np.zeros((d, d))
#     for row in X_train[y_train == label]:
#         row, mv = row.reshape(d, 1), mv.reshape(d, 1)
#         class_scatter += (row - mv).dot((row - mv).T)
#     S_W += class_scatter
#
# print('Внутриклассовая матрица разброса: %sx%s' % (S_W.shape[0], S_W.shape[1]))
#
# mean_overall = np.mean(X_train_std, axis=0)
# S_B = np.zeros((d, d))
# for i, mean_vec in enumerate(mean_vecs):
#     n = X_train[y_train == i + 1, :].shape[0]
#     mean_vec = mean_vec.reshape(d, 1)
#     mean_overall = mean_overall.reshape(d, 1)
#     S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
# print(' Межклассовая мат р ица разбр оса : %sx%s '
#       % (S_B.shape[0], S_B.shape[1]))

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh

# def rbf_kernel_pca(X, gamma, n_components):
#     sq_dists = pdist(X, 'sqeuclidean')
#     mat_sq_dists = squareform(sq_dists)
#
#     K = exp(-gamma * mat_sq_dists)
#     N = K.shape[0]
#     one_n = np.ones((N, N)) / N
#     K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
#
#     eigvals, eigvecs = eigh(K)
#     X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
#
#     return X_pc
#
#
# from sklearn.datasets import make_moons
# from matplotlib.ticker import FormatStrFormatter
#
# X, y = make_moons(n_samples=100, random_state=123)
# plt.scatter(X[y == 0, 0], X[y == 0, 1],
#             color='red', marker='^', alpha=0.5)
# plt.scatter(X[y == 1, 0], X[y == 1, 1],
#             color='blue', marker='o', alpha=0.5)
# # plt.show()
#
# X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
# ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1],
#               color='blue', marker='o', alpha=0.5)
# ax[1].scatter(X_kpca[y == 0, 0], np.zeros((50, 1)) + 0.02,
#               color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_kpca[y == 1, 0], np.zeros((50, 1)) - 0.02,
#               color='blue', marker='o', alpha=0.5)
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
# ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.lf'))
# ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.lf'))
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
# df = pd.read_csv(url, header=None)
#
# X = df.loc[:, 2:].values
# y = df.loc[:, 1].values
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# pipe_lr = Pipeline([
#     ('scl', StandardScaler()),
#     ('clf', LogisticRegression(penalty='l2', random_state=0))])
#
# train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
#                                                         X=X_train,
#                                                         y=y_train,
#                                                         train_sizes=np.linspace(0.1, 1.0, 10),
#                                                         cv=10,
#                                                         n_jobs=1)
#
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
# plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5)
# plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
# plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5)
# plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
# plt.grid()
# plt.xlabel('Чиcлo тренировочных образцов')
# plt.ylabel('Bepнocть')
# plt.legend(loc='lower right')
# plt.ylim([0.8, 1.0])
# plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# pipe_svc = Pipeline([('scl', StandardScaler()),
#                      ('clf', SVC(random_state=1))])
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# param_grid = [{'clf__C': param_range,
#                'clf__kernel': ['linear']},
#               {'clf__C': param_range,
#                'clf__gamma': param_range,
#                'clf__kernel': ['rbf']}]
# gs = GridSearchCV(estimator=pipe_svc,
#                   param_grid=param_grid,
#                   scoring='accuracy',
#                   cv=10,
#                   n_jobs=-1)
# gs = gs.fit(X_train, y_train)
#
# print(gs.best_score_)
# print(gs.best_params_)

import pyprind
import os

# pbar = pyprind.ProgBar(50000)
# labels = {'pos': 1, 'neg': 0}
# df = pd.DataFrame()
#
# for s in ('test', 'train'):
#     for l in ('pos', 'neg'):
#         path = '../datasets/aclimdb/%s/%s' % (s, l)
#         for file in os.listdir(path):
#             with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
#                 txt = infile.read()
#             df = df.append([[txt, labels[l]]], ignore_index=True)
#             pbar.update()
#
# df.columns = ['review', 'sentiment']
#
# np.random.seed(0)
# df = df.reindex(np.random.permutation(df.index))
# df.to_csv('../datasets/aclimdb/movie_data.csv', index=False)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.stem.porter import PorterStemmer


# count = CountVectorizer()
# docs = np.array([
#     'The sun is shining ',
#     'The weather is sweet',
#     'The sun is shining and the weather is sweet, and one and one is two'])
# bag = count.fit_transform(docs)
# print(count.vocabulary_)
# print(bag.toarray())
#
# tfidf = TfidfTransformer()
# np.set_printoptions(precision=2)
# print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


# def tokenizer(text):
#     return text.split()
#
#
# porter = PorterStemmer()
#
#
# def tokenizer_porter(text):
#     return [porter.stem(word) for word in text.split()]


# print(tokenizer_porter('runners like running and thus they run'))

# df = pd.read_csv('../datasets/aclimdb/movie_data.csv')
# df['review'] = df['review'].apply(preprocessor)
# X_train = df.loc[:25000, 'review'].values
# y_train = df.loc[:25000, 'sentiment'].values
# X_test = df.loc[25000:, 'review'].values
# y_test = df.loc[25000:, 'sentiment'].values

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

#
# tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
# param_grid = [{'vect__ngram_range': [(1, 1)],
#                'vect__stop_words': [None],
#                'vect__tokenizer': [tokenizer,
#                                     tokenizer_porter],
#                'clf__penalty': ['l1', 'l2'],
#                'clf__C': [1.0, 10.0, 100.0]},
#               {'vect__ngram_range': [(1, 1)],
#                'vect__stop_words': [None],
#                'vect__tokenizer': [tokenizer,
#                                    tokenizer_porter],
#                'vect__use_idf': [False],
#                'vect__norm': [None],
#                'clf__penalty': ['l1', 'l2'],
#                'clf__C': [1.0, 10.0, 100.0]}]
# lr_tfidf = Pipeline([('vect', tfidf),
#                      ('clf', LogisticRegression(random_state=0))])
#
# gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
#                            scoring='accuracy',
#                            cv=5, verbose=1,
#                            n_jobs=-1)
# gs_lr_tfidf.fit(X_train, y_train)

# from nltk.corpus import stopwords
# stop = stopwords.words('english')


# def tokenizer(text):
#     text = re.sub('<[^>]*>', '', text)
#     emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
#     text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
#     # tokenized = [w for w in text.split() if w not in stop]
#     tokenized = [w for w in text.split()]
#     return tokenized
#
#
# def stream_docs(path):
#     with open(path, 'r', encoding='utf-8') as csv:
#         next(csv)
#         for line in csv:
#             text, label = line[:-3], int(line[-2])
#             yield text, label
#
#
# def get_minibatch(doc_stream, size):
#     docs, y = [], []
#     try:
#         for _ in range(size):
#             text, label = next(doc_stream)
#             docs.append(text)
#             y.append(label)
#     except StopIteration:
#         return None, None
#     return docs, y
#
#
# from sklearn.feature_extraction.text import HashingVectorizer
# from sklearn.linear_model import SGDClassifier
#
# vect = HashingVectorizer(decode_error='ignore',
#                          n_features=2 ** 21,
#                          preprocessor=None,
#                          tokenizer=tokenizer)
# clf = SGDClassifier(loss='log', random_state=1, n_iter_no_change=1)
# doc_stream = stream_docs(path='../datasets/aclimdb/movie_data.csv')
#
# pbar = pyprind.ProgBar(45)
# classes = np.array([0, 1])
# for _ in range(45):
#     X_train, y_train = get_minibatch(doc_stream, size=1000)
#     if not X_train:
#         break
#     X_train = vect.transform(X_train)
#     clf.partial_fit(X_train, y_train, classes=classes)
#     pbar.update()
#
# X_test, y_test = get_minibatch(doc_stream, size=5000)
# X_test = vect.transform(X_test)
# print('Bepнocть:', clf.score(X_test, y_test))
#
# clf = clf.partial_fit(X_test, y_test)
#
# import pickle
#
# dest = os.path.join('movieclassifier', 'pkl_objects')
# if not os.path.exists(dest):
#     os.makedirs(dest)
# pickle.dump(clf,
#             open(os.path.join(dest, 'classifier.pkl'), 'wb'),
#             protocol=4)


import seaborn as sns

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
# df = pd.read_csv(url, header=None, sep='\s+')
# df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RA0',
#               'ТАХ', 'PTRATIO', 'В', 'LSTAT', 'MEDV']

# sns.set(style='whitegrid', context='notebook')
# cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
# sns.pairplot(df[cols], size=2.5)
# plt.show()

# cm = np.corrcoef(df[cols].values.T)
# sns.set(font_scale=1.5)
# hm = sns.heatmap(cm,
#                  cbar=True,
#                  annot=True,
#                  square=True,
#                  fmt='.2f',
#                  annot_kws={'size': 15},
#                  yticklabels=cols,
#                  xticklabels=cols)
#
# plt.show()

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# X, y = make_blobs(n_samples=150,
#                   n_features=2,
#                   centers=3,
#                   cluster_std=0.5,
#                   shuffle=True,
#                   random_state=0)

# plt.scatter(X[:, 0],
#             X[:, 1],
#             marker='o',
#             s=50)
# plt.grid()
# plt.show()

# km = KMeans(n_clusters=3,
#             init='random',
#             n_init=10,
#             max_iter=300,
#             tol=1e-04,
#             random_state=0)
# y_km = km.fit_predict(X)
#
# plt.scatter(X[y_km == 0, 0],
#             X[y_km == 0, 1],
#             s=50,
#             c='lightgreen',
#             marker='s',
#             label='кластер 1')
# plt.scatter(X[y_km == 1, 0],
#             X[y_km == 1, 1],
#             s=50,
#             c='orange',
#             marker='o',
#             label='кластер 2')
# plt.scatter(X[y_km == 2, 0],
#             X[y_km == 2, 1],
#             s=50,
#             c='lightblue',
#             marker='v',
#             label='кластер 3')
# plt.scatter(km.cluster_centers_[:, 0],
#             km.cluster_centers_[:, 1],
#             s=250,
#             marker='*',
#             c='red',
#             label='центроиды')
# plt.legend()
# plt.grid()
# plt.show()

from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
# km = KMeans(n_clusters=2, random_state=0)
# y_km = km.fit_predict(X)
# ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1], c='lightblue', marker='o', s=40, label='кластер 1')
# ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1], c='red', marker='s', s=40, label='кластер 2')
# ax1.set_title('Кластеризация по методу k средних')
#
# ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
# y_ac = ac.fit_predict(X)
# ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c='lightblue', marker='o', s=40, label='кластер 1')
# ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c='red', marker='s', s=40, label='кластер 2')
# ax2.set_title('Агломеративная кластеризация')
# plt.legend()
# plt.show()

from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')

y_db = db.fit_predict(X)
plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1], c='lightblue', marker='o', s=40, label='кластер 1')
plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1], c='red', marker='s', s=40, label='кластер 2')

plt.legend()
plt.show()
