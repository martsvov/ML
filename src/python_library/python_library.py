import numpy as np
import pandas as pd
import numexpr
# from pandas_datareader import data
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
from datetime import datetime
import re

# index = [('California', 2000), ('California', 2010),
#          ('New York', 2000), ('New York', 2010),
#          ('Texas', 2000), ('Texas', 2010)]
# index = pd.MultiIndex.from_tuples(index)
#
# populations = [33871648, 37253956,
#                18976457, 19378102,
#                20851820, 25145561]
# pop = pd.Series(populations, index=index)
# pop_df = pop.unstack()

# ind = pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
# ind.names = ['state', 'year']
#
# df = pd.DataFrame(np.random.rand(4, 2),
#                   index=ind,
#                   columns=['data1', 'data2'])

# index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
#                                    names=['year', 'visit'])
# columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'],
#                                       ['HR', 'Temp']],
#                                      names=['subject', 'type'])
#
# data = np.round(np.random.randn(4, 6), 1)
# data[:, ::2] *= 10
# data += 37
#
# health_data = pd.DataFrame(data, index=index, columns=columns)


# def make_df(cols, ind):
#     data = {c: [str(c) + str(i) for i in ind] for c in cols}
#     return pd.DataFrame(data, ind)
#
#
# df1 = make_df('ABC', [1, 2])
# df2 = make_df('BCD', [3, 4])
# df3 = pd.concat([df1, df2], join='inner')

# df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
#                     'group': ['Accounting', 'Engineering', 'Engineering',
#                               'HR']})
# df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
#                     'hire_date': [2004, 2008, 2012, 2014]})
#
# df3 = pd.merge(df1, df2)

# pop = pd.read_csv('../data/state-population.csv')
# areas = pd.read_csv('../data/state-areas.csv')
# abbrevs = pd.read_csv('../data/state-abbrevs.csv')
#
# merged = pd.merge(pop, abbrevs, how='outer', left_on='state/region', right_on='abbreviation')
# merged = merged.drop('abbreviation', 1)
#
# merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
# merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
#
# final = pd.merge(merged, areas, on='state', how='left')
# uniq = merged.loc[merged['state'].isnull(), 'state/region'].unique()

# planets = sns.load_dataset('planets')

# df = planets.groupby('method')['year'].describe().unstack()

# rng = np.random.RandomState(0)
# df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
#                    'data1': range(6),
#                    'data2': rng.randint(0, 10, 6)},
#                   columns=['key', 'data1', 'data2'])
# df_agr = df.groupby('key').transform(lambda x: x - x.mean())
#
#
# def norm_by_data2(x):
#     x['data1'] /= x['data2'].sum()
#     return x

# decade = 10 * (planets['year'] // 10)
# decade_s = decade.astype(str) + 's'
#
# decade_gr = planets.groupby(['method', decade_s])['number'].sum().unstack().fillna(0)

# titanic = sns.load_dataset('titanic')
#
# age = pd.cut(titanic['age'], [0, 18, 80])
# pt = titanic.pivot_table('survived', ['sex', age], 'class')

# births = pd.read_csv('../data/births.csv')
# births['decade'] = 10 * (births['year'] // 10)
# pt = births.pivot_table('births', index='decade', columns='gender', aggfunc='sum').plot()
#
# plt.ylabel('total births per year')
# plt.show()

# quartiles = np.percentile(births['births'], [25, 50, 75])
# mu = quartiles[1]
# sig = 0.74 * (quartiles[2] - quartiles[0])
#
# births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
# births['day'] = births['day'].astype(int)
# births.index = pd.to_datetime(10000 * births.year + 100 * births.month + births.day, format='%Y%m%d')
# births['dayofweek'] = births.index.dayofweek

# births.pivot_table('births', index='dayofweek',
#                    columns='decade', aggfunc='mean').plot()
# plt.gca().set_xticklabels([' ', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
# plt.ylabel('mean births by day')

# births_by_date = births.pivot_table('births', [births.index.month, births.index.day])
# births_by_date.index = [pd.datetime(2012, month, day) for (month, day) in births_by_date.index]
#
# fig, ax = plt.subplots(figsize=(12, 4))
# births_by_date.plot(ax=ax)
# plt.show()

# monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam', 'Eric Idle', 'Terry Jones', 'Michael Palin'])
# monte.str.findall(r'^[^AEIOU]*.*[^aeiou]$')

# with open('../data/recipeitems-latest.json', 'r') as f:
#     data = (line.strip() for line in f)
#     data_json = "[{0}]".format(','.join(data))
# recipes = pd.read_json(data_json)
#
# spice_list = ['salt', 'pepper', 'oregano', 'sage', 'parsley', 'rosemary', 'tarragon', 'thyme', 'paprika', 'cumin']
# spice_df = pd.DataFrame(dict((spice, recipes.ingredients.str.contains(spice, re.IGNORECASE)) for spice in spice_list))

# index = pd.DatetimeIndex(['2014-07-04', '2014-08-04', '2015-07-04', '2015-08-04'])
# data = pd.Series([0, 1, 2, 3], index=index)

# dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015', '2015-Jul-6', '07-07-2015', '20150708'])
# dates = pd.date_range('2015-07-01', periods=8, freq='MS')

# goog = data.DataReader('GOOG', start='2004', end='2016', data_source='stooq')
# goog = goog['Close']
# goog = goog.sort_index()

# goog.plot(alpha=0.5, style='-')
# goog.resample('BA').mean().plot(style=':')
# goog.asfreq('BA').plot(style='--')
# plt.legend(['input', 'resample', 'asfreq'], loc='upper left')

# fig, ax = plt.subplots(2, sharex=True)
# data = goog.iloc[-10:]
# data.asfreq('D').plot(ax=ax[0], marker='o')
# data.asfreq('D', method='bfill').plot(ax=ax[1], style='-o')
# data.asfreq('D', method='ffill').plot(ax=ax[1], style='--o')
# ax[1].legend(["back-fill", "forward-fill"])

# _, ax = plt.subplots(3, sharey=True)
# goog = goog.asfreq('D', method='pad')
# goog.plot(ax=ax[0])
# goog.shift(900).plot(ax=ax[1])
# goog.tshift(900).plot(ax=ax[2])
#
# local_max = pd.to_datetime('2007-11-05')
# offset = pd.Timedelta(900, 'D')
# ax[0].legend(['input'], loc=2)
# ax[0].get_xticklabels()[4].set(weight='heavy', color='red')
# ax[0].axvline(local_max, alpha=0.3, color='red')
# ax[1].legend(['shift(900)'], loc=2)
# ax[1].get_xticklabels()[4].set(weight='heavy', color='red')
# ax[1].axvline(local_max + offset, alpha=0.3, color='red')
# ax[2].legend(['tshift(900)'], loc=2)
# ax[2].get_xticklabels()[1].set(weight='heavy', color='red')
# ax[2].axvline(local_max + offset, alpha=0.3, color='red');

# ROI = 100 * (goog.tshift(-365) / goog - 1)
# ROI.plot()
# plt.ylabel('% Return on Investment')

# rolling = goog.rolling(365, center=True)
# data = pd.DataFrame({'input': goog, 'one-year rolling_mean': rolling.mean(), 'one-year rolling_std': rolling.std()})
# ax = data.plot(style=['-', '--', ':'])
# ax.lines[0].set_alpha(0.3)

# data = pd.read_csv('../data/FremontBridge.csv', index_col='Date', parse_dates=True)
# data.columns = ['Total', 'East', 'West']
# data.plot()
# plt.ylabel('Hourly Bicycle Count')

# weekly = data.resample('W').sum()
# weekly.plot(style=[':', '--', '-'])
# plt.ylabel('Weekly bicycle count')

# daily = data.resample('D').sum()
# daily.rolling(50, center=True, win_type='gaussian').sum(std=10).plot(style=[':', '--', '-']);
# plt.ylabel('mean hourly count')

# by_time = data.groupby(data.index.time).mean()
# hourly_ticks = 4 * 60 * 60 * np.arange(6)
# by_time.plot(xticks=hourly_ticks, style=[':', '--', '-'])

# weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')
# by_time = data.groupby([weekend, data.index.time]).mean()
#
# fig, ax = plt.subplots(1, 2, figsize=(14, 5))
# by_time.loc['Weekday'].plot(ax=ax[0], title='Weekdays', xticks=hourly_ticks, style=[':', '--', '-'])
# by_time.loc['Weekend'].plot(ax=ax[1], title='Weekends', xticks=hourly_ticks, style=[':', '--', '-'])

# nrows, ncols = 100000, 100
# rng = np.random.RandomState(42)
# df1, df2 = (pd.DataFrame(rng.rand(nrows, ncols)) for i in range(4))

plt.style.use('seaborn-whitegrid')

# fig = plt.figure()
# ax = plt.axes()
#
# x = np.linspace(0, 10, 50)

# ax.plot(x, np.sin(x))
# plt.plot(x, x + 0, '-g')
# plt.plot(x, x + 1, '--c')
# plt.plot(x, x + 2, '-.k')
# plt.plot(x, x + 3, ':r')

# plt.plot(x, np.sin(x))
# plt.axis([-1, 11, -1.5, 1.5])
# plt.title("A Sine Curve")
# plt.xlabel("x")
# plt.ylabel("sin(x)")

# ax.plot(x, np.sin(x), '-g', label='sin(x)')
# ax.plot(x, np.cos(x), ':b', label='cos(x)')
# ax.axis('equal')
# ax.legend()

# ax.plot(x, np.sin(x))
# ax.set(xlim=(0, 10), ylim=(-2, 2), xlabel='x', ylabel='sin(x)', title='A Simple Plot')

# rng = np.random.RandomState(0)
# for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
#     plt.plot(rng.rand(5), rng.rand(5), marker, label="marker='{0}'".format(marker))
# plt.legend(numpoints=1)
# plt.xlim(0, 1.8)

# rng = np.random.RandomState(0)
# x = rng.randn(100)
# y = rng.randn(100)
# colors = rng.rand(100)
# sizes = 1000 * rng.rand(100)
# plt.scatter(x, y, c=colors, s=sizes, alpha=0.3, cmap='viridis')
# plt.colorbar()

# from sklearn.datasets import load_iris
#
# iris = load_iris()
# features = iris.data.T
# plt.scatter(features[0], features[1], alpha=0.2, s=100*features[3], c=iris.target, cmap='viridis')
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[1])

# x = np.linspace(0, 10, 50)
# dy = 0.8
# y = np.sin(x) + dy * np.random.randn(50)
# plt.errorbar(x, y, yerr=dy, fmt='.k')
# plt.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)

# from sklearn.gaussian_process import GaussianProcessRegressor
#
# model = lambda x: x * np.sin(x)
# xdata = np.array([1, 3, 5, 6, 8])
# ydata = model(xdata)
#
# gp = GaussianProcessRegressor()
# gp.fit(xdata[:, np.newaxis], ydata)
# xfit = np.linspace(0, 10, 1000)
# yfit, MSE = gp.predict(xfit[:, np.newaxis], return_std=True)
# dyfit = 2 * np.sqrt(MSE)
#
# plt.plot(xdata, ydata, 'or')
# plt.plot(xfit, yfit, '-', color='gray')
# plt.fill_between(xfit, yfit - dyfit, yfit + dyfit, color='gray', alpha=0.2)
# plt.xlim(0, 10)

# def f(x, y):
#     return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
#
#
# x = np.linspace(0, 5, 50)
# y = np.linspace(0, 5, 40)
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)
# plt.contour(X, Y, Z, colors='black')
# plt.contourf(X, Y, Z, 20, cmap='RdGy')
# plt.colorbar()

# plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy')
# plt.colorbar()
# plt.axis(aspect='image')

# contours = plt.contour(X, Y, Z, 3, colors='black')
# plt.clabel(contours, inline=True, fontsize=8)
# plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy', alpha=0.5)
# plt.colorbar()

# data = np.random.randn(1000)
# plt.hist(data, bins=30, normed=True, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')

# x1 = np.random.normal(0, 0.8, 1000)
# x2 = np.random.normal(-2, 1, 1000)
# x3 = np.random.normal(3, 2, 1000)
#
# kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
# plt.hist(x1, **kwargs)
# plt.hist(x2, **kwargs)
# plt.hist(x3, **kwargs)

# mean = [0, 0]
# cov = [[1, 1], [1, 2]]
# x, y = np.random.multivariate_normal(mean, cov, 10000).T
# plt.hist2d(x, y, bins=30, cmap='Blues')
# cb = plt.colorbar()
# cb.set_label('counts in bin')
#
# plt.hexbin(x, y, gridsize=30, cmap='Blues')
# cb = plt.colorbar()
# cb.set_label('count in bin')

# x = np.linspace(0, 10, 1000)
# fig, ax = plt.subplots()
# ax.plot(x, np.sin(x), '-b', label='Sine')
# ax.plot(x, np.cos(x), '--r', label='Cosine')
# ax.axis('equal')
# ax.legend(frameon=False, loc='lower center', ncol=2)

# y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
# lines = plt.plot(x, y)
# plt.legend(lines[:2], ['first', 'second'])

# cities = pd.read_csv('../data/california_cities.csv')
#
# lat, lon = cities['latd'], cities['longd']
# population, area = cities['population_total'], cities['area_total_km2']
#
# plt.scatter(lon, lat, label=None, c=np.log10(population), cmap='viridis', s=area, linewidth=0, alpha=0.5)
# plt.axis(aspect='equal')
# plt.xlabel('longitude')
# plt.ylabel('latitude')
# plt.colorbar(label='log$_{10}$(population)')
# plt.clim(3, 7)
#
# for area in [100, 300, 500]:
#     plt.scatter([], [], c='k', alpha=0.3, s=area, label=str(area) + ' km$^2$')
#
# plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
# plt.title('California Cities: Area and Population')
#
# plt.show()

# plt.style.use('classic')
# x = np.linspace(0, 10, 1000)
# I = np.sin(x) * np.cos(x[:, np.newaxis])
# plt.imshow(I)
# plt.colorbar()

# speckles = (np.random.random(I.shape) < 0.01)
# I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))
# plt.figure(figsize=(10, 3.5))
# plt.subplot(1, 2, 1)
# plt.imshow(I, cmap='RdBu')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.imshow(I, cmap='RdBu')
# plt.colorbar(extend='both')
# plt.clim(-1, 1)

from sklearn.datasets import load_digits

# digits = load_digits(n_class=6)
# fig, ax = plt.subplots(8, 8, figsize=(6, 6))
# for i, axi in enumerate(ax.flat):
#     axi.imshow(digits.images[i], cmap='binary')
#     axi.set(xticks=[], yticks=[])

# mean = [0, 0]
# cov = [[1, 1], [1, 2]]
# x, y = np.random.multivariate_normal(mean, cov, 3000).T
#
# fig = plt.figure(figsize=(6, 6))
# grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
# main_ax = fig.add_subplot(grid[:-1, 1:])
# y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
# x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
# main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)
#
# x_hist.hist(x, 40, histtype='stepfilled',
#             orientation='vertical', color='gray')
# x_hist.invert_yaxis()
# y_hist.hist(y, 40, histtype='stepfilled',
#             orientation='horizontal', color='gray')
# y_hist.invert_xaxis()

from mpl_toolkits import mplot3d

# ax = plt.axes(projection='3d')
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')
#
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')


# def f(x, y):
#     return np.sin(np.sqrt(x ** 2 + y ** 2))
#
#
# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

import seaborn as sns

sns.set()

# rng = np.random.RandomState(0)
# x = np.linspace(0, 10, 500)
# y = np.cumsum(rng.randn(500, 6), 0)
#
# plt.plot(x, y)
# plt.legend('ABCDEF', ncol=2, loc='upper left')

# data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]],
#                                      size=2000)
# data = pd.DataFrame(data, columns=['x', 'y'])
# for col in 'xy':
#     plt.hist(data[col], normed=True, alpha=0.5)

# planets = sns.load_dataset('planets')
# planets.head()
#
# with sns.axes_style('white'):
#     g = sns.factorplot("year", data=planets, aspect=4.0, kind='count',
#                        hue='method', order=range(2001, 2015))
#     g.set_ylabels('Number of Planets Discovered')
#
# plt.show()
