import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

sns.set()

births = pd.read_csv('../data/births.csv')
births['decade'] = 10 * (births['year'] // 10)
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

index = pd.DatetimeIndex(['2014-07-04', '2014-08-04', '2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)

print(data)
