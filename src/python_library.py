import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn

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

df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering',
                              'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})

df3 = pd.merge(df1, df2)
print(df3)
