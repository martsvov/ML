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

index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'],
                                      ['HR', 'Temp']],
                                     names=['subject', 'type'])

data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37

health_data = pd.DataFrame(data, index=index, columns=columns)

print(health_data)
