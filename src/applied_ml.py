import os
import tarfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import hashlib
import matplotlib.pyplot as plt
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("..\datasets", "housing")

HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL,
                       housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column,
                           hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


fetch_housing_data(HOUSING_URL, HOUSING_PATH)
df = load_housing_data(HOUSING_PATH)
# df.hist(bins=50, figsize=(20, 15))
# plt.show()

# df_with_id = df.reset_index()
# train_set, test_set = split_train_test(df, 0.2)
# train_set, test_set = split_train_test_by_id(df_with_id, 0.2, 'index')

df['income_cat'] = np.ceil(df['median_income'] / 1.5)
df['income_cat'].where(df['income_cat'] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df['income_cat']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

# print(df['income_cat'].value_counts() / len(df))
# print(strat_train_set['income_cat'].value_counts() / len(strat_train_set))
# print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))

# train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

housing = strat_train_set.copy()
# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population'] / 100, label='population',
             figsize=(10, 7), c='median_house_value',
             cmap=plt.get_cmap('jet'), colorbar=True,
             )

plt.legend()
plt.show()
