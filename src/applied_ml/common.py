import os
import tarfile
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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

# housing = strat_train_set.copy()
# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
#              s=housing['population'] / 100, label='population',
#              figsize=(10, 7), c='median_house_value',
#              cmap=plt.get_cmap('jet'), colorbar=True,
#              )

# plt.legend()

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))

# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

# housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
# housing["population_per_household"] = housing["population"] / housing["households"]

# corr_matrix = housing.corr()
# corr_matrix["median_house_value"].sort_values(ascending=False)
# plt.show()

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing_num = housing.drop("ocean_proximity", axis=1)
#
# imputer = SimpleImputer(strategy="median")
# imputer.fit(housing_num)

# X = imputer.transform(housing_num)
# housing_transform = pd.DataFrame(X, columns=housing_num.columns)

# housing_cat = housing["ocean_proximity"]
# housing_cat_encoded, housing_categories = housing_cat.factorize()
#
# encoder = OneHotEncoder()
# housing_cat_hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, Х, y=None):
        rooms_per_household = Х[:, rooms_ix] / Х[:, household_ix]
        population_per_household = Х[:, population_ix] / Х[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = Х[:, bedrooms_ix] / Х[:, rooms_ix]
            return np.c_[Х, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[Х, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

# num_attribs = slice(0, -1)
# cat_attribs = slice(-1)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
#
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Суммы оценок: " , scores)
    print("Cpeднee: " , scores.mean())
    print("Стандартное отклонение: ", scores.std())


display_scores(tree_rmse_scores)

print()
