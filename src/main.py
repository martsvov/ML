import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data\Part 2 - Regression\Section 6 - Polynomial Regression\Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_x = LabelEncoder()
# x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
# onehotencoder = OneHotEncoder(categorical_features=[3])
# x = onehotencoder.fit_transform(x).toarray()
#
# x = x[:, 1:]
#
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y,
#     test_size=0.2,
#     random_state=0
# )

# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# x_train = sc_x.fit_transform(x_train)
# x_test = sc_x.transform(x_test)
#
# import statsmodels.formula.api as sm
# x = np.append(arr=np.ones((50, 1)).astype(int), values=x, axis=1)
# x_opt = x[:, [0, 3, 5]]
# regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()

# plt.scatter(x_test, y_test, color = 'red')
# plt.plot(x_train, regressor.predict(x_train), color = 'blue')
# plt.title('Salary vs Experience (Test set)')
# plt.xlabel('Years or Experience')
# plt.ylabel('Salary')
# plt.show()

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# plt.scatter(x, y, color = 'red')
# plt.plot(x, lin_reg.predict(x), color = 'blue')
# plt.title('Salary vs Level (Linear)')
# plt.xlabel('Level')
# plt.ylabel('Salary')
# plt.show()

# x_grid = np.arange(min(x), max(x)+0.1, 0.1)
# x_grid = np.reshape(x_grid, (len(x_grid), 1))
#
# plt.scatter(x, y, color = 'red')
# plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
# plt.title('Salary vs Level (Polynomial)')
# plt.xlabel('Level')
# plt.ylabel('Salary')
# plt.show()

print(lin_reg.predict([[6.5]]))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))