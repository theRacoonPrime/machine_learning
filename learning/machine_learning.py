import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (
    train_test_split, GridSearchCV,cross_val_score)
import plotly.express as px
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             r2_score)


file_path = '/Users/andrey/Downloads/data_csv.csv'
df = pd.read_csv(file_path)


X = df[['USD_CAP']]
y = df['PC_GDP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


ridge = Ridge()


param_grid = {'alpha': [0.1, 1, 10]}  # Adjust the range of alpha values


grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')


grid_search.fit(X_train, y_train)


best_alpha = grid_search.best_params_['alpha']
print("Best Alpha:", best_alpha)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Best Model:", best_model)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)


plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('USD per Capita')
plt.ylabel('% of GDP')
plt.title('Ridge Regression')
plt.legend()
plt.show()
