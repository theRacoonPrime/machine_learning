import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score)
import plotly.express as px
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             r2_score)

# File path
file_path = '/Users/andrey/Downloads/data_csv.csv'

# Read data
df = pd.read_csv(file_path)

# List of countries
countries = ['USA', 'CAN', 'GBR', 'FRA', 'DEU', 'ITA', 'JPN']

# Lists to store metrics
mse_list = []
mae_list = []
r2_list = []

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))
axes = axes.flatten()

# Iterate over countries
for i, country in enumerate(countries):
    try:
        # Filter data for the country
        country_df = df[df['LOCATION'] == country]

        # Check if there are enough samples
        if len(country_df) < 2:
            raise ValueError(f"Not enough samples for country {country}")

        # Features and target
        X = country_df[['USD_CAP']]
        y = country_df['PC_GDP']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Ridge regression model
        ridge = Ridge()

        # Grid search for hyperparameter tuning
        param_grid = {'alpha': [0.1, 1, 10]}
        grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Best alpha
        best_alpha = grid_search.best_params_['alpha']

        # Best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Append metrics to lists
        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)

        # Scatter plot
        axes[i].scatter(X_test, y_test, color='blue', label='Actual')
        axes[i].plot(X_test, y_pred, color='red', linewidth=3, label='Predicted')
        axes[i].set_xlabel('USD per Capita')
        axes[i].set_ylabel('% of GDP')
        axes[i].set_title(f'Ridge Regression - {country}')
        axes[i].legend()

        # Print metrics
        print(f"Country: {country}, Best Alpha: {best_alpha}, Mean Squared Error: {mse}, Mean Absolute Error:"
              f" {mae}, R^2 Score: {r2}")

    except Exception as e:
        print(f"Error processing {country}: {str(e)}")
        continue

# Adjust layout
plt.tight_layout()
plt.show()


