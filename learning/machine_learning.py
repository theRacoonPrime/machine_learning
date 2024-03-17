import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px

file_path = '/Users/andrey/Downloads/data_csv.csv'
df = pd.read_csv(file_path)


# def plot_pairplot(df):
#     plt.figure(figsize=(10, 10))
#     sns.pairplot(df)
#     plt.show()
#
#
# plot_pairplot(df)
#
#
# def plot_lineplot(df, x, y, hue=None):
#     plt.figure(figsize=(14,6))
#     sns.lineplot(data=df, x=x, y=y, hue=hue)
#     plt.show()
#
#
# grcprt = df[df['LOCATION'].isin(['PRT', 'GRC'])]
#
# plot_lineplot(grcprt, x='TIME', y='TOTAL_SPEND', hue='LOCATION')
#
#
# sns.lmplot(data=df, x='USD_CAP', y='PC_GDP',
#            height=8, aspect=1.5,
#            scatter_kws={'s': 50, 'alpha': 0.5},
#            line_kws={'lw': 2})
# plt.xlabel('USD per Capita')
# plt.ylabel('% of GDP')
# plt.title('Linear Regression of USD_CAP vs PC_GDP')
#
# plt.tight_layout()
# plt.show()


def plot_lmplot(df, x, y):
    plt.figure(figsize=(12, 8))
    sns.lmplot(data=df, x=x, y=y,
               height=8, aspect=1.5,
               scatter_kws={'s': 50, 'alpha': 0.5},
               line_kws={'lw': 2})
    plt.xlabel('USD per Capita')
    plt.ylabel('% of GDP')
    plt.title('Linear Regression of USD_CAP vs PC_GDP')
    plt.tight_layout()
    plt.show()


plot_lmplot(df, x='USD_CAP', y='PC_GDP')

