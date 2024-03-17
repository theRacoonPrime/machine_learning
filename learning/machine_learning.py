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

# plt.figure(figsize=(10, 10))
# sns.pairplot(df)
# plt.show()
#
# fig = px.line(df, x='TIME', y='TOTAL_SPEND', color='LOCATION')
# fig.show()

# grcprt = df[df['LOCATION'].isin(['PRT', 'GRC'])]
# plt.figure(figsize=(14,6))
# sns.lineplot(data=grcprt, x='TIME', y='TOTAL_SPEND', hue='LOCATION')
# plt.show()
plt.figure(figsize=(14,6))
sns.set_style('darkgrid')
lineplot = sns.lineplot(data=df, x='TIME', y='TOTAL_SPEND', hue='LOCATION')
num_items_per_row = 10
num_legend_rows = -(-len(df['LOCATION'].unique()) // num_items_per_row)
plt.legend(title='LOCATION', bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=num_items_per_row)
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()
# plt.figure(figsize=(12,8))

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


# plt.figure(figsize=(12, 8))
# sns.heatmap(df.corr(), annot=True, cmap='Greens')
# plt.title('Correlation Heatmap of Housing Data')
# plt.show()
