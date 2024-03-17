import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file_path = '/Users/andrey/Downloads/data_csv.csv'
data = pd.read_csv(file_path)

data_show = pd.DataFrame({
    'Location': ['AUS', 'LTU', 'RUS'],
    'Time': [1971, 1972, 1975],
    'PC_Health': [15.992, 11.849, 28.942],
})
print(data.info())