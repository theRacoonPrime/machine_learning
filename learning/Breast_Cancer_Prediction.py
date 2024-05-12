import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("/Users/andrey/Downloads/data.csv")
# print(data.shape)
#
# print(data.describe().T)

data = data.drop_duplicates()

data = pd.get_dummies(data,drop_first=True)

# print(data)

data["diagnosis_M"].unique()
# print(sns.displot(data=data, x="diagnosis_M"))

plt.figure(figsize=(25,20))
print(sns.heatmap(data=data.corr(), annot=True))
