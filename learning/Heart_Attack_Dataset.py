import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/andrey/Downloads/Medicaldataset.csv')


# plt.figure(figsize=(10, 6))
# sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()
#
#
# plt.figure(figsize=(6, 6))
# sns.countplot(x='Gender', data=df, palette='pastel')
# plt.title('Gender Distribution')
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.show()


# plt.figure(figsize=(10, 6))
# sns.histplot(df['Heart rate'], bins=20, kde=True, color='salmon')
# plt.title('Heart Rate Distribution')
# plt.xlabel('Heart Rate')
# plt.ylabel('Frequency')
# plt.show()


plt.figure(figsize=(10, 6))
sns.kdeplot(df['Systolic blood pressure'], shade=True, color='orange', label='Systolic BP')
sns.kdeplot(df['Diastolic blood pressure'], shade=True, color='purple', label='Diastolic BP')
plt.title('Blood Pressure Distribution')
plt.xlabel('Blood Pressure')
plt.ylabel('Density')
plt.legend()
plt.show()

