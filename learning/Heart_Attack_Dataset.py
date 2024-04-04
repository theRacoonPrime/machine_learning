import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/andrey/Downloads/Medicaldataset.csv')


plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(6, 6))
sns.countplot(x='Gender', data=df, palette='pastel')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['Heart rate'], bins=20, kde=True, color='salmon')
plt.title('Heart Rate Distribution')
plt.xlabel('Heart Rate')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
sns.kdeplot(df['Systolic blood pressure'], shade=True, color='orange', label='Systolic BP')
sns.kdeplot(df['Diastolic blood pressure'], shade=True, color='purple', label='Diastolic BP')
plt.title('Blood Pressure Distribution')
plt.xlabel('Blood Pressure')
plt.ylabel('Density')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['Blood sugar'], bins=20, kde=True, color='green')
plt.title('Blood Sugar Distribution')
plt.xlabel('Blood Sugar')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['CK-MB'], bins=20, kde=True, color='blue')
plt.title('CK-MB Distribution')
plt.xlabel('CK-MB')
plt.ylabel('Frequency')
plt.show()

# Troponin distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Troponin'], bins=20, kde=True, color='red')
plt.title('Troponin Distribution')
plt.xlabel('Troponin')
plt.ylabel('Frequency')
plt.show()

# Result Distribution
plt.figure(figsize=(6, 6))
sns.countplot(x='Result', data=df, palette='pastel')
plt.title('Result Distribution')
plt.xlabel('Result')
plt.ylabel('Count')
plt.show()
