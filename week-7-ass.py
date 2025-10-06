# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print(df.head())
    print(df.info())
    print(df.isnull().sum())

    df.dropna(inplace=True)  # No missing values, but included for completeness
except Exception as e:
    print("Error loading dataset:", e)

# Task 2: Basic Data Analysis
print(df.describe())
grouped = df.groupby('species')['petal length (cm)'].mean()
print(grouped)

# Task 3: Data Visualization
plt.figure(figsize=(12, 8))

# Line chart (simulated time-series using index)
plt.subplot(2, 2, 1)
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()

# Bar chart
plt.subplot(2, 2, 2)
grouped.plot(kind='bar', color='skyblue')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')

# Histogram
plt.subplot(2, 2, 3)
plt.hist(df['sepal width (cm)'], bins=10, color='lightgreen')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')

# Scatter plot
plt.subplot(2, 2, 4)
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')

plt.tight_layout()
plt.show()