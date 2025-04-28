# Titanic Exploratory Data Analysis (EDA) Script - Ready to Use

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the path to the Titanic dataset CSV file
# This will try to find the dataset in the same directory as this script or in a 'data' folder
default_paths = [
    'train.csv',
    'data/train.csv',
    os.path.join(os.path.dirname(__file__), 'train.csv'),
    os.path.join(os.path.dirname(__file__), 'data', 'train.csv'),
    os.path.expanduser('~/OneDrive/Desktop/train.csv')
]

data_path = None
for path in default_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    raise FileNotFoundError("Titanic dataset CSV file not found. Please place 'train.csv' in the script directory or update the path.")

# Load the Titanic dataset
titanic = pd.read_csv(data_path)

# 1. Basic Data Exploration
print("Data Info:")
print(titanic.info())
print("\nData Description:")
print(titanic.describe())

print("\nValue Counts for 'Survived':")
print(titanic['Survived'].value_counts())

# 2. Visualizations

# Pairplot for selected features
sns.pairplot(titanic[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()

# Heatmap of correlations
plt.figure(figsize=(10, 6))
# Use only numeric columns for correlation heatmap to avoid errors
numeric_cols = titanic.select_dtypes(include=['number'])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Histograms
titanic['Age'].hist(bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

titanic['Fare'].hist(bins=30)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# Boxplots
plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Age', data=titanic)
plt.title('Age Distribution by Passenger Class')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Survived', y='Fare', data=titanic)
plt.title('Fare Distribution by Survival')
plt.show()

# Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=titanic)
plt.title('Age vs Fare Colored by Survival')
plt.show()

# 3. Identify Relationships and Trends

print("\nSurvival Rate by Passenger Class:")
print(titanic.groupby('Pclass')['Survived'].mean())

print("\nSurvival Rate by Sex:")
print(titanic.groupby('Sex')['Survived'].mean())

print("\nSurvival Rate by Number of Siblings/Spouses Aboard:")
print(titanic.groupby('SibSp')['Survived'].mean())

print("\nSurvival Rate by Number of Parents/Children Aboard:")
print(titanic.groupby('Parch')['Survived'].mean())

# Survival rate by fare bins
fare_bins = pd.qcut(titanic['Fare'], 4)
print("\nSurvival Rate by Fare Bins:")
print(titanic.groupby(fare_bins)['Survived'].mean())

# Summary insights
print("\nInsights:")
print("- Higher class passengers had higher survival rates.")
print("- Females had significantly higher survival rates than males.")
print("- Passengers with fewer family members aboard had better survival chances.")
print("- Higher fare paying passengers tended to survive more often.")
print("- Age distribution varies across classes, with younger passengers more likely to survive.")
