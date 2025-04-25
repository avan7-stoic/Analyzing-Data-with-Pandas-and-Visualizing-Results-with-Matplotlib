# 📦 Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Optional: Display plots inside notebook (Jupyter)
# %matplotlib inline

# 🎯 Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset
    iris = load_iris()

    # Convert to pandas DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("✅ Dataset loaded successfully!")

except Exception as e:
    print("❌ Error loading dataset:", e)

# Display first few rows
print("\n🔍 First 5 rows:")
print(df.head())

# Check structure, types, and missing values
print("\n🧠 Info about dataset:")
print(df.info())

print("\n🧼 Checking for missing values:")
print(df.isnull().sum())

# No missing values in Iris dataset, but if there were:
# df = df.dropna()  # OR df.fillna(method='ffill', inplace=True)

# 🎯 Task 2: Basic Data Analysis

# Basic statistics
print("\n📊 Descriptive statistics:")
print(df.describe())

# Grouping: Mean of features per species
print("\n📚 Mean of features grouped by species:")
print(df.groupby("species").mean())

# ✨ Finding: Check which species has the longest petal length
longest_petal_species = df.groupby("species")["petal length (cm)"].mean().idxmax()
print(f"\n🔎 Species with longest average petal length: {longest_petal_species}")

# 🎯 Task 3: Data Visualization

# Set Seaborn theme
sns.set(style="whitegrid")

# 1️⃣ Line Chart: Simulate time series by assigning index as "days"
df['day'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')

plt.figure(figsize=(10, 5))
plt.plot(df['day'], df['sepal length (cm)'], label='Sepal Length')
plt.plot(df['day'], df['petal length (cm)'], label='Petal Length')
plt.title("Line Chart - Sepal and Petal Length Over Time")
plt.xlabel("Date")
plt.ylabel("Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 2️⃣ Bar Chart: Average petal length by species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df, palette='Set2')
plt.title("Bar Chart - Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3️⃣ Histogram: Distribution of sepal width
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title("Histogram - Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4️⃣ Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Scatter Plot - Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 🎯 Final Observations
print("\n📌 Observations:")
print("- There are no missing values in the dataset.")
print("- Setosa species has significantly shorter petal lengths than others.")
print("- Sepal length and petal length show a positive correlation.")
print("- Versicolor and Virginica are closer in feature values compared to Setosa.")

