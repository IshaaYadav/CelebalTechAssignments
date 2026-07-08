# SECTION 1: Setup & Load Data

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set plot style
sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)

# Create visuals folder if not exists
if not os.path.exists("visuals"):
    os.makedirs("visuals")

# Load Titanic dataset from seaborn
df = sns.load_dataset("titanic")

# Basic Data Overview
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types and Non-Null Counts:")
print(df.info())

print("\nBasic Statistical Summary:")
print(df.describe(include="all"))

# SECTION 2: Data Profiling

# Unique value counts for each column
print("\nUnique Values per Column:")
print(df.nunique().sort_values(ascending=False))

# Columns with missing values
print("\nMissing Values per Column:")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing)

# Column data types summary
print("\nColumn Types Summary:")
print(df.dtypes.value_counts())

# Count of categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical Columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical Columns ({len(numerical_cols)}): {numerical_cols}")

# Summary statistics for numerical features
print("\nNumerical Summary:")
print(df[numerical_cols].describe())

# Summary statistics for categorical features
print("\nCategorical Summary:")
for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False))

# SECTION 2.1: Generate Profiling Report

from ydata_profiling import ProfileReport

print("\nGenerating full profiling report...")

# Create the profile
profile = ProfileReport(df, title="Titanic Dataset EDA Report", explorative=True)

# Save to file
profile.to_file("titanic_profiling_report.html")

print("Profiling report saved as 'titanic_profiling_report.html'")

# SECTION 3: Missing Value Analysis

# Table of missing values
missing_df = df.isnull().sum().reset_index()
missing_df.columns = ['Column', 'Missing_Values']
missing_df['% Missing'] = (missing_df['Missing_Values'] / len(df)) * 100
missing_df = missing_df[missing_df['Missing_Values'] > 0].sort_values(by='% Missing', ascending=False)

print("\nColumns with Missing Values:")
print(missing_df)

# Visualize missing values using seaborn heatmap
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig("visuals/missing_values_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# Bar plot of % missing per column
sns.barplot(x='% Missing', y='Column', data=missing_df, palette='magma')
plt.title('Percentage of Missing Values by Column')
plt.tight_layout()
plt.savefig("visuals/missing_values_barplot.png", dpi=300, bbox_inches='tight')
plt.show()

# SECTION 4: Univariate Analysis

# ----------------------------
# NUMERICAL FEATURES
# ----------------------------

# Histogram with KDE for Age
sns.histplot(df['age'], kde=True, color='skyblue', bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('visuals/age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Histogram with KDE for Fare
sns.histplot(df['fare'], kde=True, color='salmon', bins=30)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('visuals/fare_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Boxplot for Fare (to check outliers)
sns.boxplot(x=df['fare'], color='orange')
plt.title('Fare Outlier Detection')
plt.xlabel('Fare')
plt.tight_layout()
plt.savefig('visuals/fare_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# CATEGORICAL FEATURES
# ----------------------------

# Countplot for Sex
sns.countplot(x='sex', data=df, palette='pastel')
plt.title('Gender Count')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visuals/gender_count.png', dpi=300, bbox_inches='tight')
plt.show()

# Countplot for Passenger Class
sns.countplot(x='pclass', data=df, palette='coolwarm')
plt.title('Passenger Class Count')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visuals/class_count.png', dpi=300, bbox_inches='tight')
plt.show()

# Countplot for Embarked
sns.countplot(x='embarked', data=df, palette='Set2')
plt.title('Embarkation Port Count')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visuals/embarked_count.png', dpi=300, bbox_inches='tight')
plt.show()

# Countplot for Survival
sns.countplot(x='survived', data=df, palette='Set1')
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visuals/survival_count.png', dpi=300, bbox_inches='tight')
plt.show()

# SECTION 5: Bivariate Analysis

# 1. Survival Rate by Gender
sns.barplot(x='sex', y='survived', data=df, palette='pastel')
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Probability')
plt.tight_layout()
plt.savefig("visuals/survival_by_gender.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Survival Rate by Passenger Class
sns.barplot(x='pclass', y='survived', data=df, palette='coolwarm')
plt.title('Survival Rate by Class')
plt.ylabel('Survival Probability')
plt.xlabel('Passenger Class')
plt.tight_layout()
plt.savefig("visuals/survival_by_class.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Age Distribution by Survival (Violin Plot)
sns.violinplot(x='survived', y='age', data=df, palette='Set2')
plt.title('Age Distribution by Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.tight_layout()
plt.savefig("visuals/age_violin_by_survival.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Fare vs Survival (Boxplot)
sns.boxplot(x='survived', y='fare', data=df, palette='Set3')
plt.title('Fare vs Survival')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.tight_layout()
plt.savefig("visuals/fare_by_survival.png", dpi=300, bbox_inches='tight')
plt.show()

# 5. Survival by Embarked Port
sns.barplot(x='embarked', y='survived', data=df, palette='magma')
plt.title('Survival Rate by Embarkation Port')
plt.xlabel('Embarked')
plt.ylabel('Survival Probability')
plt.tight_layout()
plt.savefig("visuals/survival_by_embarked.png", dpi=300, bbox_inches='tight')
plt.show()

# 6. Heatmap: Correlation Between Numerical Features
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("visuals/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# SECTION 6: Outlier Detection

# Boxplot - Age
sns.boxplot(x=df['age'], color='lightblue')
plt.title('Outlier Detection in Age')
plt.xlabel('Age')
plt.tight_layout()
plt.savefig('visuals/age_outliers_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Boxplot - Fare
sns.boxplot(x=df['fare'], color='tomato')
plt.title('Outlier Detection in Fare')
plt.xlabel('Fare')
plt.tight_layout()
plt.savefig('visuals/fare_outliers_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# IQR Method for Age
Q1_age = df['age'].quantile(0.25)
Q3_age = df['age'].quantile(0.75)
IQR_age = Q3_age - Q1_age

lower_bound_age = Q1_age - 1.5 * IQR_age
upper_bound_age = Q3_age + 1.5 * IQR_age

outliers_age = df[(df['age'] < lower_bound_age) | (df['age'] > upper_bound_age)]

print(f"\nAge Outliers Detected: {outliers_age.shape[0]} rows")
print(outliers_age[['age', 'survived', 'pclass']].head())

# IQR Method for Fare
Q1_fare = df['fare'].quantile(0.25)
Q3_fare = df['fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare

lower_bound_fare = Q1_fare - 1.5 * IQR_fare
upper_bound_fare = Q3_fare + 1.5 * IQR_fare

outliers_fare = df[(df['fare'] < lower_bound_fare) | (df['fare'] > upper_bound_fare)]

print(f"\nFare Outliers Detected: {outliers_fare.shape[0]} rows")
print(outliers_fare[['fare', 'survived', 'pclass']].head())
