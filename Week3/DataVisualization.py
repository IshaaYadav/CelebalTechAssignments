# SECTION 1: Setup & Imports

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visual aesthetics
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.0)
plt.rcParams['figure.figsize'] = (10, 6)

print("="*60)
print("           TITANIC DATASET DATA VISUALIZATION")
print("="*60)
print()

# SECTION 2: Load & Explore Dataset

# Load Titanic dataset from seaborn
df = sns.load_dataset("titanic")

# Show basic structure
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# SECTION 3: Data Cleaning & Preparation

# Create a copy to avoid warnings
df = df.copy()

# Drop columns that are less useful for visualization (you can keep them if needed)
df = df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'], axis=1, errors='ignore')

# Handle missing values

# Fill missing 'age' with median
df['age'] = df['age'].fillna(df['age'].median())

# Fill missing 'embarked' with mode
if df['embarked'].isnull().any():
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Convert to appropriate data types
df['embarked'] = df['embarked'].astype('category')
df['sex'] = df['sex'].astype('category')
df['survived'] = df['survived'].astype('category')
df['pclass'] = df['pclass'].astype('category')

# Feature Engineering (optional but helps analysis)
# Create an 'age_group' column for better visual categorization
df['age_group'] = pd.cut(df['age'],
                         bins=[0, 12, 18, 35, 60, 100],
                         labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

# Create 'family_size' as sibsp + parch + 1 (the person themself)
df['family_size'] = df['sibsp'] + df['parch'] + 1

print("\nData after cleaning and preparation:")
print(df.head())
print("\nAny remaining nulls:")
print(df.isnull().sum())

# SECTION 4: Univariate Analysis

# 1. Countplot - Gender distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', data=df, palette='pastel')
plt.title('Gender Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Countplot - Survival distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='survived', data=df, palette='Set2')
plt.title('Survival Distribution (0 = No, 1 = Yes)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 3. Countplot - Passenger Class
plt.figure(figsize=(10, 6))
sns.countplot(x='pclass', data=df, palette='coolwarm')
plt.title('Passenger Class Distribution')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 4. Histogram - Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True, color='skyblue', bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 5. Boxplot - Fare distribution (outliers)
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['fare'], color='orange')
plt.title('Fare Distribution with Outliers')
plt.xlabel('Fare')
plt.tight_layout()
plt.show()

# 6. Countplot - Age groups
plt.figure(figsize=(10, 6))
sns.countplot(x='age_group', data=df, palette='Set3')
plt.title('Passengers by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# SECTION 5: Bivariate Analysis

# 1. Survival rate by gender
plt.figure(figsize=(10, 6))
sns.barplot(x='sex', y='survived', data=df, palette='pastel', estimator=lambda x: (x==1).mean())
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.tight_layout()
plt.show()

# 2. Survival by Passenger Class
plt.figure(figsize=(10, 6))
sns.barplot(x='pclass', y='survived', data=df, palette='coolwarm', estimator=lambda x: (x==1).mean())
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.tight_layout()
plt.show()

# 3. Age vs Fare by Survival
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='fare', hue='survived', data=df, palette='Set1', alpha=0.7)
plt.title('Age vs Fare by Survival')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.tight_layout()
plt.show()

# 4. Survival by Embarkation Port
plt.figure(figsize=(10, 6))
sns.barplot(x='embarked', y='survived', data=df, palette='Set2', estimator=lambda x: (x==1).mean())
plt.title('Survival Rate by Embarkation Port')
plt.xlabel('Port of Embarkation')
plt.ylabel('Survival Rate')
plt.tight_layout()
plt.show()

# 5. Violinplot - Age distribution by survival
plt.figure(figsize=(10, 6))
sns.violinplot(x='survived', y='age', data=df, palette='Set3')
plt.title('Age Distribution by Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.tight_layout()
plt.show()

# 6. Heatmap - Correlation matrix
# Only numerical columns
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# SECTION 6: Multivariate Analysis

# 1. Catplot: Survival by class and gender
g = sns.catplot(data=df, x='pclass', hue='sex', col='survived',
                kind='count', palette='pastel', height=5, aspect=1)
g.fig.suptitle("Survival Count by Class and Gender", y=1.02)
plt.show()

# 2. Catplot: Age group, survival, and embarkation
g = sns.catplot(data=df, x='age_group', hue='survived', col='embarked',
                kind='count', palette='Set2', height=5, aspect=1)
g.fig.suptitle("Survival by Age Group & Embarkation Port", y=1.02)
plt.show()

# 3. Pairplot: Numerical relationships colored by survival
# Limit to a few numeric features
g = sns.pairplot(df[['age', 'fare', 'family_size', 'survived']], hue='survived',
                 palette='coolwarm', corner=True)
g.fig.suptitle("Pairplot of Numerical Features by Survival", y=1.02)
plt.show()

# 4. Swarmplot: Fare distribution by class and survival (limit data for readability)
plt.figure(figsize=(12, 6))
# Sample data to avoid overcrowding
df_sample = df.sample(n=min(500, len(df)), random_state=42)
sns.swarmplot(data=df_sample, x='pclass', y='fare', hue='survived', palette='Set1', dodge=True)
plt.title('Fare by Class and Survival (Sample)')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.tight_layout()
plt.show()

# SECTION 7: Advanced & Creative Visualizations

# 1. FacetGrid: Age vs Fare, split by survival
g = sns.FacetGrid(df, col="survived", height=5, aspect=1)
g.map_dataframe(sns.scatterplot, x="age", y="fare", alpha=0.7, color='teal')
g.set_axis_labels("Age", "Fare")
g.set_titles("Survived: {col_name}")
g.fig.suptitle("Fare vs Age by Survival", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# 2. KDE Plot: Age distribution by survival status
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="age", hue="survived", fill=True, palette="coolwarm", alpha=0.5)
plt.title("Age Distribution by Survival Status")
plt.xlabel("Age")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# 3. Countplot: Family Size vs Survival
# Group family size into small/medium/large
df['family_group'] = pd.cut(df['family_size'],
                            bins=[0, 1, 4, 11],
                            labels=['Solo', 'Small Family', 'Large Family'])

plt.figure(figsize=(10, 6))
sns.countplot(x='family_group', hue='survived', data=df, palette='Set2')
plt.title('Survival Count by Family Group Size')
plt.xlabel('Family Group Size')
plt.ylabel('Passenger Count')
plt.tight_layout()
plt.show()

# 4. Pie plot: Gender proportion in survivors
# Fixed: Use numeric comparison instead of string
survivors_by_gender = df[df['survived'] == 1]['sex'].value_counts()

plt.figure(figsize=(8, 8))
colors = sns.color_palette('pastel')
plt.pie(survivors_by_gender, labels=survivors_by_gender.index, autopct='%1.1f%%', colors=colors)
plt.title("Gender Distribution Among Survivors")
plt.tight_layout()
plt.show()

print("Analysis completed successfully!")
