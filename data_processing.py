import pandas as pd
import numpy as np

print("--- Starting Data Processing with Pandas ---")

# --- Simulating a Real-World Dataset ---
# We'll create a DataFrame that mimics data you might get from a CSV file.
# Notice the 'np.nan' which simulates missing data.
data = {
    'CustomerID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Age': [28, 35, 22, np.nan, 45, 30, 55, 29, 40, 23],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female'],
    'SpendingScore': [70, 85, 60, 90, 75, 80, 95, 65, 72, 88],
    'PreferredCategory': ['Electronics', 'Books', 'Electronics', 'Fashion', 'Books', 'Electronics', 'Fashion', 'Books', 'Electronics', 'Books']
}
customers_df = pd.DataFrame(data)

print("\n--- Original Customer Data (First 5 rows) ---\n", customers_df.head())
print("\n--- Original Customer Data Info ---\n")
customers_df.info() # Good for checking data types and non-null counts

# --- Data Exploration ---

# 1. Check for missing values: Essential first step!
print("\n--- Missing values per column ---\n", customers_df.isnull().sum())

# 2. Basic statistics for numerical columns: Understand data distribution.
print("\n--- Descriptive statistics for numeric columns ---\n", customers_df.describe())

# 3. Value counts for categorical data: See the distribution of categories.
print("\n--- Gender distribution ---\n", customers_df['Gender'].value_counts())
print("\n--- Preferred Category distribution ---\n", customers_df['PreferredCategory'].value_counts())

# --- Data Cleaning and Preprocessing ---

# 1. Handling Missing Values (e.g., for 'Age')
# We'll fill missing 'Age' values with the mean age of all customers.
# This is a common strategy, but others exist (median, mode, predicting missing values).
mean_age = customers_df['Age'].mean()
customers_df['Age'].fillna(mean_age, inplace=True) # inplace=True modifies the DataFrame directly

print(f"\n--- Age after filling missing values with mean ({mean_age:.2f}) ---\n", customers_df.head())
print("\n--- Missing values after filling Age (should be 0 for Age) ---\n", customers_df.isnull().sum())


# 2. Encoding Categorical Data (for AI models)
# AI models require numerical input. Text categories must be converted.

# For 'Gender': Use simple mapping if only two categories.
# Male=0, Female=1.
gender_mapping = {'Male': 0, 'Female': 1}
customers_df['Gender_Encoded'] = customers_df['Gender'].map(gender_mapping)
print("\n--- DataFrame with Gender Encoded (0=Male, 1=Female) ---\n", customers_df.head())

# For 'PreferredCategory': Use One-Hot Encoding (creates new binary columns).
# This is preferred for nominal categories (no inherent order).
print("\n--- One-Hot Encoding 'PreferredCategory' ---")
category_dummies = pd.get_dummies(customers_df['PreferredCategory'], prefix='Category', dtype=int) # dtype=int for 0/1 integers

# Concatenate the new encoded columns with the original DataFrame
customers_df = pd.concat([customers_df, category_dummies], axis=1) # axis=1 means column-wise concatenation

# Drop the original 'Gender' and 'PreferredCategory' columns now that they are encoded
customers_df.drop(['Gender', 'PreferredCategory'], axis=1, inplace=True)

print("\n--- DataFrame after One-Hot Encoding and dropping originals (First 5 rows) ---\n", customers_df.head())
print("\n--- Info of Processed DataFrame ---\n")
customers_df.info()


# 3. Feature Scaling (Important for many AI algorithms)
# Features with different scales (e.g., Age 20-50, SpendingScore 60-95) can unfairly influence models.
# Scaling brings them to a similar range. Min-Max Scaling (0 to 1) is a common method.
# Formula: (x - min) / (max - min)

min_score = customers_df['SpendingScore'].min()
max_score = customers_df['SpendingScore'].max()
customers_df['SpendingScore_Scaled'] = (customers_df['SpendingScore'] - min_score) / (max_score - min_score)

# For Age as well, as it's now numerical and filled
min_age = customers_df['Age'].min()
max_age = customers_df['Age'].max()
customers_df['Age_Scaled'] = (customers_df['Age'] - min_age) / (max_age - min_age)

print(f"\n--- SpendingScore Scaled to 0-1 (min={min_score}, max={max_score}) and Age Scaled (min={min_age:.2f}, max={max_age:.2f}) ---\n", customers_df.head())

# You would typically drop the original, unscaled columns if you're using the scaled ones for modeling
customers_df.drop(['SpendingScore', 'Age'], axis=1, inplace=True)

print("\n--- Final Preprocessed DataFrame ready for AI modeling (First 5 rows) ---\n", customers_df.head())

print("\n--- Data Processing with Pandas Completed! ---")