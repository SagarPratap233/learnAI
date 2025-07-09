import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree # Added plot_tree
import matplotlib.pyplot as plt # For plotting
from sklearn.model_selection import train_test_split # To get data ready again

print("--- Visualizing Your Decision Tree ---")

# 1. Load the Iris dataset (same as before)
try:
    df_iris = pd.read_csv('iris.csv')
except FileNotFoundError:
    print("Error: iris.csv not found. Please ensure it's in the same directory as this script.")
    exit()

# Define Features (X) and Target (y)
X = df_iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df_iris['species']

# Get feature names and class names for better visualization labels
feature_names = X.columns.tolist()
class_names = y.unique().tolist()
class_names.sort() # Sort for consistent order

# Split data (using the same random_state for consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train the Decision Tree Classifier (same as before)
model = DecisionTreeClassifier(random_state=42, max_depth=3) # Added max_depth for a cleaner visual
model.fit(X_train, y_train)

print("\nDecision Tree model trained. Preparing visualization...")

# 3. Visualize the Decision Tree
plt.figure(figsize=(20, 10)) # Adjust figure size for better readability
plot_tree(model,
          feature_names=feature_names,
          class_names=class_names,
          filled=True, # Color nodes to indicate the majority class
          rounded=True, # Round node corners
          fontsize=10) # Adjust font size

plt.title("Decision Tree for Iris Species Classification (Max Depth 3)")
plt.show() # Display the plot

print("\n--- Decision Tree Visualization Completed! ---")