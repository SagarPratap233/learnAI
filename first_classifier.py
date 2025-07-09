import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # This is our AI model algorithm!
from sklearn.metrics import accuracy_score

print("--- Starting First AI Classifier ---")

# 1. Load the Iris dataset
# We try to load the file, handling case where it might not be found
try:
    df_iris = pd.read_csv('iris.csv')
    print("\nSuccessfully loaded iris.csv!")
    print("Original Iris DataFrame (First 5 rows):\n", df_iris.head())
except FileNotFoundError:
    print("Error: iris.csv not found. Please ensure it's in the same directory as this script.")
    print("You can download it from: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    exit() # Stop the script if the file isn't there

# 2. Define Features (X) and Target (y)
# Features (X): These are the input data columns the model will use to learn.
# For Iris, these are the measurements of the flowers.
X = df_iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# Target (y): This is the output column the model will try to predict.
# For Iris, it's the species name.
y = df_iris['species']

print("\nFeatures (X) head:\n", X.head())
print("\nTarget (y) head:\n", y.head())

# 3. Split the data into training and testing sets
# It's crucial to split your data so the model learns from one portion (training data)
# and is then evaluated on a completely separate, unseen portion (testing data).
# This helps ensure the model can generalize to new, real-world data.
#
# test_size=0.3 means 30% of the data will be held back for testing, 70% for training.
# random_state=42 is a "seed" for the random number generator. Using it ensures
# that your data split will be exactly the same every time you run the script.
# This makes your results reproducible.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# 4. Choose and Train an AI Model: Decision Tree Classifier
# A Decision Tree is a powerful, yet easy-to-understand model for classification.
# It learns a series of if-then-else rules from your data.
model = DecisionTreeClassifier(random_state=42) # Again, random_state for reproducibility of the model's internal randomness

print("\n--- Training the Decision Tree Classifier ---")
model.fit(X_train, y_train) # This is the "learning" step! The model adjusts its internal rules based on X_train and y_train.

print("Model training complete!")

# 5. Make predictions on the testing data
# We now ask our trained model to predict the species for the X_test data,
# which it has never seen during training.
y_pred = model.predict(X_test)

print("\nFirst 5 Actual labels (from y_test):\n", y_test.head().tolist())
print("First 5 Predicted labels (from y_pred):\n", y_pred[:5].tolist())

# 6. Evaluate the model's performance
# Accuracy Score: The simplest evaluation metric for classification.
# It's the proportion of correctly predicted instances out of the total instances.
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy on Test Data: {accuracy:.2f}") # Format to 2 decimal places

print("\n--- First AI Classifier Completed! ---")