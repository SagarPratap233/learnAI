import numpy as np
import pandas as pd

print("Numpy Version:", np.__version__)
print("Pandas Version:", pd.__version__)

#numpy quick test 
#create 2x3 matrix 

matrix_np = np.array([[10, 15, 20],[25, 30, 35]])
print("\nNumpy Matrix:\n", matrix_np)
print("Shape:", matrix_np.shape)
print("Sum of elements", np.sum(matrix_np))

# --- Pandas Quick Test ---
# Create a simple DataFrame
data_pd = {
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'Price': [1200, 25, 75, 300],
    'Quantity': [10, 50, 30, 15]
}

df_pd = pd.DataFrame(data_pd)
print("\nPandas DataFrame:\n", df_pd)

# Calculate total sales for each product
df_pd['Total Sales'] = df_pd['Price'] * df_pd['Quantity']
print("\nDataFrame with Total Sales:\n", df_pd)

print("\nSuccessfully tested NumPy and Pandas in your virtual environment!")