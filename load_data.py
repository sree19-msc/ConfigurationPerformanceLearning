import pandas as pd
import os

# Set the correct dataset folder path
dataset_path = "D:/ISE_lab2/lab2/datasets/batlik"  # Adjust based on the dataset folder you want

# List available dataset files
print("Available datasets in 'batlik':", os.listdir(dataset_path))

# Load one dataset (e.g., corona.csv)
file_name = "corona.csv"  # Change to any dataset you want to check
data = pd.read_csv(os.path.join(dataset_path, file_name))

# Display the first 5 rows
print("\nDataset Preview:")
print(data.head())

# Display column names
print("\nColumns in the dataset:")
print(data.columns)
