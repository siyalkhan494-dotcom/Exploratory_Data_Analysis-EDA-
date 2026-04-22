"""
STEP 2: Load the Iris Dataset
File: scripts/01_load_data.py
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

# Create output directories if they don't exist
os.makedirs('../output', exist_ok=True)
os.makedirs('../data', exist_ok=True)

def load_iris_dataset():
    """
    Load Iris dataset from sklearn and convert to DataFrame
    """
    print("=" * 50)
    print("STEP 2: LOADING IRIS DATASET")
    print("=" * 50)
    
    # Load the dataset
    iris = load_iris()
    
    # Convert to DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Add species column as categorical
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Display basic info
    print(f"\n✅ Dataset loaded successfully!")
    print(f"\nShape of dataset: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print(f"\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")
    
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nLast 5 rows:")
    print(df.tail())
    
    # Save raw data to CSV
    df.to_csv('../output/iris_raw_data.csv', index=False)
    print(f"\n💾 Raw data saved to: output/iris_raw_data.csv")
    
    return df

# Run the function
if __name__ == "__main__":
    df = load_iris_dataset()