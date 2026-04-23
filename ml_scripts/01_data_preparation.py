"""
STEP 1: DATA PREPARATION FOR MACHINE LEARNING
File: ml_scripts/01_data_preparation.py

Is script mein:
1. Clean data load karenge
2. Features (X) aur target (y) alag karenge
3. Species names ko numbers mein convert karenge (encoding)
4. Data ka final shape check karenge
"""

# Import libraries
import pandas as pd
import numpy as np
import os

# Create output directories if not exist
os.makedirs('../ml_output', exist_ok=True)
os.makedirs('../ml_output/evaluation', exist_ok=True)

def prepare_data():
    """
    Data ko machine learning ke liye prepare karna
    """
    print("=" * 60)
    print("STEP 1: DATA PREPARATION FOR MACHINE LEARNING")
    print("=" * 60)
    
    # 1. Load the cleaned data (EDA se jo save kiya tha)
    print("\n1. Loading cleaned dataset...")
    df = pd.read_csv('../output/iris_raw_data.csv')
    print(f"   ✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Display first few rows to verify
    print("\n2. First 5 rows of data:")
    print(df.head())
    
    # 3. Separate features (X) and target (y)
    print("\n3. Separating features and target...")
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                    'petal length (cm)', 'petal width (cm)']
    
    X = df[feature_cols].copy()  # Features (inputs)
    y = df['species'].copy()      # Target (output)
    
    print(f"   Features (X) shape: {X.shape}")
    print(f"   Target (y) shape: {y.shape}")
    print(f"   Feature columns: {list(X.columns)}")
    
    # 4. Encode target variable (species names to numbers)
    print("\n4. Encoding target variable (species → numbers)...")
    species_mapping = {
        'setosa': 0,
        'versicolor': 1,
        'virginica': 2
    }
    
    y_encoded = y.map(species_mapping)
    
    print(f"   Species mapping: {species_mapping}")
    print(f"   Unique values in y_encoded: {y_encoded.unique()}")
    
    # 5. Check class distribution
    print("\n5. Class distribution after encoding:")
    for species, code in species_mapping.items():
        count = (y_encoded == code).sum()
        print(f"   {species} (code {code}): {count} samples")
    
    # 6. Save prepared data for later use
    print("\n6. Saving prepared data...")
    
    # Save features
    X.to_csv('../ml_output/X_features.csv', index=False)
    print(f"   ✅ Saved: ml_output/X_features.csv")
    
    # Save target
    y_encoded.to_csv('../ml_output/y_target.csv', index=False, header=['species_code'])
    print(f"   ✅ Saved: ml_output/y_target.csv")
    
    # Save feature names for reference
    with open('../ml_output/feature_names.txt', 'w') as f:
        f.write("Feature columns for Iris dataset:\n")
        f.write("=" * 40 + "\n")
        for i, col in enumerate(feature_cols):
            f.write(f"{i}: {col}\n")
    
    print(f"   ✅ Saved: ml_output/feature_names.txt")
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("✅ DATA PREPARATION COMPLETED!")
    print("=" * 60)
    print("\n📊 Summary:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Classes: {len(species_mapping)}")
    print(f"   - Data saved in: ml_output/ folder")
    
    return X, y_encoded, species_mapping

# Run the function
if __name__ == "__main__":
    X, y, mapping = prepare_data()