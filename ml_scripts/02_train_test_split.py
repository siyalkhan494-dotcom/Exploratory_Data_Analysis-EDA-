"""
STEP 2: TRAIN-TEST SPLIT
File: ml_scripts/02_train_test_split.py

Is script mein:
1. Data ko training aur testing sets mein divide karenge
2. 80% training, 20% testing (standard practice)
3. Stratified split use karenge (class balance maintain rakhne ke liye)
4. Split data ko save karenge
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_data():
    """
    Data ko train aur test sets mein divide karna
    """
    print("=" * 60)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("=" * 60)
    
    # 1. Load prepared data
    print("\n1. Loading prepared data...")
    X = pd.read_csv('../ml_output/X_features.csv')
    y = pd.read_csv('../ml_output/y_target.csv')
    y = y.values.ravel()  # Convert to 1D array
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # 2. Perform train-test split
    print("\n2. Splitting data (80% train, 20% test)...")
    
    # test_size=0.2 means 20% data testing ke liye
    # random_state=42 ensures same split every time (reproducible)
    # stratify=y ensures same class distribution in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      # 20% for testing
        random_state=42,    # For reproducibility
        stratify=y          # Maintain class balance
    )
    
    print(f"\n   Training set size: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
    print(f"   Testing set size: {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")
    
    # 3. Check class distribution in both sets
    print("\n3. Checking class distribution in train and test sets:")
    
    # Class distribution in original data
    print("\n   Original data:")
    for class_code in [0, 1, 2]:
        count = (y == class_code).sum()
        print(f"     Class {class_code}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Class distribution in training set
    print("\n   Training set:")
    for class_code in [0, 1, 2]:
        count = (y_train == class_code).sum()
        print(f"     Class {class_code}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # Class distribution in testing set
    print("\n   Testing set:")
    for class_code in [0, 1, 2]:
        count = (y_test == class_code).sum()
        print(f"     Class {class_code}: {count} samples ({count/len(y_test)*100:.1f}%)")
    
    # 4. Save split data
    print("\n4. Saving split data...")
    
    # Save training data
    X_train.to_csv('../ml_output/X_train.csv', index=False)
    pd.Series(y_train).to_csv('../ml_output/y_train.csv', index=False, header=['species_code'])
    print(f"   ✅ Saved: ml_output/X_train.csv (features)")
    print(f"   ✅ Saved: ml_output/y_train.csv (target)")
    
    # Save testing data
    X_test.to_csv('../ml_output/X_test.csv', index=False)
    pd.Series(y_test).to_csv('../ml_output/y_test.csv', index=False, header=['species_code'])
    print(f"   ✅ Saved: ml_output/X_test.csv (features)")
    print(f"   ✅ Saved: ml_output/y_test.csv (target)")
    
    # 5. Save split info
    split_info = {
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_percentage': len(X_train)/len(X)*100,
        'test_percentage': len(X_test)/len(X)*100,
        'random_state': 42,
        'stratified': True
    }
    
    with open('../ml_output/split_info.txt', 'w') as f:
        f.write("TRAIN-TEST SPLIT INFORMATION\n")
        f.write("=" * 40 + "\n")
        for key, value in split_info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"   ✅ Saved: ml_output/split_info.txt")
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("✅ TRAIN-TEST SPLIT COMPLETED!")
    print("=" * 60)
    print("\n📊 Split Summary:")
    print(f"   - Training: {len(X_train)} samples")
    print(f"   - Testing: {len(X_test)} samples")
    print(f"   - Class balance maintained: Yes")
    
    return X_train, X_test, y_train, y_test

# Run the function
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_data()