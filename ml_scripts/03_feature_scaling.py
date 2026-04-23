"""
STEP 3: FEATURE SCALING (NORMALIZATION)
File: ml_scripts/03_feature_scaling.py

Is script mein:
1. Features ko same scale par laana (Standardization)
2. StandardScaler use karenge (mean=0, std=1)
3. Training data pe fit karenge, test data pe transform
4. Scaled data ko save karenge

Kyun karte hain scaling?
- Models like SVM, KNN distance-based hain, isliye scaling important hai
- Tree-based models (Random Forest, Decision Tree) ko scaling ki zaroorat nahi
- Hum dono ke liye scaling karenge (safe side ke liye)
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

def scale_features():
    """
    Features ko standardize karna
    """
    print("=" * 60)
    print("STEP 3: FEATURE SCALING")
    print("=" * 60)
    
    # 1. Load split data
    print("\n1. Loading split data...")
    X_train = pd.read_csv('../ml_output/X_train.csv')
    X_test = pd.read_csv('../ml_output/X_test.csv')
    
    print(f"   Training features shape: {X_train.shape}")
    print(f"   Testing features shape: {X_test.shape}")
    
    # 2. Check original scales
    print("\n2. Original feature scales (before scaling):")
    for col in X_train.columns:
        print(f"   {col}: min={X_train[col].min():.2f}, max={X_train[col].max():.2f}, mean={X_train[col].mean():.2f}, std={X_train[col].std():.2f}")
    
    # 3. Create and fit scaler on training data
    print("\n3. Creating StandardScaler...")
    scaler = StandardScaler()
    
    # Fit on training data (calculates mean and std)
    X_train_scaled = scaler.fit_transform(X_train)
    print(f"   ✅ Scaler fitted on training data")
    
    # Transform test data using same scaler
    X_test_scaled = scaler.transform(X_test)
    print(f"   ✅ Test data transformed")
    
    # Convert back to DataFrame with column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # 4. Check scaled scales
    print("\n4. Scaled feature scales (after scaling):")
    for col in X_train_scaled.columns:
        print(f"   {col}: min={X_train_scaled[col].min():.2f}, max={X_train_scaled[col].max():.2f}, mean={X_train_scaled[col].mean():.2f}, std={X_train_scaled[col].std():.2f}")
    
    # 5. Save scaled data
    print("\n5. Saving scaled data...")
    
    X_train_scaled.to_csv('../ml_output/X_train_scaled.csv', index=False)
    X_test_scaled.to_csv('../ml_output/X_test_scaled.csv', index=False)
    print(f"   ✅ Saved: ml_output/X_train_scaled.csv")
    print(f"   ✅ Saved: ml_output/X_test_scaled.csv")
    
    # 6. Save the scaler for future use
    joblib.dump(scaler, '../ml_output/scaler.pkl')
    print(f"   ✅ Saved: ml_output/scaler.pkl (for future predictions)")
    
    # 7. Save scaler parameters
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist(),
        'feature_names': X_train.columns.tolist()
    }
    
    with open('../ml_output/scaler_info.txt', 'w') as f:
        f.write("STANDARD SCALER INFORMATION\n")
        f.write("=" * 40 + "\n")
        f.write(f"Features: {scaler_params['feature_names']}\n")
        f.write(f"Means: {scaler_params['mean']}\n")
        f.write(f"Standard Deviations: {scaler_params['std']}\n")
    
    print(f"   ✅ Saved: ml_output/scaler_info.txt")
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("✅ FEATURE SCALING COMPLETED!")
    print("=" * 60)
    print("\n📊 Scaling Summary:")
    print(f"   - Method: StandardScaler (mean=0, std=1)")
    print(f"   - Training data scaled: {X_train_scaled.shape}")
    print(f"   - Testing data scaled: {X_test_scaled.shape}")
    print(f"   - Scaler saved for future predictions")
    
    return X_train_scaled, X_test_scaled, scaler

# Run the function
if __name__ == "__main__":
    X_train_scaled, X_test_scaled, scaler = scale_features()