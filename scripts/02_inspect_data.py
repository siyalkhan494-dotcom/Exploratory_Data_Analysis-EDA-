"""
STEP 3: Basic Data Inspection
File: scripts/02_inspect_data.py
"""

# Import libraries
import pandas as pd
import numpy as np

def inspect_dataset():
    """
    Perform basic inspection of the Iris dataset
    """
    print("=" * 50)
    print("STEP 3: BASIC DATA INSPECTION")
    print("=" * 50)
    
    # Load the previously saved dataset
    df = pd.read_csv('../output/iris_raw_data.csv')
    
    # Convert species back to categorical (since CSV saved as string)
    df['species'] = pd.Categorical(df['species'])
    
    print("\n📊 DATA INFORMATION")
    print("-" * 40)
    
    # 1. Dataset shape
    print(f"\n1. Dataset Shape:")
    print(f"   - Rows: {df.shape[0]}")
    print(f"   - Columns: {df.shape[1]}")
    
    # 2. Column names and data types
    print(f"\n2. Column Information:")
    print(df.dtypes.to_string())
    
    # 3. Check for missing values
    print(f"\n3. Missing Values Check:")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("   ✅ No missing values found in any column!")
    else:
        print(missing_values[missing_values > 0])
    
    # 4. Check for duplicate rows
    print(f"\n4. Duplicate Rows Check:")
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        print(f"   ✅ No duplicate rows found!")
    else:
        print(f"   ⚠️ Found {duplicates} duplicate rows")
    
    # 5. Check unique values in species column
    print(f"\n5. Species Distribution:")
    species_counts = df['species'].value_counts()
    for species, count in species_counts.items():
        print(f"   - {species}: {count} samples")
    
    # 6. Display first few rows
    print(f"\n6. First 10 Rows:")
    print(df.head(10).to_string())
    
    # 7. Display last few rows
    print(f"\n7. Last 10 Rows:")
    print(df.tail(10).to_string())
    
    # 8. Random sample
    print(f"\n8. Random Sample (5 rows):")
    print(df.sample(5, random_state=42).to_string())
    
    # 9. Memory usage
    print(f"\n9. Memory Usage:")
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum() / 1024  # in KB
    print(f"   Total memory: {total_memory:.2f} KB")
    
    # 10. Summary of numerical columns
    print(f"\n10. Numerical Columns Summary:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"    Numerical columns: {numerical_cols}")
    
    # Save inspection report
    with open('../output/inspection_report.txt', 'w') as f:
        f.write("IRIS DATASET INSPECTION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        f.write(f"Missing values: {missing_values.sum()}\n")
        f.write(f"Duplicate rows: {duplicates}\n\n")
        f.write("Species distribution:\n")
        for species, count in species_counts.items():
            f.write(f"  {species}: {count}\n")
    
    print(f"\n💾 Inspection report saved to: output/inspection_report.txt")
    
    return df

# Run the function
if __name__ == "__main__":
    df = inspect_dataset()