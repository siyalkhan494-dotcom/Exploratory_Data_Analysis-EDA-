"""
STEP 4: Descriptive Statistics
File: scripts/03_statistics.py
"""

# Import libraries
import pandas as pd
import numpy as np

def descriptive_statistics():
    """
    Calculate and display descriptive statistics for Iris dataset
    """
    print("=" * 60)
    print("STEP 4: DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('../output/iris_raw_data.csv')
    
    # Get numerical columns only
    numerical_cols = ['sepal length (cm)', 'sepal width (cm)', 
                      'petal length (cm)', 'petal width (cm)']
    
    print("\n📈 OVERALL STATISTICS (All Species Combined)")
    print("-" * 50)
    
    # 1. Basic descriptive statistics
    stats = df[numerical_cols].describe()
    print(stats)
    
    # 2. Additional statistics
    print("\n📊 ADDITIONAL STATISTICS")
    print("-" * 50)
    
    additional_stats = pd.DataFrame({
        'Variance': df[numerical_cols].var(),
        'Range': df[numerical_cols].max() - df[numerical_cols].min(),
        'Median': df[numerical_cols].median(),
        'Skewness': df[numerical_cols].skew(),
        'Kurtosis': df[numerical_cols].kurtosis()
    })
    print(additional_stats.round(3))
    
    # 3. Statistics by Species
    print("\n📊 STATISTICS BY SPECIES")
    print("-" * 50)
    
    species_list = df['species'].unique()
    
    for species in species_list:
        print(f"\n🔹 {species.upper()}:")
        print("-" * 35)
        species_data = df[df['species'] == species][numerical_cols]
        print(species_data.describe().round(2))
    
    # 4. Summary statistics table
    print("\n📋 SUMMARY TABLE - Mean Values by Species")
    print("-" * 55)
    
    mean_by_species = df.groupby('species')[numerical_cols].mean().round(2)
    print(mean_by_species)
    
    print("\n📋 SUMMARY TABLE - Standard Deviation by Species")
    print("-" * 55)
    
    std_by_species = df.groupby('species')[numerical_cols].std().round(2)
    print(std_by_species)
    
    # 5. Min and Max values with species
    print("\n🔍 EXTREME VALUES")
    print("-" * 50)
    
    for col in numerical_cols:
        max_val = df[col].max()
        min_val = df[col].min()
        max_species = df[df[col] == max_val]['species'].iloc[0]
        min_species = df[df[col] == min_val]['species'].iloc[0]
        
        print(f"\n{col}:")
        print(f"  - Maximum: {max_val} cm ({max_species})")
        print(f"  - Minimum: {min_val} cm ({min_species})")
    
    # 6. Quartile analysis
    print("\n📐 QUARTILE ANALYSIS")
    print("-" * 50)
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q2 = df[col].quantile(0.50)  # median
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        print(f"\n{col}:")
        print(f"  - Q1 (25th percentile): {Q1:.2f} cm")
        print(f"  - Q2 (50th percentile/Median): {Q2:.2f} cm")
        print(f"  - Q3 (75th percentile): {Q3:.2f} cm")
        print(f"  - IQR (Interquartile Range): {IQR:.2f} cm")
    
    # 7. Check for outliers using IQR method
    print("\n⚠️ OUTLIER DETECTION (IQR Method)")
    print("-" * 50)
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if len(outliers) > 0:
            print(f"\n{col}:")
            print(f"  - Lower bound: {lower_bound:.2f}")
            print(f"  - Upper bound: {upper_bound:.2f}")
            print(f"  - Number of outliers: {len(outliers)}")
            print(f"  - Outlier species: {outliers['species'].tolist()}")
        else:
            print(f"\n{col}: ✅ No outliers detected")
    
    # Save statistics to CSV
    # Overall stats
    stats.to_csv('../output/overall_statistics.csv')
    
    # Mean by species
    mean_by_species.to_csv('../output/mean_by_species.csv')
    
    # Additional stats
    additional_stats.to_csv('../output/additional_statistics.csv')
    
    print("\n" + "=" * 60)
    print("💾 Statistics saved to output folder:")
    print("   - output/overall_statistics.csv")
    print("   - output/mean_by_species.csv")
    print("   - output/additional_statistics.csv")
    print("=" * 60)
    
    return stats, mean_by_species

# Run the function
if __name__ == "__main__":
    stats, mean_by_species = descriptive_statistics()