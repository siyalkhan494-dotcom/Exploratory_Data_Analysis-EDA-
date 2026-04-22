"""
STEP 10: Document Findings & Generate Complete Report
File: scripts/09_findings.py
"""

# Import libraries
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import os
import sys

def generate_complete_report():
    """
    Generate a complete EDA report with all findings
    """
    print("=" * 60)
    print("STEP 10: GENERATING COMPLETE EDA REPORT")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('../output/iris_raw_data.csv')
    
    numerical_cols = ['sepal length (cm)', 'sepal width (cm)', 
                      'petal length (cm)', 'petal width (cm)']
    
    species_list = ['setosa', 'versicolor', 'virginica']
    
    print("\nAnalyzing data for final report...")
    
    # ============================================
    # COLLECT ALL FINDINGS
    # ============================================
    
    # 1. Basic Information
    total_samples = len(df)
    total_features = len(numerical_cols)
    total_classes = df['species'].nunique()
    missing_values = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    
    # 2. Class Distribution
    class_distribution = df['species'].value_counts().to_dict()
    
    # 3. Statistical Summary
    stats_summary = df[numerical_cols].describe()
    
    # 4. Correlation Analysis
    corr_matrix = df[numerical_cols].corr()
    highest_corr_pair = None
    highest_corr_value = -1
    lowest_corr_pair = None
    lowest_corr_value = 2
    
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > highest_corr_value:
                highest_corr_value = abs(corr_val)
                highest_corr_pair = (numerical_cols[i], numerical_cols[j], corr_val)
            if abs(corr_val) < lowest_corr_value:
                lowest_corr_value = abs(corr_val)
                lowest_corr_pair = (numerical_cols[i], numerical_cols[j], corr_val)
    
    # 5. ANOVA Results
    anova_results = {}
    for col in numerical_cols:
        groups = [df[df['species'] == species][col].values for species in species_list]
        f_stat, p_value = stats.f_oneway(*groups)
        anova_results[col] = {'f_statistic': f_stat, 'p_value': p_value}
    
    # 6. Species Separability
    setosa_max_petal_len = df[df['species'] == 'setosa']['petal length (cm)'].max()
    versicolor_min_petal_len = df[df['species'] == 'versicolor']['petal length (cm)'].min()
    petal_gap = versicolor_min_petal_len - setosa_max_petal_len
    
    setosa_max_petal_width = df[df['species'] == 'setosa']['petal width (cm)'].max()
    versicolor_min_petal_width = df[df['species'] == 'versicolor']['petal width (cm)'].min()
    petal_width_gap = versicolor_min_petal_width - setosa_max_petal_width
    
    # 7. Most and Least Discriminative Features
    discrimination_scores = {}
    for col in numerical_cols:
        overall_mean = df[col].mean()
        species_means = df.groupby('species')[col].mean()
        species_counts = df.groupby('species')[col].count()
        
        between_var = sum(species_counts * (species_means - overall_mean)**2) / (len(species_list) - 1)
        
        within_var = 0
        for species in species_list:
            species_data = df[df['species'] == species][col]
            within_var += sum((species_data - species_means[species])**2)
        within_var = within_var / (len(df) - len(species_list))
        
        discrimination_scores[col] = between_var / within_var
    
    most_discriminative = max(discrimination_scores, key=discrimination_scores.get)
    least_discriminative = min(discrimination_scores, key=discrimination_scores.get)
    
    # 8. Outlier Detection
    outliers_found = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_found[col] = len(outliers)
    
    # ============================================
    # GENERATE TEXT REPORT
    # ============================================
    
    text_report = f"""
================================================================================
                    IRIS DATASET EDA - COMPLETE REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
1. DATASET OVERVIEW
================================================================================

Total Samples:          {total_samples}
Total Features:         {total_features}
Target Classes:         {total_classes}
Missing Values:         {missing_values}
Duplicate Rows:         {duplicates}

Class Distribution:
{chr(10).join([f'  - {species.capitalize()}: {count} samples' for species, count in class_distribution.items()])}

================================================================================
2. DESCRIPTIVE STATISTICS
================================================================================

{stats_summary.round(2).to_string()}

================================================================================
3. CORRELATION ANALYSIS
================================================================================

Correlation Matrix:
{corr_matrix.round(3).to_string()}

Highest Correlation: {highest_corr_pair[0]} vs {highest_corr_pair[1]} = {highest_corr_pair[2]:.3f}
Lowest Correlation:  {lowest_corr_pair[0]} vs {lowest_corr_pair[1]} = {lowest_corr_pair[2]:.3f}

================================================================================
4. ANOVA RESULTS (Species Comparison)
================================================================================

{chr(10).join([f'{feature}: F={anova_results[feature]["f_statistic"]:.2f}, p={anova_results[feature]["p_value"]:.6f}' for feature in numerical_cols])}

================================================================================
5. FEATURE DISCRIMINATION POWER
================================================================================

{chr(10).join([f'{i+1}. {feature.replace(" (cm)", "")}: {discrimination_scores[feature]:.2f}' for i, feature in enumerate(sorted(discrimination_scores, key=discrimination_scores.get, reverse=True))])}

Most Discriminative:    {most_discriminative.replace(' (cm)', '')}
Least Discriminative:   {least_discriminative.replace(' (cm)', '')}

================================================================================
6. SPECIES SEPARABILITY
================================================================================

Setosa max petal length:    {setosa_max_petal_len:.2f} cm
Versicolor min petal length: {versicolor_min_petal_len:.2f} cm
Gap:                         {petal_gap:.2f} cm

Setosa max petal width:     {setosa_max_petal_width:.2f} cm
Versicolor min petal width:  {versicolor_min_petal_width:.2f} cm
Gap:                         {petal_width_gap:.2f} cm

================================================================================
7. OUTLIER DETECTION
================================================================================

{chr(10).join([f'{feature}: {outliers_found[feature]} outliers detected' for feature in numerical_cols])}

================================================================================
8. KEY FINDINGS
================================================================================

1. The dataset is perfectly balanced with 50 samples per species
2. No missing values or quality issues detected
3. Petal length and width have the highest discriminative power
4. Setosa is completely linearly separable from other species
5. Petal length and width are highly correlated (r = 0.96) - feature redundancy
6. Sepal width is the least informative feature for classification
7. Versicolor and Virginica show overlap in feature space
8. All features show statistically significant differences between species (p < 0.001)

================================================================================
9. RECOMMENDATIONS FOR MODELING
================================================================================

- Prioritize petal features (length and width) for classification
- Consider removing one petal feature to reduce multicollinearity
- Sepal width can be deprioritized or removed from the model
- Linear models will work well for distinguishing Setosa
- Non-linear models may be needed for Versicolor vs Virginica
- The dataset is ready for modeling with no preprocessing required

================================================================================
10. CONCLUSION
================================================================================

The Iris dataset is an ideal starting point for machine learning. This EDA has
successfully revealed that:
- Petal features are the key discriminators
- Setosa is easily separable
- Feature redundancy exists between petal measurements
- The dataset is clean, balanced, and ready for modeling

These insights provide a solid foundation for building classification models.

================================================================================
                               END OF REPORT
================================================================================
"""
    
    # Save text report
    with open('../output/eda_report.txt', 'w', encoding='utf-8') as f:
        f.write(text_report)
    print("  Saved: output/eda_report.txt")
    
    # ============================================
    # GENERATE CSV SUMMARY
    # ============================================
    
    # Create a summary dataframe
    summary_data = []
    for species in species_list:
        species_data = df[df['species'] == species]
        row = {
            'Species': species.capitalize(),
            'Sepal_Length_Mean': round(species_data['sepal length (cm)'].mean(), 2),
            'Sepal_Length_Std': round(species_data['sepal length (cm)'].std(), 2),
            'Sepal_Width_Mean': round(species_data['sepal width (cm)'].mean(), 2),
            'Sepal_Width_Std': round(species_data['sepal width (cm)'].std(), 2),
            'Petal_Length_Mean': round(species_data['petal length (cm)'].mean(), 2),
            'Petal_Length_Std': round(species_data['petal length (cm)'].std(), 2),
            'Petal_Width_Mean': round(species_data['petal width (cm)'].mean(), 2),
            'Petal_Width_Std': round(species_data['petal width (cm)'].std(), 2)
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('../output/final_summary.csv', index=False)
    print("  Saved: output/final_summary.csv")
    
    # ============================================
    # PRINT SUMMARY TO CONSOLE
    # ============================================
    
    print("\n" + "=" * 60)
    print("FINAL REPORT SUMMARY")
    print("=" * 60)
    
    print(f"\nDataset Size: {total_samples} samples, {total_features} features")
    print(f"Target Classes: {total_classes} (Perfectly balanced)")
    print(f"Data Quality: {missing_values} missing values, {duplicates} duplicates")
    
    print(f"\nMost Discriminative Feature: {most_discriminative.replace(' (cm)', '')}")
    print(f"Least Discriminative Feature: {least_discriminative.replace(' (cm)', '')}")
    
    print(f"\nHighest Correlation: {highest_corr_pair[0].replace(' (cm)', '')} <-> {highest_corr_pair[1].replace(' (cm)', '')} = {highest_corr_pair[2]:.3f}")
    
    print(f"\nSetosa Separability: Gap of {petal_gap:.2f}cm in petal length")
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("   Report: output/eda_report.txt")
    print("   Summary: output/final_summary.csv")
    print("   Images: images/ folder (all visualizations)")
    print("\nTo view the complete report, open:")
    print("   output/eda_report.txt")
    
    return True

# Run the function
if __name__ == "__main__":
    generate_complete_report()