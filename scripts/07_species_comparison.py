"""
STEP 8: Species Comparison Analysis
File: scripts/07_species_comparison.py
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def species_comparison():
    """
    Compare features across the three species
    """
    print("=" * 60)
    print("STEP 8: SPECIES COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('../output/iris_raw_data.csv')
    
    # Create images directory if it doesn't exist
    os.makedirs('../images', exist_ok=True)
    
    numerical_cols = ['sepal length (cm)', 'sepal width (cm)', 
                      'petal length (cm)', 'petal width (cm)']
    
    species_list = ['setosa', 'versicolor', 'virginica']
    colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
    
    print("\n📊 Creating species comparison visualizations...")
    
    # ============================================
    # PLOT 1: GROUPED BAR CHART - Mean Values
    # ============================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate mean values by species
    mean_by_species = df.groupby('species')[numerical_cols].mean()
    
    # Set up bar positions
    x = np.arange(len(numerical_cols))
    width = 0.25
    multiplier = 0
    
    for species in species_list:
        offset = width * multiplier
        means = mean_by_species.loc[species].values
        bars = ax.bar(x + offset, means, width, label=species.capitalize(), 
                      color=colors[species], edgecolor='black', alpha=0.8)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        multiplier += 1
    
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Measurement (cm)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Feature Values by Species', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([col.replace(' (cm)', '') for col in numerical_cols], rotation=15)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('../images/grouped_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/grouped_bar_chart.png")
    
    # ============================================
    # PLOT 2: RADAR/SPIDER CHART
    # ============================================
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Prepare data for radar chart
    angles = np.linspace(0, 2 * np.pi, len(numerical_cols), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    for species in species_list:
        values = mean_by_species.loc[species].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=species.capitalize(), color=colors[species])
        ax.fill(angles, values, alpha=0.15, color=colors[species])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([col.replace(' (cm)', '') for col in numerical_cols], fontsize=10)
    ax.set_ylim(0, 8)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7'], fontsize=9)
    ax.set_title('Species Feature Comparison (Radar Chart)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('../images/radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/radar_chart.png")
    
    # ============================================
    # PLOT 3: HEATMAP - Mean Values by Species
    # ============================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap of mean values
    sns.heatmap(mean_by_species, annot=True, fmt='.2f', cmap='YlOrRd', 
                linewidths=0.5, linecolor='black',
                cbar_kws={'label': 'Mean Value (cm)', 'shrink': 0.8},
                annot_kws={'size': 11, 'weight': 'bold'})
    
    ax.set_title('Mean Feature Values Heatmap by Species', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Species', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../images/mean_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/mean_heatmap.png")
    
    # ============================================
    # PLOT 4: ERROR BARS (Mean ± Std Dev)
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        means = []
        stds = []
        species_labels = []
        
        for species in species_list:
            species_data = df[df['species'] == species][col]
            means.append(species_data.mean())
            stds.append(species_data.std())
            species_labels.append(species.capitalize())
        
        axes[idx].bar(species_labels, means, yerr=stds, capsize=10, 
                     color=[colors[s] for s in species_list], 
                     edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Measurement (cm)', fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            axes[idx].text(i, mean + std + 0.05, f'{mean:.2f}', 
                          ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Feature Means with Standard Deviation Error Bars', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../images/error_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/error_bars.png")
    
    # ============================================
    # STATISTICAL TESTS (ANOVA)
    # ============================================
    print("\n📊 STATISTICAL ANALYSIS")
    print("-" * 50)
    
    from scipy.stats import f_oneway
    
    print("\n🔬 ANOVA Test Results (Comparing all 3 species):")
    print("-" * 40)
    
    anova_results = []
    
    for col in numerical_cols:
        # Group data by species
        setosa_data = df[df['species'] == 'setosa'][col]
        versicolor_data = df[df['species'] == 'versicolor'][col]
        virginica_data = df[df['species'] == 'virginica'][col]
        
        # Perform ANOVA
        f_stat, p_value = f_oneway(setosa_data, versicolor_data, virginica_data)
        
        anova_results.append({
            'feature': col,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
        
        significance = "✅ SIGNIFICANT" if p_value < 0.05 else "❌ NOT SIGNIFICANT"
        print(f"\n{col}:")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Result: {significance}")
    
    # ============================================
    # POST-HOC ANALYSIS (Which pairs are different?)
    # ============================================
    print("\n\n🔬 Post-hoc Analysis (T-tests between species pairs):")
    print("-" * 50)
    
    from scipy.stats import ttest_ind
    
    posthoc_results = []
    
    for col in numerical_cols:
        print(f"\n{col}:")
        print("-" * 30)
        
        # Setosa vs Versicolor
        setosa_data = df[df['species'] == 'setosa'][col]
        versicolor_data = df[df['species'] == 'versicolor'][col]
        t_stat, p_val = ttest_ind(setosa_data, versicolor_data)
        sig = "✅" if p_val < 0.05 else "❌"
        print(f"  {sig} Setosa vs Versicolor: t={t_stat:.4f}, p={p_val:.6f}")
        
        # Setosa vs Virginica
        virginica_data = df[df['species'] == 'virginica'][col]
        t_stat, p_val = ttest_ind(setosa_data, virginica_data)
        sig = "✅" if p_val < 0.05 else "❌"
        print(f"  {sig} Setosa vs Virginica: t={t_stat:.4f}, p={p_val:.6f}")
        
        # Versicolor vs Virginica
        t_stat, p_val = ttest_ind(versicolor_data, virginica_data)
        sig = "✅" if p_val < 0.05 else "❌"
        print(f"  {sig} Versicolor vs Virginica: t={t_stat:.4f}, p={p_val:.6f}")
    
    # ============================================
    # SPECIES DISCRIMINATION POWER
    # ============================================
    print("\n\n🎯 SPECIES DISCRIMINATION POWER")
    print("-" * 50)
    
    # Calculate which features best separate species
    discrimination_scores = {}
    
    for col in numerical_cols:
        # Calculate ratio of between-group variance to within-group variance
        overall_mean = df[col].mean()
        species_means = df.groupby('species')[col].mean()
        species_counts = df.groupby('species')[col].count()
        
        # Between-group variance
        between_var = sum(species_counts * (species_means - overall_mean)**2) / (len(species_list) - 1)
        
        # Within-group variance (pooled)
        within_var = 0
        for species in species_list:
            species_data = df[df['species'] == species][col]
            within_var += sum((species_data - species_means[species])**2)
        within_var = within_var / (len(df) - len(species_list))
        
        # F-ratio (higher = better discrimination)
        f_ratio = between_var / within_var
        discrimination_scores[col] = f_ratio
    
    # Sort by discrimination power
    sorted_features = sorted(discrimination_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFeatures ranked by discrimination power (higher = better):")
    for i, (feature, score) in enumerate(sorted_features, 1):
        feature_name = feature.replace(' (cm)', '')
        stars = "⭐" * min(5, int(score / 100) + 1)
        print(f"  {i}. {feature_name}: {score:.2f} {stars}")
    
    # ============================================
    # SUMMARY TABLE
    # ============================================
    print("\n\n📋 SPECIES COMPARISON SUMMARY")
    print("-" * 60)
    
    summary_data = []
    for species in species_list:
        species_data = df[df['species'] == species]
        summary_data.append({
            'Species': species.capitalize(),
            'Sepal Length (mean ± std)': f"{species_data['sepal length (cm)'].mean():.2f} ± {species_data['sepal length (cm)'].std():.2f}",
            'Sepal Width (mean ± std)': f"{species_data['sepal width (cm)'].mean():.2f} ± {species_data['sepal width (cm)'].std():.2f}",
            'Petal Length (mean ± std)': f"{species_data['petal length (cm)'].mean():.2f} ± {species_data['petal length (cm)'].std():.2f}",
            'Petal Width (mean ± std)': f"{species_data['petal width (cm)'].mean():.2f} ± {species_data['petal width (cm)'].std():.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_df.to_csv('../output/species_comparison_summary.csv', index=False)
    print(f"\n💾 Saved: output/species_comparison_summary.csv")
    
    # Save ANOVA results
    anova_df = pd.DataFrame(anova_results)
    anova_df.to_csv('../output/anova_results.csv', index=False)
    print(f"💾 Saved: output/anova_results.csv")
    
    # ============================================
    # KEY FINDINGS
    # ============================================
    print("\n\n💡 KEY SPECIES COMPARISON FINDINGS")
    print("-" * 60)
    
    # Find largest difference
    petal_len_range = mean_by_species['petal length (cm)'].max() - mean_by_species['petal length (cm)'].min()
    petal_width_range = mean_by_species['petal width (cm)'].max() - mean_by_species['petal width (cm)'].min()
    
    print(f"\n1. Most Discriminative Feature: {sorted_features[0][0].replace(' (cm)', '')}")
    print(f"   → Difference between species: {mean_by_species[sorted_features[0][0]].max() - mean_by_species[sorted_features[0][0]].min():.2f} cm")
    
    print(f"\n2. Least Discriminative Feature: {sorted_features[-1][0].replace(' (cm)', '')}")
    print(f"   → Species overlap significantly")
    
    print(f"\n3. Setosa is completely separable from others using:")
    print(f"   → Petal length (max 1.9cm vs min 3.0cm for others)")
    print(f"   → Petal width (max 0.6cm vs min 1.0cm for others)")
    
    print(f"\n4. Versicolor and Virginica overlap on:")
    print(f"   → Sepal measurements")
    print(f"   → Some petal measurements (versicolor max 5.1cm, virginica min 4.5cm)")
    
    print("\n" + "=" * 60)
    print("✅ Species comparison analysis completed!")
    print("=" * 60)
    
    return mean_by_species, anova_results

# Run the function
if __name__ == "__main__":
    mean_by_species, anova_results = species_comparison()