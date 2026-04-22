"""
STEP 5: Univariate Analysis - Single Feature Visualizations
File: scripts/04_univariate_analysis.py
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def create_univariate_plots():
    """
    Create histograms and boxplots for each feature
    """
    print("=" * 60)
    print("STEP 5: UNIVARIATE ANALYSIS")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('../output/iris_raw_data.csv')
    
    # Create images directory if it doesn't exist
    os.makedirs('../images', exist_ok=True)
    
    numerical_cols = ['sepal length (cm)', 'sepal width (cm)', 
                      'petal length (cm)', 'petal width (cm)']
    
    print("\n📊 Creating visualizations...")
    
    # ============================================
    # PLOT 1: HISTOGRAMS FOR EACH FEATURE
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(df[col].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df[col].mean():.2f}')
        axes[idx].axvline(df[col].median(), color='green', linestyle='dashed', linewidth=2, label=f'Median: {df[col].median():.2f}')
        axes[idx].legend()
    
    plt.suptitle('Histograms of Iris Features', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../images/histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/histograms.png")
    
    # ============================================
    # PLOT 2: BOXPLOTS FOR EACH FEATURE
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].boxplot(df[col], patch_artist=True, 
                         boxprops=dict(facecolor='lightblue', color='blue'),
                         whiskerprops=dict(color='blue'),
                         capprops=dict(color='blue'),
                         medianprops=dict(color='red', linewidth=2))
        axes[idx].set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(col)
        axes[idx].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {df[col].mean():.2f}\nMedian: {df[col].median():.2f}'
        axes[idx].text(0.75, 0.95, stats_text, transform=axes[idx].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Boxplots of Iris Features', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../images/boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/boxplots.png")
    
    # ============================================
    # PLOT 3: BOXPLOTS BY SPECIES (Side by side)
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        # Create dataframe for this feature
        plot_data = []
        for species in df['species'].unique():
            species_data = df[df['species'] == species][col].values
            plot_data.append(species_data)
        
        # Create boxplot
        bp = axes[idx].boxplot(plot_data, labels=df['species'].unique(), 
                               patch_artist=True,
                               boxprops=dict(facecolor='lightblue', alpha=0.7))
        
        # Color each box differently
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[idx].set_title(f'{col} by Species', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(col)
        axes[idx].set_xlabel('Species')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distribution Across Species', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../images/boxplots_by_species.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/boxplots_by_species.png")
    
    # ============================================
    # PLOT 4: DENSITY PLOTS (KDE)
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    colors = {'setosa': '#FF9999', 'versicolor': '#66B2FF', 'virginica': '#99FF99'}
    
    for idx, col in enumerate(numerical_cols):
        for species in df['species'].unique():
            species_data = df[df['species'] == species][col]
            axes[idx].hist(species_data, bins=20, alpha=0.5, label=species, 
                          density=True, color=colors[species])
            axes[idx].axvline(species_data.mean(), color=colors[species], 
                            linestyle='dashed', linewidth=1.5, alpha=0.7)
        
        axes[idx].set_title(f'Density Plot - {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Density Distribution by Species', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../images/density_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/density_plots.png")
    
    # ============================================
    # PLOT 5: VIOLIN PLOTS (Better than boxplots)
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        # Create violin plot
        data_to_plot = [df[df['species'] == species][col].values for species in df['species'].unique()]
        
        parts = axes[idx].violinplot(data_to_plot, positions=[0, 1, 2], showmeans=True, showmedians=True)
        
        # Color the violins
        colors_violin = ['#FF9999', '#66B2FF', '#99FF99']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_violin[i])
            pc.set_alpha(0.7)
        
        axes[idx].set_xticks([0, 1, 2])
        axes[idx].set_xticklabels(df['species'].unique())
        axes[idx].set_title(f'Violin Plot - {col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(col)
        axes[idx].set_xlabel('Species')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Violin Plots Showing Full Distribution', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../images/violin_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/violin_plots.png")
    
    # ============================================
    # PRINT SUMMARY STATISTICS
    # ============================================
    print("\n📊 UNIVARIATE ANALYSIS SUMMARY")
    print("-" * 50)
    
    for col in numerical_cols:
        print(f"\n{col}:")
        print(f"  - Mean: {df[col].mean():.2f} cm")
        print(f"  - Median: {df[col].median():.2f} cm")
        print(f"  - Std Dev: {df[col].std():.2f} cm")
        print(f"  - Skewness: {df[col].skew():.2f}")
        
        # Interpretation of skewness
        skew_val = df[col].skew()
        if skew_val > 0.5:
            print(f"  - Interpretation: Right-skewed (positive skew)")
        elif skew_val < -0.5:
            print(f"  - Interpretation: Left-skewed (negative skew)")
        else:
            print(f"  - Interpretation: Approximately symmetric")
    
    print("\n" + "=" * 60)
    print("💾 All visualizations saved to images/ folder")
    print("=" * 60)
    
    return True

# Run the function
if __name__ == "__main__":
    create_univariate_plots()