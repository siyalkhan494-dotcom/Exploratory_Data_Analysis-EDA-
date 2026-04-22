"""
STEP 9: Final Visualization Compilation
File: scripts/08_final_visualization.py
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_final_visualizations():
    """
    Create the final compilation of visualizations for the report
    """
    print("=" * 60)
    print("STEP 9: FINAL VISUALIZATION COMPILATION")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('../output/iris_raw_data.csv')
    
    # Create images directory if it doesn't exist
    os.makedirs('../images', exist_ok=True)
    
    numerical_cols = ['sepal length (cm)', 'sepal width (cm)', 
                      'petal length (cm)', 'petal width (cm)']
    
    species_list = ['setosa', 'versicolor', 'virginica']
    colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
    
    print("\n📊 Creating final compilation visualizations...")
    
    # ============================================
    # FIGURE 1: COMPLETE EDA DASHBOARD (4-in-1 Plot)
    # This matches the report's figure with 4 key plots
    # ============================================
    fig = plt.figure(figsize=(16, 12))
    
    # Top-Left: Species Distribution (Bar Chart)
    ax1 = plt.subplot(2, 2, 1)
    species_counts = df['species'].value_counts()
    bars = ax1.bar(species_counts.index, species_counts.values, 
                   color=[colors[s] for s in species_counts.index],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_title('Species Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Species', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_ylim(0, 60)
    
    # Add value labels on bars
    for bar, count in zip(bars, species_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add grid
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Top-Right: Sepal Length vs Sepal Width (Scatter Plot)
    ax2 = plt.subplot(2, 2, 2)
    for species in species_list:
        species_data = df[df['species'] == species]
        ax2.scatter(species_data['sepal length (cm)'], 
                   species_data['sepal width (cm)'],
                   label=species.capitalize(), 
                   c=colors[species],
                   s=80, 
                   alpha=0.7, 
                   edgecolors='black',
                   linewidth=1)
    
    ax2.set_title('Sepal Length vs Sepal Width', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sepal Length (cm)', fontsize=11)
    ax2.set_ylabel('Sepal Width (cm)', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Bottom-Left: Feature Distributions (Boxplot)
    ax3 = plt.subplot(2, 2, 3)
    
    # Prepare data for boxplot
    boxplot_data = [df[col] for col in numerical_cols]
    bp = ax3.boxplot(boxplot_data, labels=[col.replace(' (cm)', '') for col in numerical_cols],
                     patch_artist=True, showmeans=True, meanline=True)
    
    # Color the boxes
    box_colors = ['#FFB3B3', '#B3D9FF', '#B3FFB3', '#FFCCB3']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize appearance
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    bp['means'][0].set_color('blue')
    bp['means'][0].set_linewidth(1.5)
    
    ax3.set_title('Feature Distributions', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Measurement (cm)', fontsize=11)
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Bottom-Right: Correlation Heatmap
    ax4 = plt.subplot(2, 2, 4)
    corr_matrix = df[numerical_cols].corr()
    
    heatmap = sns.heatmap(corr_matrix, 
                          annot=True, 
                          fmt='.2f',
                          cmap='coolwarm',
                          center=0,
                          square=True,
                          linewidths=0.5,
                          cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                          annot_kws={'size': 11, 'weight': 'bold'},
                          ax=ax4)
    
    ax4.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    ax4.set_xticklabels([col.replace(' (cm)', '') for col in numerical_cols], rotation=45, ha='right')
    ax4.set_yticklabels([col.replace(' (cm)', '') for col in numerical_cols], rotation=0)
    
    plt.suptitle('Iris Dataset: Exploratory Data Analysis Dashboard', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../images/eda_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/eda_dashboard.png")
    
    # ============================================
    # FIGURE 2: PETAL ANALYSIS (Most Important Features)
    # ============================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Petal Length vs Petal Width
    for species in species_list:
        species_data = df[df['species'] == species]
        axes[0].scatter(species_data['petal length (cm)'], 
                       species_data['petal width (cm)'],
                       label=species.capitalize(), 
                       c=colors[species],
                       s=100, 
                       alpha=0.7, 
                       edgecolors='black',
                       linewidth=1.5)
    
    axes[0].set_xlabel('Petal Length (cm)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Petal Width (cm)', fontsize=12, fontweight='bold')
    axes[0].set_title('Petal Length vs Petal Width\n(Most Discriminative Features)', 
                      fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Add separation line showing Setosa is completely separable
    axes[0].axvline(x=2.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Separation boundary')
    axes[0].text(2.55, 1.5, 'Setosa', fontsize=10, fontweight='bold', color='#FF6B6B')
    axes[0].text(4.5, 2.0, 'Versicolor/Virginica', fontsize=10, fontweight='bold', color='#4ECDC4')
    
    # Right: Boxplot of Petal Features by Species
    petal_cols = ['petal length (cm)', 'petal width (cm)']
    petal_data = []
    species_labels = []
    
    for species in species_list:
        for col in petal_cols:
            petal_data.append(df[df['species'] == species][col].values)
            species_labels.append(f"{species}\n{col.replace(' (cm)', '')}")
    
    bp2 = axes[1].boxplot(petal_data, labels=species_labels, patch_artist=True, showmeans=True)
    
    # Color the boxes
    for i, (species, color) in enumerate(colors.items()):
        bp2['boxes'][i*2].set_facecolor(color)
        bp2['boxes'][i*2+1].set_facecolor(color)
        bp2['boxes'][i*2].set_alpha(0.7)
        bp2['boxes'][i*2+1].set_alpha(0.5)
    
    axes[1].set_title('Petal Features Distribution by Species', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Measurement (cm)', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Petal Features Analysis - Key Discriminators', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('../images/petal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/petal_analysis.png")
    
    # ============================================
    # FIGURE 3: SPECIES PROFILES (Mean values with comparison)
    # ============================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate mean values
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
        
        # Add value labels
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        multiplier += 1
    
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Measurement (cm)', fontsize=12, fontweight='bold')
    ax.set_title('Species Comparison - Mean Feature Values', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([col.replace(' (cm)', '') for col in numerical_cols], fontsize=11)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('../images/species_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/species_profiles.png")
    
    # ============================================
    # FIGURE 4: DENSITY PLOTS FOR ALL FEATURES
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        for species in species_list:
            species_data = df[df['species'] == species][col]
            sns.kdeplot(data=species_data, label=species.capitalize(), 
                       fill=True, alpha=0.3, color=colors[species], ax=axes[idx])
        
        axes[idx].set_title(f'Density Distribution - {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Measurement (cm)', fontsize=10)
        axes[idx].set_ylabel('Density', fontsize=10)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Density Distributions by Species', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../images/density_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/density_distributions.png")
    
    # ============================================
    # FIGURE 5: CORRELATION MATRIX WITH VALUES
    # ============================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = df[numerical_cols].corr()
    
    # Create a mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    heatmap2 = sns.heatmap(corr_matrix, 
                           mask=mask,
                           annot=True, 
                           fmt='.3f',
                           cmap='RdYlBu_r',
                           center=0,
                           square=True,
                           linewidths=1,
                           linecolor='white',
                           cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient', 'pad': 0.02},
                           annot_kws={'size': 13, 'weight': 'bold'},
                           ax=ax)
    
    ax.set_title('Feature Correlation Matrix\n(Values show correlation strength)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels([col.replace(' (cm)', '') for col in numerical_cols], fontsize=11)
    ax.set_yticklabels([col.replace(' (cm)', '') for col in numerical_cols], fontsize=11)
    
    plt.tight_layout()
    plt.savefig('../images/correlation_matrix_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/correlation_matrix_final.png")
    
    # ============================================
    # FIGURE 6: PAIRPLOT (All feature combinations)
    # ============================================
    print("  ⏳ Creating final pairplot (this may take a moment)...")
    
    pairplot = sns.pairplot(df, 
                            hue='species', 
                            vars=numerical_cols,
                            palette=colors,
                            diag_kind='kde',
                            plot_kws={'alpha': 0.6, 's': 60, 'edgecolor': 'black', 'linewidth': 1},
                            diag_kws={'fill': True, 'alpha': 0.5})
    
    pairplot.fig.suptitle('Complete Feature Pairplot Analysis', y=1.02, fontsize=16, fontweight='bold')
    pairplot.savefig('../images/complete_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/complete_pairplot.png")
    
    # ============================================
    # SUMMARY OF ALL VISUALIZATIONS
    # ============================================
    print("\n📊 FINAL VISUALIZATION SUMMARY")
    print("-" * 50)
    
    visualizations = [
        "eda_dashboard.png - Main 4-in-1 dashboard (matches report)",
        "petal_analysis.png - Focus on most discriminative features",
        "species_profiles.png - Mean values comparison bar chart",
        "density_distributions.png - KDE plots for all features",
        "correlation_matrix_final.png - Clean correlation heatmap",
        "complete_pairplot.png - All feature combinations"
    ]
    
    print("\nCreated visualizations:")
    for viz in visualizations:
        print(f"  📊 {viz}")
    
    print("\n" + "=" * 60)
    print("✅ Final visualizations completed!")
    print("📁 All images saved in: images/")
    print("=" * 60)
    
    return True

# Run the function
if __name__ == "__main__":
    create_final_visualizations()