"""
STEP 6: Bivariate Analysis - Feature Relationships
File: scripts/05_bivariate_analysis.py
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

def create_bivariate_plots():
    """
    Create scatter plots to show relationships between features
    """
    print("=" * 60)
    print("STEP 6: BIVARIATE ANALYSIS")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('../output/iris_raw_data.csv')
    
    # Create images directory if it doesn't exist
    os.makedirs('../images', exist_ok=True)
    
    # Define colors for species
    species_colors = {
        'setosa': '#FF6B6B',      # Red
        'versicolor': '#4ECDC4',   # Teal
        'virginica': '#45B7D1'     # Blue
    }
    
    numerical_cols = ['sepal length (cm)', 'sepal width (cm)', 
                      'petal length (cm)', 'petal width (cm)']
    
    print("\n📊 Creating scatter plots...")
    
    # ============================================
    # PLOT 1: SEPAL LENGTH vs SEPAL WIDTH
    # ============================================
    plt.figure(figsize=(10, 8))
    
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        plt.scatter(species_data['sepal length (cm)'], 
                   species_data['sepal width (cm)'],
                   label=species, 
                   c=species_colors[species],
                   s=100, 
                   alpha=0.7, 
                   edgecolors='black',
                   linewidth=1)
    
    plt.xlabel('Sepal Length (cm)', fontsize=12, fontweight='bold')
    plt.ylabel('Sepal Width (cm)', fontsize=12, fontweight='bold')
    plt.title('Sepal Length vs Sepal Width by Species', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add mean lines
    plt.axhline(y=df['sepal width (cm)'].mean(), color='gray', linestyle='--', alpha=0.5, label='Mean Sepal Width')
    plt.axvline(x=df['sepal length (cm)'].mean(), color='brown', linestyle='--', alpha=0.5, label='Mean Sepal Length')
    
    plt.tight_layout()
    plt.savefig('../images/scatter_sepal.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/scatter_sepal.png")
    
    # ============================================
    # PLOT 2: PETAL LENGTH vs PETAL WIDTH
    # ============================================
    plt.figure(figsize=(10, 8))
    
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        plt.scatter(species_data['petal length (cm)'], 
                   species_data['petal width (cm)'],
                   label=species, 
                   c=species_colors[species],
                   s=100, 
                   alpha=0.7, 
                   edgecolors='black',
                   linewidth=1)
    
    plt.xlabel('Petal Length (cm)', fontsize=12, fontweight='bold')
    plt.ylabel('Petal Width (cm)', fontsize=12, fontweight='bold')
    plt.title('Petal Length vs Petal Width by Species', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add mean lines
    plt.axhline(y=df['petal width (cm)'].mean(), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=df['petal length (cm)'].mean(), color='brown', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('../images/scatter_petal.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/scatter_petal.png")
    
    # ============================================
    # PLOT 3: PAIRPLOT (All combinations)
    # ============================================
    print("  ⏳ Creating pairplot (this may take a moment)...")
    
    pairplot = sns.pairplot(df, hue='species', 
                            vars=numerical_cols,
                            palette=species_colors,
                            diag_kind='hist',
                            plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'black'},
                            diag_kws={'alpha': 0.7, 'edgecolor': 'black'})
    
    pairplot.fig.suptitle('Pairplot of All Iris Features', y=1.02, fontsize=16, fontweight='bold')
    pairplot.savefig('../images/pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/pairplot.png")
    
    # ============================================
    # PLOT 4: SEPAL LENGTH vs PETAL LENGTH
    # ============================================
    plt.figure(figsize=(10, 8))
    
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        plt.scatter(species_data['sepal length (cm)'], 
                   species_data['petal length (cm)'],
                   label=species, 
                   c=species_colors[species],
                   s=100, 
                   alpha=0.7, 
                   edgecolors='black',
                   linewidth=1)
    
    plt.xlabel('Sepal Length (cm)', fontsize=12, fontweight='bold')
    plt.ylabel('Petal Length (cm)', fontsize=12, fontweight='bold')
    plt.title('Sepal Length vs Petal Length by Species', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/scatter_sepal_vs_petal.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/scatter_sepal_vs_petal.png")
    
    # ============================================
    # PLOT 5: SEPAL WIDTH vs PETAL WIDTH
    # ============================================
    plt.figure(figsize=(10, 8))
    
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        plt.scatter(species_data['sepal width (cm)'], 
                   species_data['petal width (cm)'],
                   label=species, 
                   c=species_colors[species],
                   s=100, 
                   alpha=0.7, 
                   edgecolors='black',
                   linewidth=1)
    
    plt.xlabel('Sepal Width (cm)', fontsize=12, fontweight='bold')
    plt.ylabel('Petal Width (cm)', fontsize=12, fontweight='bold')
    plt.title('Sepal Width vs Petal Width by Species', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/scatter_sepwidth_vs_petwidth.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/scatter_sepwidth_vs_petwidth.png")
    
    # ============================================
    # PLOT 6: JOINT PLOT (Most correlated features)
    # ============================================
    # Create joint plot for petal length vs petal width (highest correlation)
    joint = sns.jointplot(data=df, x='petal length (cm)', y='petal width (cm)',
                          hue='species', palette=species_colors,
                          height=8, alpha=0.6, edgecolor='black')
    joint.fig.suptitle('Joint Distribution: Petal Length vs Petal Width', y=1.02, fontsize=14, fontweight='bold')
    joint.savefig('../images/jointplot_petal.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/jointplot_petal.png")
    
    # ============================================
    # ANALYSIS & INSIGHTS
    # ============================================
    print("\n📊 BIVARIATE ANALYSIS INSIGHTS")
    print("-" * 50)
    
    # Calculate correlations
    print("\n📈 Correlation Analysis:")
    print("-" * 35)
    
    correlations = {}
    for col1 in numerical_cols:
        for col2 in numerical_cols:
            if col1 != col2 and (col2, col1) not in correlations:
                corr = df[col1].corr(df[col2])
                correlations[(col1, col2)] = corr
                
                # Highlight strong correlations
                strength = ""
                if abs(corr) >= 0.8:
                    strength = "🔴 VERY STRONG"
                elif abs(corr) >= 0.6:
                    strength = "🟠 STRONG"
                elif abs(corr) >= 0.4:
                    strength = "🟡 MODERATE"
                elif abs(corr) >= 0.2:
                    strength = "🟢 WEAK"
                else:
                    strength = "⚪ VERY WEAK"
                
                print(f"  {col1[:12]} vs {col2[:12]}: {corr:.3f} - {strength}")
    
    # Species separability analysis
    print("\n🔍 Species Separability Analysis:")
    print("-" * 35)
    
    # Check which features best separate species
    setosa = df[df['species'] == 'setosa']
    versicolor = df[df['species'] == 'versicolor']
    virginica = df[df['species'] == 'virginica']
    
    print("\n  Petal Length separation:")
    print(f"    Setosa max: {setosa['petal length (cm)'].max():.2f}")
    print(f"    Versicolor min: {versicolor['petal length (cm)'].min():.2f}")
    print(f"    Gap: {versicolor['petal length (cm)'].min() - setosa['petal length (cm)'].max():.2f} cm")
    
    print("\n  Petal Width separation:")
    print(f"    Setosa max: {setosa['petal width (cm)'].max():.2f}")
    print(f"    Versicolor min: {versicolor['petal width (cm)'].min():.2f}")
    print(f"    Gap: {versicolor['petal width (cm)'].min() - setosa['petal width (cm)'].max():.2f} cm")
    
    # Check overlap between versicolor and virginica
    print("\n  Versicolor vs Virginica Overlap:")
    print(f"    Petal Length - Versicolor max: {versicolor['petal length (cm)'].max():.2f}")
    print(f"    Petal Length - Virginica min: {virginica['petal length (cm)'].min():.2f}")
    print(f"    Overlap exists: {'YES' if virginica['petal length (cm)'].min() <= versicolor['petal length (cm)'].max() else 'NO'}")
    
    print("\n" + "=" * 60)
    print("💾 All visualizations saved to images/ folder")
    print("=" * 60)
    
    return True

# Run the function
if __name__ == "__main__":
    create_bivariate_plots()