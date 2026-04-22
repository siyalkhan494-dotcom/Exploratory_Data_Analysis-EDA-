"""
STEP 7: Correlation Analysis & Heatmap
File: scripts/06_correlation.py
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

def correlation_analysis():
    """
    Perform correlation analysis and create heatmaps
    """
    print("=" * 60)
    print("STEP 7: CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('../output/iris_raw_data.csv')
    
    # Create images directory if it doesn't exist
    os.makedirs('../images', exist_ok=True)
    
    # Get numerical columns only
    numerical_cols = ['sepal length (cm)', 'sepal width (cm)', 
                      'petal length (cm)', 'petal width (cm)']
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    print("\n📊 CORRELATION MATRIX")
    print("-" * 50)
    print(corr_matrix.round(4))
    
    # ============================================
    # PLOT 1: BASIC HEATMAP
    # ============================================
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, 
                annot=True,           # Show correlation values
                fmt='.3f',            # 3 decimal places
                cmap='coolwarm',      # Color scheme (red=positive, blue=negative)
                center=0,             # Center colormap at 0
                square=True,          # Square cells
                linewidths=0.5,       # Lines between cells
                linecolor='white',    # Line color
                cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                annot_kws={'size': 12, 'weight': 'bold'})
    
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    plt.savefig('../images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n  ✅ Saved: images/correlation_heatmap.png")
    
    # ============================================
    # PLOT 2: MASKED HEATMAP (Upper triangle only)
    # ============================================
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, 
                mask=mask,            # Hide upper triangle
                annot=True, 
                fmt='.3f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                annot_kws={'size': 12, 'weight': 'bold'})
    
    plt.title('Correlation Heatmap (Lower Triangle)', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    plt.savefig('../images/correlation_heatmap_masked.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/correlation_heatmap_masked.png")
    
    # ============================================
    # PLOT 3: CLUSTERED HEATMAP
    # ============================================
    plt.figure(figsize=(12, 10))
    
    # Create clustered heatmap (groups similar correlations together)
    sns.clustermap(corr_matrix, 
                   annot=True, 
                   fmt='.3f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   figsize=(10, 10),
                   dendrogram_ratio=0.15,
                   cbar_pos=(0.02, 0.76, 0.03, 0.2),
                   annot_kws={'size': 12})
    
    plt.savefig('../images/correlation_clustermap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: images/correlation_clustermap.png")
    
    # ============================================
    # DETAILED CORRELATION ANALYSIS
    # ============================================
    print("\n📈 DETAILED CORRELATION ANALYSIS")
    print("-" * 50)
    
    # Create a list to store correlation insights
    insights = []
    
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            col1 = numerical_cols[i]
            col2 = numerical_cols[j]
            corr_value = corr_matrix.iloc[i, j]
            
            # Determine strength
            if abs(corr_value) >= 0.8:
                strength = "VERY STRONG"
                symbol = "🔴"
            elif abs(corr_value) >= 0.6:
                strength = "STRONG"
                symbol = "🟠"
            elif abs(corr_value) >= 0.4:
                strength = "MODERATE"
                symbol = "🟡"
            elif abs(corr_value) >= 0.2:
                strength = "WEAK"
                symbol = "🟢"
            else:
                strength = "VERY WEAK"
                symbol = "⚪"
            
            direction = "positive" if corr_value > 0 else "negative"
            
            insight = {
                'feature1': col1,
                'feature2': col2,
                'correlation': corr_value,
                'strength': strength,
                'direction': direction,
                'symbol': symbol
            }
            insights.append(insight)
            
            print(f"\n{symbol} {col1} vs {col2}")
            print(f"   Correlation: {corr_value:.4f}")
            print(f"   Strength: {strength} ({direction})")
    
    # ============================================
    # SPECIES-SPECIFIC CORRELATIONS
    # ============================================
    print("\n\n📊 SPECIES-SPECIFIC CORRELATIONS")
    print("-" * 50)
    
    species_list = df['species'].unique()
    
    for species in species_list:
        species_df = df[df['species'] == species][numerical_cols]
        species_corr = species_df.corr()
        
        print(f"\n🔹 {species.upper()}:")
        print(species_corr.round(4))
        
        # Save species correlation to CSV
        species_corr.to_csv(f'../output/correlation_{species}.csv')
        print(f"   💾 Saved: output/correlation_{species}.csv")
    
    # ============================================
    # CORRELATION INTERPRETATION
    # ============================================
    print("\n\n📝 CORRELATION INTERPRETATION")
    print("-" * 50)
    
    print("\n🔴 VERY STRONG Correlations (|r| ≥ 0.8):")
    for ins in insights:
        if ins['strength'] == "VERY STRONG":
            print(f"   • {ins['feature1']} & {ins['feature2']}: {ins['correlation']:.3f}")
    
    print("\n🟠 STRONG Correlations (0.6 ≤ |r| < 0.8):")
    for ins in insights:
        if ins['strength'] == "STRONG":
            print(f"   • {ins['feature1']} & {ins['feature2']}: {ins['correlation']:.3f}")
    
    print("\n🟡 MODERATE Correlations (0.4 ≤ |r| < 0.6):")
    for ins in insights:
        if ins['strength'] == "MODERATE":
            print(f"   • {ins['feature1']} & {ins['feature2']}: {ins['correlation']:.3f}")
    
    # ============================================
    # KEY FINDINGS
    # ============================================
    print("\n\n💡 KEY CORRELATION FINDINGS")
    print("-" * 50)
    
    # Find highest correlation
    highest_corr = max(insights, key=lambda x: abs(x['correlation']))
    print(f"\n1. Highest Correlation: {highest_corr['feature1']} & {highest_corr['feature2']}")
    print(f"   → {highest_corr['correlation']:.4f}")
    print(f"   → This indicates strong linear relationship")
    
    # Find lowest correlation
    lowest_corr = min(insights, key=lambda x: abs(x['correlation']))
    print(f"\n2. Lowest Correlation: {lowest_corr['feature1']} & {lowest_corr['feature2']}")
    print(f"   → {lowest_corr['correlation']:.4f}")
    print(f"   → These features are nearly independent")
    
    # Check for multicollinearity
    print(f"\n3. Multicollinearity Check:")
    high_corr_pairs = [ins for ins in insights if abs(ins['correlation']) > 0.8]
    if high_corr_pairs:
        print(f"   ⚠️ Found {len(high_corr_pairs)} pairs with very high correlation")
        print(f"   → This suggests feature redundancy")
        print(f"   → Consider using only one from each highly correlated pair")
    
    # ============================================
    # EXPORT CORRELATION MATRIX
    # ============================================
    corr_matrix.to_csv('../output/correlation_matrix.csv')
    print(f"\n💾 Saved: output/correlation_matrix.csv")
    
    # Create correlation summary report
    with open('../output/correlation_report.txt', 'w') as f:
        f.write("CORRELATION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write("Correlation Matrix:\n")
        f.write(corr_matrix.round(4).to_string())
        f.write("\n\n" + "=" * 50 + "\n")
        f.write("KEY INSIGHTS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"1. Highest correlation: {highest_corr['feature1']} vs {highest_corr['feature2']} = {highest_corr['correlation']:.4f}\n")
        f.write(f"2. Lowest correlation: {lowest_corr['feature1']} vs {lowest_corr['feature2']} = {lowest_corr['correlation']:.4f}\n")
        f.write(f"3. Petal features are highly correlated (redundant)\n")
        f.write(f"4. Sepal width has weak/near-zero correlation with other features\n")
    
    print(f"💾 Saved: output/correlation_report.txt")
    
    print("\n" + "=" * 60)
    print("✅ Correlation analysis completed!")
    print("=" * 60)
    
    return corr_matrix

# Run the function
if __name__ == "__main__":
    corr_matrix = correlation_analysis()
    