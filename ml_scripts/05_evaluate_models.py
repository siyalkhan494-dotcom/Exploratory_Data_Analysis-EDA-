"""
STEP 5: MODEL EVALUATION & COMPARISON
File: ml_scripts/05_evaluate_models.py

Is script mein:
1. Sabhi trained models ko test data pe evaluate karenge
2. Metrics calculate karenge (Accuracy, Precision, Recall, F1-Score)
3. Confusion Matrix banaenge
4. Sab models ki comparison karenge
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def evaluate_models():
    """
    Sabhi models ko evaluate karna
    """
    print("=" * 60)
    print("STEP 5: MODEL EVALUATION & COMPARISON")
    print("=" * 60)
    
    # ============================================
    # Create all necessary folders
    # ============================================
    print("\n📁 Creating output folders...")
    os.makedirs('../ml_output', exist_ok=True)
    os.makedirs('../ml_output/models', exist_ok=True)
    os.makedirs('../ml_output/evaluation', exist_ok=True)
    os.makedirs('../ml_output/plots', exist_ok=True)
    print("   ✅ All folders created/verified")
    
    # 1. Load test data
    print("\n1. Loading test data...")
    X_test = pd.read_csv('../ml_output/X_test_scaled.csv')
    y_test = pd.read_csv('../ml_output/y_test.csv')
    y_test = y_test.values.ravel()
    
    print(f"   Test features: {X_test.shape}")
    print(f"   Test target: {y_test.shape}")
    
    # 2. Load all trained models
    print("\n2. Loading trained models...")
    all_models_path = '../ml_output/models/all_models.pkl'
    
    # Check if models file exists
    if not os.path.exists(all_models_path):
        print(f"   ❌ Error: {all_models_path} not found!")
        print("   Please run 04_train_all_models.py first")
        return None, None
    
    all_models = joblib.load(all_models_path)
    print(f"   Loaded {len(all_models)} models")
    
    # 3. Evaluate each model
    print("\n3. Evaluating models on test data...")
    print("-" * 70)
    
    results = []
    confusion_matrices = {}
    classification_reports = {}
    
    for model_name, model in all_models.items():
        print(f"\n   📊 {model_name}")
        print("   " + "-" * 50)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[model_name] = cm
        
        # Classification Report
        report = classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolor', 'Virginica'])
        classification_reports[model_name] = report
        
        # Print metrics
        print(f"      ✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"      ✅ Precision: {precision:.4f}")
        print(f"      ✅ Recall:    {recall:.4f}")
        print(f"      ✅ F1-Score:  {f1:.4f}")
        
        # Print confusion matrix
        print(f"\n      Confusion Matrix:")
        print(f"      {cm}")
    
    # 4. Create results dataframe
    print("\n4. Creating comparison table...")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    print("\n   📊 MODEL COMPARISON (Sorted by Accuracy):")
    print("   " + "-" * 70)
    print(results_df.to_string(index=False))
    
    # 5. Save results
    print("\n5. Saving evaluation results...")
    results_df.to_csv('../ml_output/evaluation/model_comparison.csv', index=False)
    print(f"   ✅ Saved: ml_output/evaluation/model_comparison.csv")
    
    # Save classification reports
    with open('../ml_output/evaluation/classification_reports.txt', 'w') as f:
        f.write("CLASSIFICATION REPORTS\n")
        f.write("=" * 50 + "\n\n")
        for model_name, report in classification_reports.items():
            f.write(f"\n{model_name}\n")
            f.write("-" * 40 + "\n")
            f.write(report)
            f.write("\n" + "=" * 50 + "\n")
    
    print(f"   ✅ Saved: ml_output/evaluation/classification_reports.txt")
    
    # 6. Plot comparison bar chart
    print("\n6. Creating comparison visualizations...")
    
    try:
        # Bar chart of accuracies
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#4CAF50' if i == 0 else '#2196F3' for i in range(len(results_df))]
        bars = ax.bar(results_df['Model'], results_df['Accuracy'] * 100, color=colors, edgecolor='black')
        
        # Add value labels on bars
        for bar, acc in zip(bars, results_df['Accuracy'] * 100):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison - Test Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('../ml_output/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: ml_output/plots/model_comparison.png")
    except Exception as e:
        print(f"   ⚠️ Could not save model_comparison.png: {e}")
    
    # 7. Plot confusion matrices
    print("\n7. Creating confusion matrix plots...")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
            if idx < 5:  # Only first 5 models
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                           yticklabels=['Setosa', 'Versicolor', 'Virginica'],
                           ax=axes[idx])
                axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('Predicted', fontsize=10)
                axes[idx].set_ylabel('Actual', fontsize=10)
        
        # Hide extra subplot
        axes[5].axis('off')
        
        plt.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../ml_output/plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: ml_output/plots/confusion_matrices.png")
    except Exception as e:
        print(f"   ⚠️ Could not save confusion_matrices.png: {e}")
    
    # 8. Find best model
    print("\n8. Identifying best model...")
    best_model_name = results_df.iloc[0]['Model']
    best_accuracy = results_df.iloc[0]['Accuracy']
    
    print(f"\n   🏆 BEST MODEL: {best_model_name}")
    print(f"      Accuracy: {best_accuracy*100:.2f}%")
    
    # Save best model info
    with open('../ml_output/evaluation/best_model_info.txt', 'w') as f:
        f.write("BEST MODEL INFORMATION\n")
        f.write("=" * 40 + "\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Test Accuracy: {best_accuracy*100:.2f}%\n")
        f.write(f"Total models evaluated: {len(all_models)}\n")
    
    # 9. Summary
    print("\n" + "=" * 60)
    print("✅ MODEL EVALUATION COMPLETED!")
    print("=" * 60)
    print(f"\n📊 Evaluation Summary:")
    print(f"   - Total models evaluated: {len(all_models)}")
    print(f"   - Best model: {best_model_name} ({best_accuracy*100:.2f}%)")
    print(f"   - Results saved in: ml_output/evaluation/")
    print(f"   - Plots saved in: ml_output/plots/")
    
    return results_df, best_model_name

# Run the function
if __name__ == "__main__":
    results, best_model = evaluate_models()