"""
RUN ALL ML SCRIPTS IN SEQUENCE
File: ml_scripts/run_all_ml.py

Ye script sabhi ML steps ko ek saath execute karega
"""

import subprocess
import sys
import os

def run_all_ml():
    """
    Sabhi ML scripts ko order mein run karna
    """
    print("=" * 60)
    print("RUNNING COMPLETE ML PIPELINE")
    print("=" * 60)
    
    scripts = [
        '01_data_preparation.py',
        '02_train_test_split.py',
        '03_feature_scaling.py',
        '04_train_all_models.py',
        '05_evaluate_models.py',
        '06_hyperparameter_tuning.py',
        '07_final_model.py'
    ]
    
    for script in scripts:
        print(f"\n\n{'=' * 60}")
        print(f"Running: {script}")
        print('=' * 60)
        
        result = subprocess.run([sys.executable, script], 
                               cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode != 0:
            print(f"\n❌ Error in {script}. Stopping execution.")
            break
        else:
            print(f"\n✅ Completed: {script}")
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE ML PIPELINE FINISHED!")
    print("=" * 60)
    print("\n📁 Output Files Generated:")
    print("   - ml_output/X_features.csv")
    print("   - ml_output/X_train_scaled.csv")
    print("   - ml_output/X_test_scaled.csv")
    print("   - ml_output/models/ (all trained models)")
    print("   - ml_output/evaluation/model_comparison.csv")
    print("   - ml_output/plots/model_comparison.png")
    print("   - ml_output/plots/confusion_matrices.png")
    print("   - ml_output/models/final_model.pkl")
    print("   - ml_output/predict_function.py")

if __name__ == "__main__":
    run_all_ml()