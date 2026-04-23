"""
STEP 6: HYPERPARAMETER TUNING
File: ml_scripts/06_hyperparameter_tuning.py

Is script mein:
1. Best model ke parameters optimize karenge
2. GridSearchCV use karenge (try different combinations)
3. Cross-validation se overfitting check karenge
4. Tuned model ko save karenge

Hyperparameters kya hain?
- Model ke "knobs" jo hum adjust kar sakte hain
- Example: KNN mein 'k' (neighbors ki sankhya)
- Example: Random Forest mein 'n_estimators' (trees ki sankhya)
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

def tune_best_model():
    """
    Best model ka hyperparameter tuning karna
    """
    print("=" * 60)
    print("STEP 6: HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading training data...")
    X_train = pd.read_csv('../ml_output/X_train_scaled.csv')
    y_train = pd.read_csv('../ml_output/y_train.csv')
    y_train = y_train.values.ravel()
    
    print(f"   Training data: {X_train.shape}")
    
    # 2. Define models and parameter grids
    print("\n2. Setting up hyperparameter grids...")
    
    # Random Forest parameters
    rf_params = {
        'n_estimators': [50, 100, 200],      # Number of trees
        'max_depth': [None, 10, 20, 30],      # Tree depth
        'min_samples_split': [2, 5, 10],      # Min samples to split node
        'min_samples_leaf': [1, 2, 4]         # Min samples in leaf
    }
    
    # SVM parameters
    svm_params = {
        'C': [0.1, 1, 10, 100],              # Regularization parameter
        'kernel': ['rbf', 'linear'],          # Kernel type
        'gamma': ['scale', 'auto', 0.1, 1]    # Kernel coefficient
    }
    
    # KNN parameters
    knn_params = {
        'n_neighbors': [3, 5, 7, 9, 11],      # Number of neighbors
        'weights': ['uniform', 'distance'],    # Weight function
        'metric': ['euclidean', 'manhattan']   # Distance metric
    }
    
    print("   ✅ Parameter grids defined")
    print(f"      - Random Forest: {sum(len(v) for v in rf_params.values())} combinations")
    print(f"      - SVM: {sum(len(v) for v in svm_params.values())} combinations")
    print(f"      - KNN: {sum(len(v) for v in knn_params.values())} combinations")
    
    # 3. Hyperparameter tuning for Random Forest
    print("\n3. Tuning Random Forest...")
    print("   " + "-" * 40)
    
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(
        rf, rf_params, 
        cv=5,                    # 5-fold cross-validation
        scoring='accuracy',      # Optimize for accuracy
        n_jobs=-1,               # Use all CPU cores
        verbose=1
    )
    
    rf_grid.fit(X_train, y_train)
    
    print(f"\n   ✅ Best Random Forest parameters:")
    for param, value in rf_grid.best_params_.items():
        print(f"      {param}: {value}")
    print(f"      Best CV Accuracy: {rf_grid.best_score_:.4f} ({rf_grid.best_score_*100:.2f}%)")
    
    # 4. Hyperparameter tuning for SVM
    print("\n4. Tuning SVM...")
    print("   " + "-" * 40)
    
    svm = SVC(random_state=42)
    svm_grid = GridSearchCV(
        svm, svm_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    svm_grid.fit(X_train, y_train)
    
    print(f"\n   ✅ Best SVM parameters:")
    for param, value in svm_grid.best_params_.items():
        print(f"      {param}: {value}")
    print(f"      Best CV Accuracy: {svm_grid.best_score_:.4f} ({svm_grid.best_score_*100:.2f}%)")
    
    # 5. Hyperparameter tuning for KNN
    print("\n5. Tuning KNN...")
    print("   " + "-" * 40)
    
    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(
        knn, knn_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    knn_grid.fit(X_train, y_train)
    
    print(f"\n   ✅ Best KNN parameters:")
    for param, value in knn_grid.best_params_.items():
        print(f"      {param}: {value}")
    print(f"      Best CV Accuracy: {knn_grid.best_score_:.4f} ({knn_grid.best_score_*100:.2f}%)")
    
    # 6. Compare tuned models
    print("\n6. Comparing tuned models...")
    print("   " + "-" * 50)
    
    tuned_results = {
        'Random Forest': rf_grid.best_score_,
        'SVM': svm_grid.best_score_,
        'KNN': knn_grid.best_score_
    }
    
    best_tuned_model = max(tuned_results, key=tuned_results.get)
    
    print(f"\n   Tuned Model Accuracies:")
    for model, acc in tuned_results.items():
        print(f"      {model}: {acc*100:.2f}%")
    
    print(f"\n   🏆 Best Tuned Model: {best_tuned_model} ({tuned_results[best_tuned_model]*100:.2f}%)")
    
    # 7. Save tuned models
    print("\n7. Saving tuned models...")
    
    joblib.dump(rf_grid.best_estimator_, '../ml_output/models/tuned_random_forest.pkl')
    joblib.dump(svm_grid.best_estimator_, '../ml_output/models/tuned_svm.pkl')
    joblib.dump(knn_grid.best_estimator_, '../ml_output/models/tuned_knn.pkl')
    
    print(f"   ✅ Saved: ml_output/models/tuned_random_forest.pkl")
    print(f"   ✅ Saved: ml_output/models/tuned_svm.pkl")
    print(f"   ✅ Saved: ml_output/models/tuned_knn.pkl")
    
    # 8. Cross-validation on best model
    print("\n8. Cross-validation on best model...")
    
    if best_tuned_model == 'Random Forest':
        best_model = rf_grid.best_estimator_
    elif best_tuned_model == 'SVM':
        best_model = svm_grid.best_estimator_
    else:
        best_model = knn_grid.best_estimator_
    
    # Perform 10-fold cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='accuracy')
    
    print(f"\n   10-Fold Cross-Validation Results:")
    print(f"      Mean Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    print(f"      Std Deviation: {cv_scores.std():.4f}")
    print(f"      Individual folds: {cv_scores}")
    
    # 9. Save tuning results
    tuning_summary = {
        'best_model': best_tuned_model,
        'best_accuracy': tuned_results[best_tuned_model],
        'rf_best_params': rf_grid.best_params_,
        'rf_best_score': rf_grid.best_score_,
        'svm_best_params': svm_grid.best_params_,
        'svm_best_score': svm_grid.best_score_,
        'knn_best_params': knn_grid.best_params_,
        'knn_best_score': knn_grid.best_score_,
        'cv_mean_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    with open('../ml_output/evaluation/tuning_results.txt', 'w') as f:
        f.write("HYPERPARAMETER TUNING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        for key, value in tuning_summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n   ✅ Saved: ml_output/evaluation/tuning_results.txt")
    
    # 10. Summary
    print("\n" + "=" * 60)
    print("✅ HYPERPARAMETER TUNING COMPLETED!")
    print("=" * 60)
    print(f"\n📊 Tuning Summary:")
    print(f"   - Best tuned model: {best_tuned_model}")
    print(f"   - CV Accuracy: {tuned_results[best_tuned_model]*100:.2f}%")
    print(f"   - 10-fold CV mean: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
    
    return best_tuned_model, tuning_summary

# Run the function
if __name__ == "__main__":
    best_model, summary = tune_best_model()