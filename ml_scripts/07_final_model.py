"""
STEP 7: FINAL MODEL SELECTION & SAVE
File: ml_scripts/07_final_model.py

Is script mein:
1. Best tuned model select karenge
2. Final model ko poore training data pe train karenge
3. Test data pe final evaluation karenge
4. Model ko save karenge for deployment
5. Prediction function banayenge
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

def final_model():
    """
    Final model select karna aur save karna
    """
    print("=" * 60)
    print("STEP 7: FINAL MODEL SELECTION & SAVE")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading data...")
    X_train = pd.read_csv('../ml_output/X_train_scaled.csv')
    X_test = pd.read_csv('../ml_output/X_test_scaled.csv')
    y_train = pd.read_csv('../ml_output/y_train.csv').values.ravel()
    y_test = pd.read_csv('../ml_output/y_test.csv').values.ravel()
    
    print(f"   Training: {X_train.shape}")
    print(f"   Testing: {X_test.shape}")
    
    # 2. Load best tuned model
    print("\n2. Loading best tuned model...")
    
    # Try to load tuned models and find best
    tuned_rf = joblib.load('../ml_output/models/tuned_random_forest.pkl')
    tuned_svm = joblib.load('../ml_output/models/tuned_svm.pkl')
    tuned_knn = joblib.load('../ml_output/models/tuned_knn.pkl')
    
    # Evaluate each on test data
    models = {
        'Random Forest': tuned_rf,
        'SVM': tuned_svm,
        'KNN': tuned_knn
    }
    
    print("\n   Evaluating tuned models on test data:")
    print("   " + "-" * 40)
    
    best_model_name = None
    best_accuracy = 0
    best_model = None
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"      {model_name}: {acc*100:.2f}%")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = model_name
            best_model = model
    
    print(f"\n   🏆 Best model on test data: {best_model_name} ({best_accuracy*100:.2f}%)")
    
    # 3. Train final model on complete training data
    print("\n3. Training final model on complete training data...")
    
    # Best model already trained, we'll use it
    final_model = best_model
    
    # 4. Final evaluation
    print("\n4. Final evaluation on test data...")
    
    y_pred_final = final_model.predict(X_test)
    
    final_accuracy = accuracy_score(y_test, y_pred_final)
    final_cm = confusion_matrix(y_test, y_pred_final)
    final_report = classification_report(y_test, y_pred_final, 
                                        target_names=['Setosa', 'Versicolor', 'Virginica'])
    
    print(f"\n   ✅ Final Model Accuracy: {final_accuracy*100:.2f}%")
    print(f"\n   Confusion Matrix:")
    print(f"   {final_cm}")
    print(f"\n   Classification Report:")
    print(final_report)
    
    # 5. Save final model
    print("\n5. Saving final model...")
    
    joblib.dump(final_model, '../ml_output/models/final_model.pkl')
    print(f"   ✅ Saved: ml_output/models/final_model.pkl")
    
    # Save model info
    model_info = {
        'model_name': best_model_name,
        'accuracy': final_accuracy,
        'test_samples': len(X_test),
        'features': list(X_train.columns)
    }
    
    with open('../ml_output/models/final_model_info.txt', 'w') as f:
        f.write("FINAL MODEL INFORMATION\n")
        f.write("=" * 40 + "\n")
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"   ✅ Saved: ml_output/models/final_model_info.txt")
    
    # 6. Create prediction function
    print("\n6. Creating prediction function...")
    
    # Load scaler for new predictions
    scaler = joblib.load('../ml_output/scaler.pkl')
    
    def predict_species(sepal_length, sepal_width, petal_length, petal_width):
        """
        New flower measurements ke liye species predict karna
        
        Parameters:
        - sepal_length: Sepal length in cm
        - sepal_width: Sepal width in cm
        - petal_length: Petal length in cm
        - petal_width: Petal width in cm
        
        Returns:
        - species_name: Predicted species (Setosa, Versicolor, or Virginica)
        """
        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = final_model.predict(features_scaled)[0]
        
        # Convert number to species name
        species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        
        return species_mapping[prediction]
    
    # Save prediction function
    with open('../ml_output/predict_function.py', 'w') as f:
        f.write("""
# Prediction function for Iris species
import numpy as np
import joblib

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    '''
    Predict Iris species from measurements
    '''
    # Load model and scaler
    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('../scaler.pkl')
    
    # Create and scale features
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    
    return species_mapping[prediction]
""")
    
    print(f"   ✅ Saved: ml_output/predict_function.py")
    
    # 7. Test prediction function
    print("\n7. Testing prediction function...")
    
    test_cases = [
        (5.1, 3.5, 1.4, 0.2),  # Should be Setosa
        (5.9, 3.0, 5.1, 1.8),  # Should be Versicolor or Virginica
        (6.5, 3.0, 5.5, 2.0),  # Should be Virginica
    ]
    
    print("\n   Test Predictions:")
    print("   " + "-" * 50)
    for sl, sw, pl, pw in test_cases:
        result = predict_species(sl, sw, pl, pw)
        print(f"   Measurements: {sl}, {sw}, {pl}, {pw} → Predicted: {result}")
    
    # 8. Final summary
    print("\n" + "=" * 60)
    print("✅ FINAL MODEL READY FOR DEPLOYMENT!")
    print("=" * 60)
    print(f"\n📊 Final Model Summary:")
    print(f"   - Model: {best_model_name}")
    print(f"   - Test Accuracy: {final_accuracy*100:.2f}%")
    print(f"   - Model saved in: ml_output/models/final_model.pkl")
    print(f"   - Prediction function saved in: ml_output/predict_function.py")
    
    return final_model, predict_species

# Run the function
if __name__ == "__main__":
    model, predict_func = final_model()