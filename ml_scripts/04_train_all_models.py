"""
STEP 4: TRAIN ALL CLASSIFICATION MODELS
File: ml_scripts/04_train_all_models.py

Is script mein:
1. Sabhi models train karenge
2. Models ko save karenge
3. Training time measure karenge
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import os
import time

def train_all_models():
    """
    Sabhi models ko train karna
    """
    print("=" * 60)
    print("STEP 4: TRAINING ALL CLASSIFICATION MODELS")
    print("=" * 60)
    
    # ============================================
    # IMPORTANT: Create folders if they don't exist
    # ============================================
    print("\n📁 Creating output folders...")
    os.makedirs('../ml_output', exist_ok=True)
    os.makedirs('../ml_output/models', exist_ok=True)
    print("   ✅ Folders created/verified")
    
    # 1. Load scaled data
    print("\n1. Loading scaled data...")
    X_train = pd.read_csv('../ml_output/X_train_scaled.csv')
    y_train = pd.read_csv('../ml_output/y_train.csv')
    y_train = y_train.values.ravel()  # Convert to 1D array
    
    print(f"   Training features: {X_train.shape}")
    print(f"   Training target: {y_train.shape}")
    
    # 2. Define all models
    print("\n2. Creating model instances...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine (SVM)': SVC(random_state=42)
    }
    
    print(f"   Total models to train: {len(models)}")
    
    # 3. Train each model
    print("\n3. Training models...")
    print("-" * 50)
    
    trained_models = {}
    training_times = {}
    
    for model_name, model in models.items():
        print(f"\n   Training: {model_name}")
        
        # Measure training time
        start_time = time.time()
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Calculate training time
        end_time = time.time()
        train_time = end_time - start_time
        
        # Store trained model
        trained_models[model_name] = model
        training_times[model_name] = train_time
        
        print(f"      ✅ Training complete in {train_time:.3f} seconds")
        
        # Save individual model (FIXED: Create safe filename)
        model_filename = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace("'", '').replace('-', '_')
        model_path = f'../ml_output/models/{model_filename}.pkl'
        joblib.dump(model, model_path)
        print(f"      💾 Saved: {model_path}")
    
    # 4. Display training times summary
    print("\n" + "=" * 50)
    print("Training Times Summary:")
    print("-" * 50)
    for model_name, train_time in training_times.items():
        print(f"   {model_name}: {train_time:.4f} seconds")
    
    # 5. Save all models in one file (optional)
    print("\n4. Saving all models together...")
    all_models_path = '../ml_output/models/all_models.pkl'
    joblib.dump(trained_models, all_models_path)
    print(f"   ✅ Saved: {all_models_path}")
    
    # 6. Save model names for reference
    print("\n5. Saving model names...")
    with open('../ml_output/models/model_names.txt', 'w') as f:
        f.write("TRAINED MODELS\n")
        f.write("=" * 40 + "\n")
        for model_name in models.keys():
            f.write(f"- {model_name}\n")
        f.write("\n" + "=" * 40 + "\n")
        f.write(f"Total models: {len(models)}\n")
        f.write(f"Training data size: {X_train.shape[0]} samples\n")
        f.write(f"Features: {X_train.shape[1]}\n")
    
    print(f"   ✅ Saved: ml_output/models/model_names.txt")
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n📊 Models trained: {len(models)}")
    print(f"   - Training data size: {X_train.shape[0]} samples")
    print(f"   - Features: {X_train.shape[1]}")
    print(f"   - All models saved in: ml_output/models/")
    
    # List saved files
    print("\n📁 Saved model files:")
    for model_name in models.keys():
        model_filename = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace("'", '').replace('-', '_')
        print(f"   - {model_filename}.pkl")
    print(f"   - all_models.pkl")
    print(f"   - model_names.txt")
    
    return trained_models

# Run the function
if __name__ == "__main__":
    models = train_all_models()