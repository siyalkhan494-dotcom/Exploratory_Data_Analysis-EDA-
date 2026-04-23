
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
