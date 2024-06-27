# app/model.py
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import pickle
# Load the trained model
model = load_model('app/data/diabetes_model.h5')

# Load the scaler
scaler = np.load('app/data/scaler.npy', allow_pickle=True).item()


model_medical2  = load_model('app/data1/diabetes_model.h5')

# Load the encoders and scaler
with open('app/data1/gender_encoder.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)
with open('app/data1/smoking_history_encoder.pkl', 'rb') as f:
    smoking_history_encoder = pickle.load(f)
with open('app/data1/scaler.pkl', 'rb') as f:
    scaler_medical2 = pickle.load(f)



# Load the trained model for symptoms


# Load the trained model for medical examinations
model_medical = load_model('app/data/diabetes_model2.h5')
scaler_medical = joblib.load('app/data/scaler2.pkl')
