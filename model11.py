# app/model.py
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('app/data/diabetes_model.h5')

# Load the scaler
scaler = np.load('app/data/scaler.npy', allow_pickle=True).item()
