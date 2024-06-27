import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pandas as pd

model = load_model('app/model.h5')
scaler = joblib.load('app/scaler.pkl')
label_encoders = joblib.load('app/label_encoders.pkl')

def preprocess_input(data, feature_columns):
    df = pd.DataFrame(data, index=[0])
    
    # Apply label encoding for categorical columns
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])
    
    df = df.reindex(columns=feature_columns, fill_value=0)
    features = scaler.transform(df)
    return features

def predict_outcome(data, feature_columns):
    features = preprocess_input(data, feature_columns)
    probability = model.predict(features)[0][0]
    classification = 'Positive' if probability > 0.5 else 'Negative'
    return classification, probability
