from flask import Flask
from config import Config
import pickle
from flask_babel import Babel

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Construct core Flask application.
def init_app():
    app = Flask(__name__)
    # import configuration
    app.config.from_object(Config)
    with app.app_context():
        # Import parts of our core Flask app
        from . import routes

        # Import Dash application
        from .plotlydash.dashboard import init_dashboard
        app = init_dashboard(app)

        return app


# load machine learning models
#rf_model = pickle.load(open('app/data/moodel.pkl', 'rb'))

model = load_model('app/data/diabetes_model.h5')
scaler = np.load('app/data/scaler.npy', allow_pickle=True).item()
