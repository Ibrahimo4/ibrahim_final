import os
from flask import render_template, current_app as app, jsonify, send_from_directory, request
import numpy as np
from .model import model, scaler, model_medical, scaler_medical, model_medical2, scaler_medical2, gender_encoder, smoking_history_encoder
from .forms import DiagnoseForm, MedicalExamForm, MedicalExamForm2
from tensorflow.keras.models import load_model
#import numpy as np
from .model1 import predict_outcome
import joblib
import pickle
@app.route("/", methods=['GET'])
def home():
    return render_template("home.html", title='Home')

@app.route("/medical_exam1", methods=['GET'])
def medical_exam1():
    form = DiagnoseForm()
    return render_template("medical_exam1.html", form=form, title='Diagnose')

@app.route('/medical_diagnosis1', methods=['POST'])
def medical_diagnosis1():
    form = DiagnoseForm()
    if form.validate_on_submit():
        form_dict = form.data
        form_dict.pop('csrf_token')
        form_dict.pop('submit')
        form_dict['gender'] = (form_dict['gender'] == 'True')
        features = list(form_dict.values())

        input_data = np.array([features])
        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)
        prediction = (prediction > 0.5).astype(int)

        accuracy = model.predict(input_data_scaled).max() * 100
        accuracy = "{:.2f}".format(round(accuracy, 2))

        results = {'prediction': 'Positive' if prediction[0][0] == 1 else 'Negative', 'accuracy': accuracy}
        return render_template('medical_result1.html', prediction=results['prediction'], accuracy=results['accuracy'])

    return render_template('error.html', message=form.errors)

@app.route("/about")
def about():
    return render_template("about.html", title='About')

@app.route("/report")
def report():
    return render_template("report.html", title='Report')

@app.route("/report2")
def report2():
    return render_template("report2.html", title='Report')

@app.route("/report3")
def report3():
    return render_template("report3.html", title='Report')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            age = int(request.form['Age'])
            gender = 1 if request.form['Gender'] == 'Male' else 0
            polyuria = int(request.form['Polyuria'])
            polydipsia = int(request.form['Polydipsia'])
            weight_loss = int(request.form['Weight_loss'])
            polyphagia = int(request.form['Polyphagia'])
            visual_blurring = int(request.form['visual_blurring'])
            itching = int(request.form['Itching'])
            irritability = int(request.form['Irritability'])
            delayed_healing = int(request.form['delayed_healing'])
            muscle_stiffness = int(request.form['muscle_stiffness'])
            alopecia = int(request.form['Alopecia'])
            weakness = int(request.form['weakness'])
            partial_paresis = int(request.form['partial_paresis'])
            obesity = int(request.form['Obesity'])

            input_data = np.array([[age, gender, polyuria, polydipsia, weight_loss, polyphagia, visual_blurring, itching, irritability, delayed_healing, muscle_stiffness, alopecia, weakness, partial_paresis, obesity]])
            input_data_scaled = scaler.transform(input_data)
            my_prediction = model.predict(input_data_scaled)
            my_prediction = (my_prediction > 0.5).astype(int)[0][0]

            if my_prediction == 1:
                prediction_text = 'Positive (Early Diabetes Detected)'
            else:
                prediction_text = 'Negative (No Early Diabetes Detected)'

            return render_template('predict.html', prediction=prediction_text)
        except ValueError:
            return render_template('error.html', message="Invalid input. Please enter valid integers for all fields.")
        except Exception as e:
            return render_template('error.html', message=str(e))

@app.route("/medical_exam", methods=['GET'])
def medical_exam():
    form = MedicalExamForm()
    return render_template("medical_exam.html", form=form, title='Medical Exam')

@app.route('/medical_diagnosis', methods=['POST'])
def medical_diagnosis():
    form = MedicalExamForm()
    if form.validate_on_submit():
        form_dict = form.data
        form_dict.pop('csrf_token')
        form_dict.pop('submit')
        features = list(form_dict.values())

        input_data = np.array([features])
        input_data_scaled = scaler_medical.transform(input_data)

        prediction = model_medical.predict(input_data_scaled)
        prediction = (prediction > 0.5).astype(int)

        accuracy = model_medical.predict(input_data_scaled).max() * 100
        accuracy = "{:.2f}".format(round(accuracy, 2))

        results = {'prediction': 'Positive' if prediction[0][0] == 1 else 'Negative', 'accuracy': accuracy}
        return render_template('medical_result.html', prediction=results['prediction'], accuracy=results['accuracy'])

    return render_template('error.html', message=form.errors)

'''@app.route("/medical_exam2", methods=['GET'])
def medical_exam2():
    form = MedicalExamForm2()
    return render_template("medical_exam2.html", form=form, title='Medical Exam 2')

@app.route('/medical_diagnosis2', methods=['POST'])
def medical_diagnosis2():
    form = MedicalExamForm2()
    prediction = None
    if form.validate_on_submit():
        gender = form.gender.data
        age = form.age.data
        hypertension = form.hypertension.data
        heart_disease = form.heart_disease.data
        smoking_history = form.smoking_history.data
        bmi = form.bmi.data
        HbA1c_level = form.HbA1c_level.data
        blood_glucose_level = form.blood_glucose_level.data
        
        # Encode categorical variables
        gender = gender_encoder.transform([gender])[0]
        smoking_history = smoking_history_encoder.transform([smoking_history])[0]
        
        # Create feature array
        features = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
        features = scaler_medical2.transform(features)
        
        # Predict
        prediction = model_medical2.predict(features)
        prediction = (prediction > 0.5).astype(int)[0][0]

        accuracy = model_medical2.predict(features).max() * 100
        accuracy = "{:.2f}".format(round(accuracy, 2))

        results = {'prediction': 'Positive' if prediction[0][0] == 1 else 'Negative', 'accuracy': accuracy}
        return render_template('medical_result2.html', prediction=results['prediction'], accuracy=results['accuracy'])
 
    return render_template('error.html', message=form.errors)
'''

models2 = load_model('app/data1/diabetes_model.h5')

# Load the encoders and scaler
with open('app/data1/gender_encoder.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)
with open('app/data1/smoking_history_encoder.pkl', 'rb') as f:
    smoking_history_encoder = pickle.load(f)
with open('app/data1/scaler.pkl', 'rb') as f:
    scalers2 = pickle.load(f)
    
@app.route("/medical_exam2", methods=['GET'])
def medical_exam2():
    form = MedicalExamForm2()
    return render_template("medical_exam2.html", form=form, title='Medical Exam 2')
'''
@app.route('/medical_diagnosis2', methods=['POST'])
def medical_diagnosis2():
    if request.method == 'POST':
        try:
            gender = gender_encoder.transform([request.form['gender']])[0]
            age = float(request.form['age'])
            hypertension = int(request.form['hypertension'])
            heart_disease = int(request.form['heart_disease'])
            smoking_history = smoking_history_encoder.transform([request.form['smoking_history']])[0]
            bmi = float(request.form['bmi'])
            HbA1c_level = float(request.form['HbA1c_level'])
            blood_glucose_level = float(request.form['blood_glucose_level'])

            features = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
            features = scalers2.transform(features)

            prediction = models2.predict(features)
            prediction = (prediction > 0.5).astype(int)[0][0]
            accuracy = models2.predict(features).max() * 100
            accuracy = "{:.2f}".format(round(accuracy, 2))

            result = {'prediction': 'Positive' if prediction == 1 else 'Negative', 'accuracy': accuracy}
            return render_template('medical_result2.html', prediction=result['prediction'], accuracy=result['accuracy'])
        except ValueError as e:
            return render_template('error.html', message=str(e))

    return render_template('error.html', message="Invalid request method")


@app.route('/medical_diagnosis2', methods=['POST'])
def medical_diagnosis2():
 # prediction = None
  form = MedicalExamForm2()
    if form.validate_on_submit():
        gender = gender_encoder.transform([request.form['Gender']])[0]
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = smoking_history_encoder.tr<ansform([request.form['smoking_history']])[0]
        bmi = float(request.form['bmi'])
        HbA1c_level = float(request.form['HbA1c_level'])
        blood_glucose_level = float(request.form['bl<ood_glucose_level'])

        features = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level,blood_glucose_level]])
        features = scalers2.transform(features)

        prediction = models2.predict(features)
        prediction = (prediction > 0.5).astype(int)[0][0]
        accuracy = model_medical2.predict(features).max() * 100
        accuracy = "{:.2f}".format(round(accuracy, 2))

        results = {'prediction': 'Positive' if prediction[0][0] == 1 else 'Negative', 'accuracy': accuracy}
        return render_template('medical_result2.html', prediction=results['prediction'], accuracy=results['accuracy'])
        
        
 
#    return render_template('error.html', message=form.errors)       

@app.route('/medical_diagnosis2', methods=['POST'])
def medical_diagnosis2():
    form = MedicalExamForm2()
    if form.validate_on_submit():
        gender = gender_encoder.transform([form.gender.data])[0]
        age = float(form.age.data)
        hypertension = int(form.hypertension.data)
        heart_disease = int(form.heart_disease.data)
        smoking_history = smoking_history_encoder.transform([form.smoking_history.data])[0]
        bmi = float(form.bmi.data)
        HbA1c_level = float(form.HbA1c_level.data)
        blood_glucose_level = float(form.blood_glucose_level.data)

        features = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
        features = scalers2.transform(features)

        prediction = models2.predict(features)
        prediction = (prediction > 0.5).astype(int)[0][0]
        accuracy = models2.predict(features).max() * 100
        accuracy = "{:.2f}".format(round(accuracy, 2))

        result = 'Positive' if prediction == 1 else 'Negative'

        return render_template('medical_result2.html', prediction=result, accuracy=accuracy)
    return render_template('error.html', message=form.errors)

'''

import pandas as pd
df = pd.read_csv('app/data1/diabetes.csv')
feature_columns = df.drop(columns=['diabetes']).columns.tolist()

@app.route('/medical_diagnosis2', methods=['POST'])
def medical_diagnosis2():

    form = MedicalExamForm2()
    if form.validate_on_submit():
        data = {field.name: field.data for field in form}
        data.pop('submit')
        classification, probability = predict_outcome(data, feature_columns)
        return render_template('medical_result2.html', classification=classification, probability=probability)
    return render_template('home.html', form=form)
