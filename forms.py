from flask_wtf import FlaskForm
from wtforms import SubmitField, BooleanField, FloatField, RadioField, StringField 
from wtforms.validators import DataRequired, NumberRange
#from flask_wtf import FlaskForm
from wtforms import BooleanField, RadioField, IntegerField, SubmitField
from wtforms.validators import InputRequired
from wtforms_html5 import AutoAttrMeta
from wtforms.widgets import NumberInput
class DiagnoseForm(FlaskForm):
    age = FloatField('Age', validators=[DataRequired(), NumberRange(min=0, max=120)])
    gender = RadioField('Gender', choices=[(1, 'Male'), (0, 'Female')], validators=[DataRequired()])
    polyuria = BooleanField('Polyuria')
    polydipsia = BooleanField('Polydipsia')
    sudden_wl = BooleanField('Sudden Weight Loss')
    weakness = BooleanField('Weakness')
    polyphagia = BooleanField('Polyphagia')
    genital_thrush = BooleanField('Genital Thrush')
    visual_blurring = BooleanField('Visual Blurring')
    itching = BooleanField('Itching')
    irritability = BooleanField('Irritability')
    delayed_healing = BooleanField('Delayed Healing')
    partial_paresis = BooleanField('Partial Paresis')
    muscle_stiffness = BooleanField('Muscle Stiffness')
    alopecia = BooleanField('Alopecia')
    obesity = BooleanField('Obesity')
    submit = SubmitField('Submit')

class MedicalExamForm(FlaskForm):
    pregnancies = IntegerField('Pregnancies', validators=[DataRequired(), NumberRange(min=0)])
    glucose = IntegerField('Glucose', validators=[DataRequired(), NumberRange(min=0)])
    blood_pressure = IntegerField('Blood Pressure', validators=[DataRequired(), NumberRange(min=0)])
    skin_thickness = IntegerField('Skin Thickness', validators=[DataRequired(), NumberRange(min=0)])
    insulin = IntegerField('Insulin', validators=[DataRequired(), NumberRange(min=0)])
    bmi = FloatField('BMI', validators=[DataRequired(), NumberRange(min=0)])
    diabetes_pedigree_function = FloatField('Diabetes Pedigree Function', validators=[DataRequired(), NumberRange(min=0)])
    age = IntegerField('Age', validators=[DataRequired(), NumberRange(min=0, max=120)])
    submit = SubmitField('Predict')

'''
class MedicalExamForm2(FlaskForm):
    gender = RadioField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')], validators=[DataRequired()])
    age = FloatField('Age', validators=[DataRequired(), NumberRange(min=0, max=120)])
    hypertension = BooleanField('Hypertension')
    heart_disease = BooleanField('Heart Disease')
    smoking_history = RadioField('Smoking History', choices=[
        ('current', 'Current'), 
        ('ever', 'Ever'), 
        ('former', 'Former'), 
        ('never', 'Never'), 
        ('not_current', 'Not Current')
    ], validators=[DataRequired()])
    bmi = FloatField('BMI', validators=[DataRequired(), NumberRange(min=0)])
    HbA1c_level = FloatField('HbA1c Level', validators=[DataRequired(), NumberRange(min=0)])
    blood_glucose_level = IntegerField('Blood Glucose Level', validators=[DataRequired(), NumberRange(min=0)])
    submit = SubmitField('Predict')
'''
class MedicalExamForm2(FlaskForm):
    gender = RadioField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')], validators=[DataRequired()])
    age = FloatField('Age', validators=[DataRequired(), NumberRange(min=0, max=120)])
    hypertension = BooleanField('Hypertension')
    heart_disease = BooleanField('Heart Disease')
    smoking_history = RadioField('Smoking History', choices=[
        ('current', 'Current'), 
        ('ever', 'Ever'), 
        ('former', 'Former'), 
        ('never', 'Never'), 
        ('not_current', 'Not Current')
    ], validators=[DataRequired()])
    bmi = FloatField('BMI', validators=[DataRequired(), NumberRange(min=0)])
    HbA1c_level = FloatField('HbA1c Level', validators=[DataRequired(), NumberRange(min=0)])
    blood_glucose_level = IntegerField('Blood Glucose Level', validators=[DataRequired(), NumberRange(min=0)])
    submit = SubmitField('Predict')


