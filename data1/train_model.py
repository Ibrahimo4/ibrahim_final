import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
#import matplotlib.pyplot as plt
#import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Encode categorical variables
gender_encoder = LabelEncoder()
smoking_history_encoder = LabelEncoder()

data['gender'] = gender_encoder.fit_transform(data['gender'])
data['smoking_history'] = smoking_history_encoder.fit_transform(data['smoking_history'])

# Split the data into features and target
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('diabetes_model.h5')

# Save the scaler and encoders
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('gender_encoder.pkl', 'wb') as f:
    pickle.dump(gender_encoder, f)
with open('smoking_history_encoder.pkl', 'wb') as f:
    pickle.dump(smoking_history_encoder, f)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('static/model_accuracy.png')
plt.clf()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('static/model_loss.png')
plt.clf()
