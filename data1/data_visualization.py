import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Encode categorical variables
gender_encoder = LabelEncoder()
smoking_history_encoder = LabelEncoder()

# Fit on all possible values
data['gender'] = gender_encoder.fit_transform(data['gender'])
data['smoking_history'] = smoking_history_encoder.fit_transform(data['smoking_history'])

# Save the encoders
with open('gender_encoder.pkl', 'wb') as f:
    pickle.dump(gender_encoder, f)

with open('smoking_history_encoder.pkl', 'wb') as f:
    pickle.dump(smoking_history_encoder, f)

# Plotting the distribution of the target variable
sns.countplot(x='diabetes', data=data)
plt.title('Distribution of Diabetes')
plt.savefig('static/diabetes_distribution.png')
plt.clf()

# Plotting the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('static/correlation_heatmap.png')
plt.clf()

# Plotting histograms for numerical features
data.hist(bins=30, figsize=(20, 15))
plt.savefig('static/histograms.png')
plt.clf()

# Pairplot to see pairwise relationships
sns.pairplot(data, hue='diabetes')
plt.savefig('static/pairplot.png')
plt.clf()

# Split the data into features and target
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
