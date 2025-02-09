import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd

# Load the saved LSTM model
model_path = "activity_recognition_model.h5"  # Update with the correct path
model = load_model(model_path)

# Load dataset for testing
file_path = r"c:\Users\91750\Downloads\Acc_Gyro_data.xlsx"
df = pd.read_excel(file_path)

# Select features and output label
features = ['t_body_acc_mean()_X', 't_body_acc_mean()_Y', 't_body_acc_mean()_Z',
            't_body_gyro_mean()_X', 't_body_gyro_mean()_Y', 't_body_gyro_mean()_Z']
output_label = 'activity_Id'

X = df[features].values  # Sensor features
y = df[output_label].values  # Activity labels

# Normalize input features
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

# Encode activity labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

def create_sequences(X, time_steps=5):
    """Create sequences of fixed time steps."""
    Xs = []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:i + time_steps])
    return np.array(Xs)

time_steps = 30  # Using 5 timestamps as input
X_test_seq = create_sequences(X, time_steps)

# Make predictions
predictions = model.predict(X_test_seq)
predicted_labels = np.argmax(predictions, axis=1)

# Decode predictions
predicted_activities = encoder.inverse_transform(predicted_labels)
print("Predicted Activities:", predicted_activities)