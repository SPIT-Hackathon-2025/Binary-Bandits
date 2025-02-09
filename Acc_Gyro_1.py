import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load dataset
file_path = r"c:\Users\91750\Downloads\Acc_Gyro_data.xlsx"
df = pd.read_excel(file_path)

# Select features and output label
features = ['t_body_acc_mean()_X', 't_body_acc_mean()_Y', 't_body_acc_mean()_Z',
            't_body_gyro_mean()_X', 't_body_gyro_mean()_Y', 't_body_gyro_mean()_Z']
output_label = 'activity_Id'

X = df[features].values  # Sensor features
y = df[output_label].values  # Activity labels

# Normalize input features
scaler = MinMaxScaler(feature_range=(-1, 1))  # Use (-1,1) instead of (0,1) for better LSTM learning
X = scaler.fit_transform(X)

# Encode activity labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)  # Convert labels to one-hot encoding

# Function to create sequences (longer sequences help LSTMs)
def create_sequences(X, y, time_steps=30):  # Increased time steps from 10 → 30
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])  # Sequence of 30 time steps
        ys.append(y[i + time_steps])    # Corresponding label
    return np.array(Xs), np.array(ys)

time_steps = 30  # Increased for better temporal learning
X, y = create_sequences(X, y, time_steps)

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Define an improved LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(time_steps, len(features))),  # More LSTM units
    BatchNormalization(),
    Dropout(0.3),  # Increased dropout to reduce overfitting
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y.shape[1], activation='softmax')  # Output layer for classification
])

# Compile model with Adam optimizer and learning rate scheduling
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {acc * 100:.2f}%')

# Save model for mobile deployment
model.save("activity_recognition_model.h5")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.legend()
plt.title("LSTM Model Accuracy")
plt.show()
