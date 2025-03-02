import joblib
import os
import requests
import random
import numpy as np
from dotenv import load_dotenv
import tensorflow as tf
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model  # Explicitly import load_model and Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# ✅ Load API Key from .env
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# ✅ Fix: Ensure `models/` is correctly referenced
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the project root
MODELS_DIR = os.path.join(BASE_DIR, "models")  # ✅ Now correctly points to `models/`

# --- StockLSTM Class (as provided earlier) ---
class StockLSTM:
    def __init__(self, sequence_length=60):
        """
        Initializes the LSTM model without requiring data.

        :param sequence_length: Number of past days to use for prediction.
        """
        self.sequence_length = sequence_length  # Number of past days for prediction
        self.scaler = MinMaxScaler(feature_range=(0,1))  # Normalization
        self.model = None  # Placeholder for the LSTM model
        self.history = None  # Placeholder for training history
        self.predictions = None
        self.y_test_actual = None
        self.scaled_data = None  # Placeholder for scaled data

    def preprocess_data(self, data):
        """Scales data and prepares training and testing sets."""
        self.data = data  # Store original data
        self.scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1,1))
        self.X, self.y = self.create_sequences(self.scaled_data)

        # Split dataset: 80% training, 20% testing
        train_size = int(len(self.X) * 0.8)
        self.X_train, self.y_train = self.X[:train_size], self.y[:train_size] # corrected typo: y_train.[:train_size] -> y[:train_size]
        self.X_test, self.y_test = self.X[train_size:], self.y[train_size:]

    def create_sequences(self, data):
        """
        Converts time-series data into sequences for LSTM input.

        :param data: Scaled time-series data
        :return: X (features) and y (target)
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length])
        return np.array(X), np.array(y)

    def build_model(self):
        """Defines and compiles the LSTM model."""
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def train_model(self, data, epochs=50, batch_size=32):
        """
        Trains the LSTM model. Data must be provided.

        :param data: DataFrame containing the closing prices.
        :param epochs: Number of epochs to train.
        :param batch_size: Batch size for training.
        """
        if data is None:
            raise ValueError("Training data must be provided.")

        # Preprocess the data before training
        self.preprocess_data(data)

        if self.model is None:
            self.build_model()

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

        self.history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stopping]
        )

    def predict_test_set(self):
        """Predicts values for the test set and compares with actual prices."""
        if self.model is None or self.X_test is None:
            raise ValueError("Model is not trained or test data is unavailable.")

        # Generate predictions
        self.predictions = self.model.predict(self.X_test)
        self.predictions = self.scaler.inverse_transform(self.predictions)

        # Store actual test values
        self.y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1,1))

        # Plot actual vs predicted values
        plt.figure(figsize=(12,6))
        plt.plot(self.data.index[-len(self.y_test_actual):], self.y_test_actual,
                 label="Actual Prices", color="blue")
        plt.plot(self.data.index[-len(self.predictions):], self.predictions,
                 label="Predicted Prices", linestyle="dashed", color="red")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.title("Stock Price Prediction vs Actual")
        plt.legend()
        plt.show()

        return self.predictions, self.y_test_actual


    def predict_single_day(self, past_60_days):
        """
        Predicts the closing price for the next day based on the past 60 days of closing prices.

        :param past_60_days: A Pandas Series containing the last 60 days of closing prices (index must be DatetimeIndex).
        :return: A tuple (predicted_date as 'YYYY-MM-DD', predicted_price).
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet. Train the model before making predictions.")

        # Ensure input is a NumPy array and reshape for the model
        past_60_days_values = np.array(past_60_days.values).reshape(-1, 1)
        scaled_input = self.scaler.transform(past_60_days_values)
        scaled_input = scaled_input.reshape(1, self.sequence_length, 1)

        # Make the prediction
        predicted_price = self.model.predict(scaled_input)
        predicted_price = self.scaler.inverse_transform(predicted_price)

        # Extract the last available date from the Pandas Series index
        last_date = past_60_days.index[-1]

        # Move to the next valid business day
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:  # Skip weekends (Saturday=5, Sunday=6)
            next_date += timedelta(days=1)

        # Format the next date as a string
        predicted_date_str = next_date.strftime('%Y-%m-%d')

        return predicted_date_str, predicted_price[0, 0]


    def compute_rmse(self):
        """Computes RMSE (Root Mean Squared Error) between actual and predicted prices."""
        if self.y_test_actual is None or self.predictions is None:
            raise ValueError("Predictions or actual values are missing. Run predict_test_set() first.")

        rmse = np.sqrt(mean_squared_error(self.y_test_actual, self.predictions))
        print(f"RMSE: {rmse:.2f}")
        return rmse

    def save_model(self, filename="lstm_stock_model.h5"):
        """Saves the trained model."""
        if self.model:
            self.model.save(filename)
            print(f"Model saved as {filename}")
        else:
            print("No model to save. Train or load a model first.")

    def load_model(self, filename="lstm_stock_model.h5"):
        """Loads a previously saved model."""
        self.model = tf.keras.models.load_model(filename) # Corrected: use tf.keras.models.load_model for .h5
        print(f"Model loaded from {filename}")
        return self.model  # <--- ⚠️ IMPORTANT: Return just the Keras model for now for compatibility

def get_available_models():
    """List available models in the models directory."""
    print(f"Debug: MODELS_DIR is: {MODELS_DIR}") # Keep debug prints for now
    try:
        files_in_models_dir = os.listdir(MODELS_DIR)
        print(f"Debug: Files in MODELS_DIR: {files_in_models_dir}")
        return [f for f in files_in_models_dir if f.endswith(".h5")]
    except FileNotFoundError:
        print(f"Debug: MODELS_DIR not found: {MODELS_DIR}")
        return [] # Return empty list if directory not found

def load_model(model_name): # Keep this load_model function, but modify it
    """Load a Keras .h5 model and return a StockLSTM instance."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        try:
            # Load the Keras model using TensorFlow's load_model
            keras_model = tf.keras.models.load_model(model_path)
            # Create a StockLSTM instance and attach the loaded Keras model
            lstm_model = StockLSTM() # ✅ Instantiate StockLSTM here
            lstm_model.model = keras_model # Assign the loaded Keras model

            print(f"✅ Successfully loaded Keras .h5 model {model_name} into StockLSTM.")
            print(f"Debug - load_model() in utils/model_utils.py is returning type: {type(lstm_model)}") # Debug print here! - Now returning lstm_model

            return lstm_model # ✅ Return the StockLSTM instance - Corrected to return StockLSTM instance
            # return keras_model # Return just the Keras model for now - directly return keras_model # <- OLD: Incorrectly returning Keras model

        except Exception as e:
            print(f"⚠️ Error loading Keras .h5 model {model_name} into StockLSTM: {e}")
            return None
    return None