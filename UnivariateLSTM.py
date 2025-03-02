from pickle import NONE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

class StockLSTM:
    def __init__(self, data, sequence_length=60):
        """
        Initializes the LSTM model for stock prediction.
        
        :param data: DataFrame containing the closing prices.
        :param sequence_length: Number of past days to use for prediction.
        """
        self.data = data  # Original data
        self.sequence_length = sequence_length  # Number of past days for prediction
        self.scaler = MinMaxScaler(feature_range=(0,1))  # Normalization
        self.model = None  # Placeholder for the LSTM model
        self.history = None  # Placeholder for training history
        self.predictions = None
        self.y_test_actual = None
        
        # Preprocess the data
        self.preprocess_data()

    def preprocess_data(self):
        """Scales data and prepares training and testing sets."""
        self.scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1,1))
        self.X, self.y = self.create_sequences(self.scaled_data)
        
        # Split dataset: 80% training, 20% testing
        train_size = int(len(self.X) * 0.8)
        self.X_train, self.y_train = self.X[:train_size], self.y[:train_size]
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

    def train_model(self, epochs=50, batch_size=32):
        """
        Trains the LSTM model.
        
        :param epochs: Number of epochs to train.
        :param batch_size: Batch size for training.
        """
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

    def evaluate_model(self):
        """Evaluates model performance and plots training history."""
        plt.figure(figsize=(10,5))
        plt.plot(self.history.history['loss'], label="Training Loss", color="blue")
        plt.plot(self.history.history['val_loss'], label="Validation Loss", color="red", linestyle="dashed")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("LSTM Training & Validation Loss")
        plt.legend()
        plt.show()

    def predict_test_set(self):
        """Predicts values for the test set and compares with actual prices."""
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

        # Return for optional usage
        return self.predictions, self.y_test_actual


    def forecast_future(self, future_days=7):
        """
        Predicts future stock prices.
        
        :param future_days: Number of days to forecast
        :return: Future dates and predicted prices
        """
        future_predictions = []
        last_sequence = self.scaled_data[-self.sequence_length:]

        for _ in range(future_days):
            next_prediction = self.model.predict(last_sequence.reshape(1, self.sequence_length, 1))
            future_predictions.append(next_prediction[0, 0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = next_prediction

        future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))
        future_dates = pd.date_range(self.data.index[-1], periods=future_days+1, freq='B')[1:]

        # Plot future predictions
        plt.figure(figsize=(12,6))
        plt.plot(self.data.index[-100:], self.data[-100:], label="Actual Prices", color="blue")
        plt.plot(future_dates, future_predictions, label="Predicted Prices (Next 7 Days)", linestyle="dashed", color="green")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.title("Stock Price: 7-Day Forecast")
        plt.legend()
        plt.show()

        return future_dates, future_predictions

    def compute_rmse(self):

        """Computes RMSE (Root Mean Squared Error) between actual and predicted prices."""
        rmse = np.sqrt(mean_squared_error(self.y_test_actual, self.predictions))
        print(f"RMSE: {rmse:.2f}")
        return rmse

    def print_predicted_vs_actual(self, n_days=10):
        """
        Prints the predicted vs actual prices for the last `n_days`.

        :param n_days: Number of days to print the comparison
        """
        print("\nPredicted vs Actual Prices:")
        print(f"{'Date':<12}{'Actual Price':<15}{'Predicted Price'}")
        print("-" * 40)

        test_dates = self.data.index[-len(self.y_test_actual):]
        for i in range(-n_days, 0):
            date = test_dates[i].strftime("%Y-%m-%d")
            actual = round(self.y_test_actual[i][0], 2)
            predicted = round(self.predictions[i][0], 2)
            print(f"{date:<12}{actual:<15}{predicted}")

    def forecast_future(self, future_days=7):
        """
        Predicts future stock prices.
        
        :param future_days: Number of days to forecast
        :return: Future dates and predicted prices
        """
        future_predictions = []
        last_sequence = self.scaled_data[-self.sequence_length:]

        for _ in range(future_days):
            next_prediction = self.model.predict(last_sequence.reshape(1, self.sequence_length, 1))
            future_predictions.append(next_prediction[0, 0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = next_prediction

        future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))
        future_dates = pd.date_range(self.data.index[-1], periods=future_days+1, freq='B')[1:]

        # Plot future predictions
        plt.figure(figsize=(12,6))
        plt.plot(self.data.index[-100:], self.data[-100:], label="Actual Prices", color="blue")
        plt.plot(future_dates, future_predictions, label="Predicted Prices (Next 7 Days)", linestyle="dashed", color="green")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.title("Stock Price: 7-Day Forecast")
        plt.legend()
        plt.show()

        return future_dates, future_predictions

    def save_model(self, filename="lstm_stock_model.h5"):
        """Saves the trained model."""
        self.model.save(filename)
        print(f"Model saved as {filename}")

    def load_model(self, filename="lstm_stock_model.h5"):
        """Loads a previously saved model."""
        from tensorflow.keras.models import load_model
        self.model = load_model(filename)
        print(f"Model loaded from {filename}")
