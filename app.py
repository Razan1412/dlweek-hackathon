import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Set Streamlit page configuration
st.set_page_config(page_title="Stock LSTM Model - Actual vs Predicted", layout="wide")

# ----------------- STOCK OPTIONS -----------------
stock_options = ["AAPL", "AMD", "NVDA", "TSM", "GOOG", "MSFT", "AMZN", "META", "TSLA", "QCOM"]

# ----------------- HELPER FUNCTIONS -----------------

def create_sequences(data, sequence_length=60):
    """
    Converts a 1D array of stock prices into sequences for LSTM.
    Ensures data is properly reshaped to (num_samples, sequence_length, 1).
    """
    X, y = [], []
    
    # 🚨 Check if there is enough data
    if len(data) <= sequence_length:
        return np.array([]), np.array([])  # Return empty arrays

    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])

    # Convert lists to NumPy arrays
    if len(X) == 0:
        return np.array([]), np.array([])  # Return empty arrays if no sequences are created

    X, y = np.array(X), np.array(y)

    # 🚨 Check again before reshaping to avoid IndexError
    if X.shape[0] == 0:
        return np.array([]), np.array([])

    # Reshape to 3D shape for LSTM (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def build_lstm_model(sequence_length=60):
    """
    Builds an LSTM model with proper input shape.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ----------------- STREAMLIT UI -----------------

st.title("📈 Stock LSTM Model - Actual vs Predicted")

# Dropdown for selecting a stock
ticker = st.selectbox("Select a Stock:", stock_options)

if st.button("Train & Predict"):
    # Fetch 3 years (1095 days) of historical data from yfinance
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="1095d")  # Fetch 3 years of data

    if hist_data.empty or len(hist_data) < 60:
        st.error(f"⚠️ Not enough data to train the model for {ticker}.")
    else:
        st.write(f"### Last 3 Years of Historical Data for {ticker}", hist_data)

        # Extract closing prices and normalize data
        close_data = hist_data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_data_scaled = scaler.fit_transform(close_data)

        # Create sequences
        sequence_length = 60
        X, y = create_sequences(close_data_scaled, sequence_length)

        # 🚨 Prevent training if sequences could not be created
        if X.shape[0] == 0:
            st.error(f"⚠️ Not enough data to create training sequences for {ticker}. Try another stock.")
        else:
            # Split into training (80%) and test (20%) sets
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Train the model
            model = build_lstm_model(sequence_length)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

            # Predict on test data
            y_pred_scaled = model.predict(X_test)

            # Convert predicted values back to original scale
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_original = scaler.inverse_transform(y_pred_scaled)

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

            # Create a date index for test set
            test_dates = hist_data.index[train_size + sequence_length:]

            # Plot actual vs predicted values
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_dates, y=y_test_original.flatten(), mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=test_dates, y=y_pred_original.flatten(), mode='lines', name='Predicted'))

            # Add RMSE inside the chart as a legend
            fig.update_layout(
                title=f"{ticker} - Actual vs Predicted",
                xaxis_title="Date",
                yaxis_title="Stock Price (USD)",
                legend_title=f"RMSE: {rmse:.4f}",
                legend=dict(
                    x=0, y=1.1,  # Positioning above the chart
                    orientation="h"
                )
            )

            st.plotly_chart(fig, use_container_width=True)
