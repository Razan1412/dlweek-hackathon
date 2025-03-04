import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from utils.model_utils import get_available_models, load_model


# Set Streamlit page configuration
st.set_page_config(page_title="Financial & AI Trading Dashboard", layout="wide")

# Sidebar: Navigation
page = st.sidebar.selectbox("Select Page", ["Financial Dashboard", "Price Predictor Model", "Stock LSTM Model - Actual vs Predicted Visualizations"])

if page == "Financial Dashboard":
    st.title("Financial Dashboard")

    # Generate dummy data
    dates = pd.date_range(start="2025-03-01", periods=30)
    sentiment = np.random.uniform(low=-1, high=1, size=30)
    predicted = np.linspace(100, 110, 30) + np.random.normal(0, 1, 30)
    actual = np.linspace(100, 109, 30) + np.random.normal(0, 1, 30)
    supply_chain_score = np.random.uniform(low=0, high=100, size=30)


    
    # ---- New Section: Open-Close Price Visualization ---- #
    st.header("📊 Tech Stock Open-Close Prices (Today)")

    # Predefined tech stock tickers
    tech_stocks = ["AAPL", "AMD", "NVDA", "TSM", "GOOG", "MSFT", "AMZN", "META", "TSLA", "QCOM"]

    # Multi-select for choosing stocks
    selected_tickers = st.multiselect("Select tech stocks:", tech_stocks, default=["AAPL", "NVDA", "TSLA"])

    # Fetch and display stock open-close prices
    if selected_tickers:
        open_prices = []
        close_prices = []
        tickers_displayed = []

        for ticker in selected_tickers:
            stock_data = yf.Ticker(ticker).history(period="1d")
            if not stock_data.empty:
                open_prices.append(stock_data['Open'].iloc[0])
                close_prices.append(stock_data['Close'].iloc[0])
                tickers_displayed.append(ticker)

        # Create bar chart using Plotly
        if tickers_displayed:
            fig_open_close = go.Figure()

            # Open Price bars
            fig_open_close.add_trace(go.Bar(
                x=tickers_displayed,
                y=open_prices,
                name="Open Price",
                marker_color="royalblue"
            ))

            # Close Price bars
            fig_open_close.add_trace(go.Bar(
                x=tickers_displayed,
                y=close_prices,
                name="Close Price",
                marker_color="tomato"
            ))

            # Layout settings for theme consistency
            fig_open_close.update_layout(
                title="Open vs Close Prices (Today)",
                xaxis_title="Stock Ticker",
                yaxis_title="Price (USD)",
                barmode="group",  # Grouped bars
                template="plotly_dark",
                font=dict(size=14)
            )

            # Display the bar chart
            st.plotly_chart(fig_open_close, use_container_width=True)

if page == "Price Predictor Model":
    st.title("📈 Fine Tuned LSTMs")

    # Sidebar for AI Trading Strategy options
    st.sidebar.header("Indicate Model and Ticker")
    available_models = get_available_models()
    selected_model = st.sidebar.selectbox("Choose a model:", available_models)
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

    # Fetch 60 days worth of historical stock prices using yfinance
    stock_data = yf.Ticker(ticker)
    hist_data = stock_data.history(period="60d")
    st.write("### 60 Day Historical Stock Prices", hist_data)

    # Load the selected model (if available)
    model = load_model(selected_model) if selected_model else None
      # --- FIT THE SCALER HERE AFTER LOADING THE MODEL ---
    if model: # Only fit scaler if model loaded successfully
        print("Debug - Fitting scaler to hist_data['Close']...")
        try:
            model.scaler.fit(hist_data['Close'].values.reshape(-1, 1)) # Fit scaler to historical close prices
            print("Debug - Scaler fitted successfully.")
        except Exception as e:
            st.error(f"⚠️ Error fitting scaler: {e}")
            model = None # Disable model if scaler fitting fails
            print(f"Debug - Error fitting scaler: {e}")

    # Predict button
    if st.sidebar.button("Predict"):
        if not model:
            st.error("⚠️ No model loaded. Please check the model selection.")
        else:
            # live_data = hist_data.copy() # No need for copy anymore
            print("Debug - Type of hist_data:", type(hist_data))
            print("Debug - Content of hist_data:", hist_data.head())

            # --- Prepare input for predict_single_day correctly ---
            past_60_days_close = hist_data['Close'].tail(60) # Get last 60 'Close' prices as Series
            print("Debug - Type of past_60_days_close:", type(past_60_days_close))
            print("Debug - Content of past_60_days_close:", past_60_days_close.head())

            # Perform prediction using the loaded model (StockLSTM instance)
            predicted_date_str, predicted_price_val = model.predict_single_day(past_60_days_close) # Call predict_single_day
            print("Debug - Prediction Output (predict_single_day):", (predicted_date_str, predicted_price_val))

            # --- Extract predicted_price (it's already a float from predict_single_day) ---
            predicted_price = predicted_price_val # predicted_price_val is already the float value


            # Display prediction results
            st.subheader(f"Stock: {ticker}")
            current_price = hist_data['Close'].iloc[-1]
            st.metric(label="Current Price Right Now", value=f"${current_price:.2f}")
            st.metric(label="Predicted Price at Close Today", value=f"${predicted_price:.2f}") # Should now be a float, no TypeError

            

    # ---- New Section: Additional Financial Indicators ---- #
    st.header("📊 Additional Financial Indicators")

    # Function to compute financial indicators
    def compute_technical_indicators(stock_data):
        """Compute technical indicators from stock data."""
        stock_data = stock_data.copy()

        # Compute Moving Averages
        stock_data["10-day MA"] = stock_data["Close"].rolling(window=10).mean()
        stock_data["50-day MA"] = stock_data["Close"].rolling(window=50).mean()

        # Compute Volatility (10-day standard deviation of % change)
        stock_data["Volatility"] = stock_data["Close"].pct_change().rolling(window=10).std()

        # Compute Momentum (Price difference over 10 days)
        stock_data["Momentum"] = stock_data["Close"] - stock_data["Close"].shift(10)

        # Compute RSI
        delta = stock_data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        stock_data["RSI"] = 100 - (100 / (1 + rs))

        # Compute Trading Volume (use last available volume)
        stock_data["Trading Volume"] = stock_data["Volume"]

        # Drop NaN values from rolling calculations
        stock_data.dropna(inplace=True)

        return stock_data

    # Predefined tech stock tickers
    tech_stocks = ["AAPL", "AMD", "NVDA", "TSM", "GOOG", "MSFT", "AMZN", "META", "TSLA", "QCOM"]

    # ✅ FIX: Add a unique key to prevent duplicate ID errors
    selected_tickers = st.multiselect("Select tech stocks:", tech_stocks, default=["AAPL", "NVDA", "TSLA"], key="multiselect_indicators")

    # Fetch and display financial indicators
    if selected_tickers:
        for ticker in selected_tickers:
            stock_data = yf.download(ticker, period="6mo")

            # Check if stock data is available
            if stock_data.empty:
                st.warning(f"No data available for {ticker}. Skipping...")
                continue

            # Compute technical indicators
            stock_data = compute_technical_indicators(stock_data)

            # Extract latest values for each indicator
            latest_rsi = stock_data["RSI"].iloc[-1]
            latest_50_ma = stock_data["50-day MA"].iloc[-1]
            latest_10_ma = stock_data["10-day MA"].iloc[-1]
            latest_momentum = stock_data["Momentum"].iloc[-1]
            latest_volatility = stock_data["Volatility"].iloc[-1]
            latest_volume = stock_data["Trading Volume"].dropna().iloc[-1] if not stock_data["Trading Volume"].isna().all() else "N/A"

            # Create a DataFrame to display the indicators in a table
            df_indicators = pd.DataFrame({
                "Indicator": ["RSI", "50-day Moving Average", "10-day Moving Average", "Momentum", "Volatility", "Volume"],
                "Value": [
                    f"{latest_rsi:.2f}",
                    f"{latest_50_ma:.2f}",
                    f"{latest_10_ma:.2f}",
                    f"{latest_momentum:.2f}",
                    f"{latest_volatility:.4f}",
                    f"{int(latest_volume):,}" if latest_volume != "N/A" else "N/A"
                ]
            })

            # Display the table for each stock
            st.subheader(f"{ticker} - Key Indicators")
            st.table(df_indicators)
elif page == "Stock LSTM Model - Actual vs Predicted Visualizations":
            
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
