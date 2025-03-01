import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from utils.model_utils import get_available_models, load_model, fetch_live_stock_data
import yfinance as yf  # New import for yfinance
from stock_lstm import StockLSTM

# Set Streamlit page configuration
st.set_page_config(page_title="Financial & AI Trading Dashboard", layout="wide")

# Sidebar: Navigation
page = st.sidebar.selectbox("Select Page", ["Financial Dashboard", "AI Trading Strategy"])

if page == "Financial Dashboard":
    st.title("Comprehensive Financial Dashboard")

    # Generate dummy data
    dates = pd.date_range(start="2025-03-01", periods=30)
    sentiment = np.random.uniform(low=-1, high=1, size=30)
    predicted = np.linspace(100, 110, 30) + np.random.normal(0, 1, 30)
    actual = np.linspace(100, 109, 30) + np.random.normal(0, 1, 30)
    supply_chain_score = np.random.uniform(low=0, high=100, size=30)

    # Market Sentiment Analysis
    st.header("Market Sentiment Analysis")
    fig_sentiment = go.Figure()
    fig_sentiment.add_trace(go.Scatter(x=dates, y=sentiment, mode='lines+markers', name='Sentiment Score'))
    fig_sentiment.update_layout(title="Daily Market Sentiment", xaxis_title="Date", yaxis_title="Sentiment Score")
    st.plotly_chart(fig_sentiment, use_container_width=True)

# Import the StockLSTM class from your local file
from stock_lstm import StockLSTM

# Set the page configuration (optional)
st.set_page_config(page_title="Stock LSTM Forecaster", layout="wide")

st.title("📈 Stock LSTM Forecaster")

# 1. User inputs a ticker symbol
ticker = st.text_input("Enter a stock ticker (e.g. AAPL):", "AAPL")

# 2. Button to fetch data, train the model, and forecast the next 30 days
if st.button("Train & Forecast Next 30 Days"):
    # Fetch 60 days of historical data from yfinance
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="60d")  # last 60 days

    # Check if we have enough data
    if hist_data.empty or len(hist_data) < 60:
        st.error("Not enough data to train the model (need at least 60 days).")
    else:
        st.write("### Last 60 Days of Historical Data", hist_data)

        # Extract closing prices as a pandas Series
        close_data = hist_data['Close']

        # 3. Initialize the LSTM model with sequence_length=60
        lstm_model = StockLSTM(data=close_data, sequence_length=60)

        # 4. Train the model (adjust epochs/batch_size as needed)
        lstm_model.train_model(epochs=10, batch_size=32)

        # 5. Forecast the next 30 days
        future_days = 30
        future_dates, future_preds = lstm_model.forecast_future(future_days=future_days)

        # 6. Display forecast results in a table
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": future_preds.flatten()
        })
        forecast_df.set_index("Date", inplace=True)
        st.write(f"## Next {future_days} Days Forecast")
        st.table(forecast_df)

        # 7. Plot the actual recent prices vs. forecast
        fig = go.Figure()

        # Plot the last 30 days of actual data for context
        last_30_data = close_data[-30:]
        fig.add_trace(
            go.Scatter(
                x=last_30_data.index,
                y=last_30_data.values,
                mode='lines',
                name='Actual (Last 30 Days)'
            )
        )

        # Plot the forecast
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_preds.flatten(),
                mode='lines',
                name=f'Forecast (Next {future_days} Days)'
            )
        )

        fig.update_layout(
            title=f"{ticker} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Supply Chain Analysis
    st.header("Supply Chain Analysis")
    fig_supply = go.Figure()
    fig_supply.add_trace(go.Bar(x=dates, y=supply_chain_score, name='Supply Chain Score'))
    fig_supply.update_layout(title="Supply Chain Health Indicator", xaxis_title="Date", yaxis_title="Score")
    st.plotly_chart(fig_supply, use_container_width=True)

    # Additional Financial Indicators (displayed as a table)
    st.header("Additional Financial Indicators")
    data = {
        "Indicator": ["RSI", "50-day Moving Average", "Bollinger Bands", "Volume"],
        "Value": ["45", "102.5", "Upper: 110, Lower: 95", "1.2M"]
    }
    df_indicators = pd.DataFrame(data)
    st.table(df_indicators)

elif page == "AI Trading Strategy":
    st.title("📈 AI Trading Strategy")

    # Sidebar for AI Trading Strategy options
    st.sidebar.header("Trading Strategy Options")
    available_models = get_available_models()
    selected_model = st.sidebar.selectbox("Choose a model:", available_models)
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

    # New addition: Fetch 60 days worth of historical stock prices using yfinance
    stock_data = yf.Ticker(ticker)
    hist_data = stock_data.history(period="60d")
    st.write("### 60 Day Historical Stock Prices", hist_data)

    # Load the selected model (if available)
    model = load_model(selected_model) if selected_model else None

    # Predict button
    if st.sidebar.button("Predict"):
        if not model:
            st.error("⚠️ No model loaded. Please check the model selection.")
        else:
            live_data = fetch_live_stock_data(ticker)
            if not live_data:
                st.error(f"⚠️ Unable to fetch live data for {ticker}.")
            else:
                # Perform prediction using the loaded model
                prediction = model.predict([live_data])
                predicted_price, action = prediction[0]

                # Display prediction results
                st.subheader(f"Stock: {ticker}")
                st.metric(label="Current Price", value=f"${live_data[0]:.2f}")
                st.metric(label="Predicted Price", value=f"${predicted_price:.2f}")
                st.success(f"**Recommendation: {action}**")

                # Display additional stock data in a table
                st.write("### Stock Data")
                st.table({
                    "Metric": ["Current Price", "SMA-50", "SMA-200", "RSI"],
                    "Value": [f"${live_data[0]:.2f}", f"${live_data[1]:.2f}", f"${live_data[2]:.2f}", f"{live_data[3]:.2f}"]
                })
