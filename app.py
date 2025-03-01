import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import os
from utils.model_utils import get_available_models, load_model, fetch_live_stock_data

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

    # Predicted vs Actual Closing Price
    st.header("Predicted vs Actual Closing Price")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=dates, y=predicted, mode='lines', name='Predicted'))
    fig_price.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Actual'))
    fig_price.update_layout(title="Predicted vs Actual Closing Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_price, use_container_width=True)



    # Additional Financial Indicators (displayed as a table)
    st.header("Additional Financial Indicators")
    data = {
        "Indicator": ["RSI", "50-day Moving Average", "Bollinger Bands", "Volume"],
        "Value": ["45", "102.5", "Upper: 110, Lower: 95", "1.2M"]
    }
    df_indicators = pd.DataFrame(data)
    st.table(df_indicators)

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

elif page == "AI Trading Strategy":
    st.title("📈 AI Trading Strategy")

    # Sidebar for AI Trading Strategy options
    st.sidebar.header("Trading Strategy Options")
    available_models = get_available_models()
    selected_model = st.sidebar.selectbox("Choose a model:", available_models)
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

    # Fetch 60 days worth of historical stock prices using yfinance
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
selected_tickers = st.multiselect("Select tech stocks:", tech_stocks, default=["AAPL", "NVDA", "TSLA"], key="multiselect_tickers")

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
