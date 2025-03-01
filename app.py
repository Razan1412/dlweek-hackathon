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
