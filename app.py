import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

    # Predicted vs Actual Clos
