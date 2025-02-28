import streamlit as st
import os
from utils.model_utils import get_available_models, load_model, fetch_live_stock_data

# ✅ Set Streamlit page title
st.set_page_config(page_title="AI Trading Strategy - Domain Experts", layout="wide")

# ✅ Sidebar: Model selection
st.sidebar.header("Select Model")
available_models = get_available_models()
selected_model = st.sidebar.selectbox("Choose a model:", available_models)

# ✅ Sidebar: Stock selection
st.sidebar.header("Select Stock")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

# ✅ Load model
model = load_model(selected_model) if selected_model else None

# ✅ Main Title
st.title("📈 AI Trading Strategy")

# ✅ Predict button
if st.sidebar.button("Predict"):
    if not model:
        st.error("⚠️ No model loaded. Please check the model selection.")
    else:
        live_data = fetch_live_stock_data(ticker)

        if not live_data:
            st.error(f"⚠️ Unable to fetch live data for {ticker}.")
        else:
            prediction = model.predict([live_data])
            predicted_price, action = prediction[0]

            # ✅ Display results
            st.subheader(f"Stock: {ticker}")
            st.metric(label="Current Price", value=f"${live_data[0]:.2f}")
            st.metric(label="Predicted Price", value=f"${predicted_price:.2f}")
            st.success(f"**Recommendation: {action}**")

            # ✅ Show stock data table
            st.write("### Stock Data")
            st.table({
                "Metric": ["Current Price", "SMA-50", "SMA-200", "RSI"],
                "Value": [f"${live_data[0]:.2f}", f"${live_data[1]:.2f}", f"${live_data[2]:.2f}", f"{live_data[3]:.2f}"]
            })
