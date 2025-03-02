import joblib
import os
import requests
import random
import numpy as np
from dotenv import load_dotenv
import tensorflow as tf

# ✅ Load API Key from .env
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# ✅ Fix: Ensure `models/` is correctly referenced
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the project root
MODELS_DIR = os.path.join(BASE_DIR, "models")  # ✅ Now correctly points to `models/`

def get_available_models():
    """List available models in the models directory."""
    return [f for f in os.listdir(MODELS_DIR) if f.endswith(".h5")]

def load_model(model_name):
    """Load a Keras .h5 model from the models directory."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path) # Use tf.keras.models.load_model for .h5
            return model
        except Exception as e:
            print(f"⚠️ Error loading Keras .h5 model {model_name}: {e}")
            return None
    return None

def fetch_live_stock_data(ticker):
    """Fetch stock data from Alpha Vantage."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": "5min",
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if "Time Series (5min)" not in data:
            print(f"⚠️ Alpha Vantage API error: {data}")
            return None

        latest_timestamp = sorted(data["Time Series (5min)"].keys())[-1]
        latest_data = data["Time Series (5min)"][latest_timestamp]

        close_price = float(latest_data["4. close"])
        sma_50 = close_price  # Placeholder for now
        sma_200 = close_price  # Placeholder for now
        rsi = 50.0  # Placeholder for now

        return [close_price, sma_50, sma_200, rsi]

    except Exception as e:
        print(f"⚠️ Error fetching stock data: {e}")
        return None
