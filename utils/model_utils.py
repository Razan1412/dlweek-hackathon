import joblib
import os
import requests
import random
import numpy as np
from dotenv import load_dotenv
import tensorflow as tf
import pandas_ta as ta

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
    """Fetch stock data from yfinance and calculate SMA and RSI."""
    try:
        stock = yf.Ticker(ticker)
        # Fetch historical data (adjust period as needed for indicator calculations)
        hist_data = stock.history(period="6mo") # Fetch 6 months to calculate 50 and 200 day SMAs, and RSI

        if hist_data.empty:
            print(f"⚠️ Error: No data found for ticker {ticker} from yfinance.", file=sys.stderr) # Print error to stderr
            return None

        # Calculate Simple Moving Averages (SMA) using pandas_ta
        sma_50 = ta.sma(hist_data['Close'], length=50).iloc[-1]  # 50-day SMA, last value
        sma_200 = ta.sma(hist_data['Close'], length=200).iloc[-1] # 200-day SMA, last value

        # Calculate Relative Strength Index (RSI) using pandas_ta
        rsi = ta.rsi(hist_data['Close'], length=14).iloc[-1] # 14-day RSI, last value

        # Get the latest close and open price (from the most recent day)
        latest_data = hist_data.iloc[-1]
        close_price = latest_data['Close']
        open_price = latest_data['Open']

        return [close_price, sma_50, sma_200, rsi, open_price]

    except Exception as e:
        print(f"⚠️ Error fetching stock data for {ticker} from yfinance: {e}", file=sys.stderr) # Print error to stderr
        return None
    except Exception as e:
        print(f"⚠️ Error fetching stock data: {e}")
        return None
