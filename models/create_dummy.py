import joblib
import os
import random

# ✅ Define DummyTradingModel
class DummyTradingModel:
    """A dummy model that generates random BUY/SELL signals."""
    def predict(self, X):
        return [(random.uniform(50, 500), random.choice(["BUY", "SELL"])) for _ in X]

# ✅ Fix: Ensure `models/` is created at the correct level
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to `models/` folder
MODEL_DIR = os.path.join(BASE_DIR)  # ✅ No extra nesting

# ✅ Make sure the `models/` directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Save the dummy model inside `models/`
model_path = os.path.join(MODEL_DIR, "dummy2.pkl")
dummy_model = DummyTradingModel()
joblib.dump(dummy_model, model_path)

print(f"✅ Dummy model saved successfully at {model_path}")
