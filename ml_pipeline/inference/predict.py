import joblib
import pandas as pd
from pathlib import Path

# Dynamically resolve the path to the saved_models directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "saved_models"

# Define model and scaler paths
FREIGHT_MODEL_PATH = MODEL_DIR / "predict_freight_model.pkl"
FLAG_MODEL_PATH = MODEL_DIR / "predict_flag_invoice.pkl"
FREIGHT_SCALER_PATH = MODEL_DIR / "scaler.pkl"
FLAG_SCALER_PATH = MODEL_DIR / "flag_scaler.pkl"

def load_models():
    """Helper to load models only if they exist."""
    try:
        freight_model = joblib.load(FREIGHT_MODEL_PATH)
        flag_model = joblib.load(FLAG_MODEL_PATH)
        freight_scaler = joblib.load(FREIGHT_SCALER_PATH)
        flag_scaler = joblib.load(FLAG_SCALER_PATH)
        return freight_model, flag_model, freight_scaler, flag_scaler
    except FileNotFoundError as e:
        print(f"Warning: ML models not found. Running in fallback mode. Error: {e}")
        return None, None, None, None

def run_inference(extracted_features: dict) -> dict:
    """
    Takes extracted invoice features, runs them through the ML models,
    and returns predictions.
    """
    quantity = extracted_features.get("quantity", 1)
    invoice_value = extracted_features.get("invoice_value", 0.0)

    # Load all four ML artifacts
    freight_model, flag_model, freight_scaler, flag_scaler = load_models()

    if not freight_model or not flag_model or not freight_scaler or not flag_scaler:
        # Fallback logic if models aren't compiled yet
        return {
            "expected_freight_cost": round(invoice_value * 0.05, 2), 
            "risk_flag": invoice_value > 10000 
        }

    # Format input for Scikit-Learn as a DataFrame to retain feature names
    input_data = pd.DataFrame([{
        "quantity": quantity,
        "invoice_value": invoice_value
    }])
    
    # 1. Predict Freight Cost (Regression)
    freight_input_scaled = freight_scaler.transform(input_data)
    predicted_freight = freight_model.predict(freight_input_scaled)[0]

    # 2. Predict Risk Flag (Classification)
    flag_input_scaled = flag_scaler.transform(input_data)
    risk_prediction = flag_model.predict(flag_input_scaled)[0]
    
    return {
        "expected_freight_cost": round(float(predicted_freight), 2),
        "risk_flag": bool(risk_prediction == 1)
    }