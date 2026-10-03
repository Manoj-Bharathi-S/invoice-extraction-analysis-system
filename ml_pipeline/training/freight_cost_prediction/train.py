import joblib
from pathlib import Path
import logging

from .data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from .model_evaluation import train_linear_regression, train_decision_tree, train_random_forest, evaluate_model

# ... (keep your existing logger configuration) ...
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing vendor invoice training pipeline...")

    # Anchor everything to the script location
    base_dir = Path(__file__).resolve().parents[2]
    db_path = str(base_dir / "data" / "inventory.db")
    
    # FIX: Dynamically anchor the model directory
    model_dir = base_dir / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading vendor invoice data from database: %s", db_path)
    df = load_vendor_invoice_data(db_path)

    logger.info("Extracting, scaling features, and targets from dataset...")
    # FIX: Unpack the scaler returned from preprocessing
    X, y, scaler = prepare_features(df)
    
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    logger.info("Training Linear Regression model...")
    lr_model = train_linear_regression(X_train, y_train)
    
    logger.info("Training Decision Tree model...")
    dt_model = train_decision_tree(X_train, y_train)
    
    logger.info("Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)

    logger.info("Evaluating models on test data...")
    results = [
        evaluate_model(lr_model, X_test, y_test, "Linear Regression"),
        evaluate_model(dt_model, X_test, y_test, "Decision Tree"),
        evaluate_model(rf_model, X_test, y_test, "Random Forest")
    ]

    best_model_data = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_data["model_name"]
    logger.info("Best model selected: %s (MAE: %s)", best_model_name, best_model_data["mae"])
    
    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }[best_model_name]
    
    # FIX: Save both the best model AND the scaler
    model_path = model_dir / "predict_freight_model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    logger.info("Artifacts saved to: %s", model_dir)

if __name__ == "__main__":
    main()