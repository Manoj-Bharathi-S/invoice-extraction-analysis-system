import joblib
from pathlib import Path
import logging

from .data_preprocessing import load_invoice_data, apply_labels, split_and_scale_data
from .model_evaluation import train_random_forest, evaluate_classifier

logger = logging.getLogger(__name__)

# FIX: Align features with the OCR Extractor
FEATURES = ["quantity", "invoice_value"]
TARGET = "flag_invoice"

def main():
    logger.info("Initializing invoice flagging pipeline...")
    
    # FIX: Dynamic pathing
    base_dir = Path(__file__).resolve().parents[2]
    db_path = str(base_dir / "data" / "inventory.db")
    
    model_dir = base_dir / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_invoice_data(db_path)
    df = apply_labels(df)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(df, FEATURES, TARGET)

    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # FIX: Save the GridSearch's best estimator and a dedicated Flag Scaler
    model_path = model_dir / "predict_flag_invoice.pkl"
    scaler_path = model_dir / "flag_scaler.pkl"
    
    joblib.dump(grid_search.best_estimator_, model_path)
    joblib.dump(scaler, scaler_path)
    
    logger.info("Pipeline executed successfully!")

if __name__ == "__main__":
    main()