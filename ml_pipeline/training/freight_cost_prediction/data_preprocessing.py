import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Added scaler

def load_vendor_invoice_data(db_path:str):
    connection = sqlite3.connect(db_path)
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, connection)
    connection.close()
    return df

def prepare_features(df: pd.DataFrame):
    """
    Ensure the column names exactly match the API's input.
    Assuming your database has 'Quantity' and 'Dollars' columns.
    """
    # Create a copy to rename columns to match what predict.py sends exactly
    features = df[["Quantity", "Dollars"]].copy()
    features.rename(columns={"Quantity": "quantity", "Dollars": "invoice_value"}, inplace=True)
    
    X = features
    y = df["Freight"]
    
    # Fit the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame so Scikit-Learn retains feature names
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled_df, y, scaler