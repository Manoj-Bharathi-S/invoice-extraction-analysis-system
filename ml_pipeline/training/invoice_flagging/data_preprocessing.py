import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

def load_invoice_data(db_path: str):
    logger.info("Connecting to database at: %s", db_path)
    connection = sqlite3.connect(db_path)
    # The same SQL query you wrote
    query = """
            WITH purchase_agg AS(
            SELECT
                p.PONumber,
                COUNT(DISTINCT p.Brand) as total_brands,
                SUM(p.Quantity) as total_item_quantity,
                SUM(p.Dollars) as total_item_dollars,
                AVG(julianday(ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
            FROM purchases p
            GROUP BY p.PONumber
            )
            SELECT
            vi.Quantity as invoice_quantity,
            vi.Dollars as invoice_dollars,
            vi.Freight,
            (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
            (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
            pa.total_brands,
            pa.total_item_quantity,
            pa.total_item_dollars,
            pa.avg_receiving_delay
            FROM vendor_invoice as vi
            LEFT JOIN purchase_agg pa
                ON vi.PONumber = pa.PONumber
            """
    df = pd.read_sql_query(query, connection)
    connection.close()
    
    # FIX: Rename columns to perfectly match the API's OCR output
    df.rename(columns={"invoice_quantity": "quantity", "invoice_dollars": "invoice_value"}, inplace=True)
    return df

def create_invoice_risk_label(row):
    if (abs(row["invoice_value"] - row["total_item_dollars"]) > 5):
        return 1
    if row["avg_receiving_delay"] > 10:
        return 2
    return 0

def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df

def split_and_scale_data(df, features, target, test_size=0.2, random_state=42):
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame so Scikit-Learn retains feature names for the API
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler