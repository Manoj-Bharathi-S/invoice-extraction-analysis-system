import sqlite3
import hashlib
import json
from datetime import datetime
from pathlib import Path

# Dynamically resolve DB path
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "audit_ledger.db"

def init_db():
    """Initializes the SQLite database with the audit ledger table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            file_hash TEXT,
            expected_freight_cost REAL,
            risk_flag BOOLEAN,
            previous_hash TEXT,
            current_hash TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_previous_hash() -> str:
    """Retrieves the hash of the last transaction to maintain the chain."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT current_hash FROM audit_ledger ORDER BY id DESC LIMIT 1')
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "0" * 64  # Genesis hash

def log_transaction(filename: str, file_bytes: bytes, ml_results: dict) -> str:
    """
    Cryptographically hashes the transaction and logs it to the SQL database.
    Returns the unique transaction hash.
    """
    # 1. Hash the original file to prove exactly what was uploaded
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    
    # 2. Retrieve previous chain state
    previous_hash = get_previous_hash()
    
    # 3. Create the block payload
    timestamp = datetime.utcnow().isoformat()
    payload = {
        "timestamp": timestamp,
        "filename": filename,
        "file_hash": file_hash,
        "freight_cost": ml_results["expected_freight_cost"],
        "risk_flag": ml_results["risk_flag"],
        "previous_hash": previous_hash
    }
    
    # 4. Hash the payload to create the immutable transaction signature
    payload_string = json.dumps(payload, sort_keys=True).encode('utf-8')
    current_hash = hashlib.sha256(payload_string).hexdigest()
    
    # 5. Save to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO audit_ledger 
        (timestamp, filename, file_hash, expected_freight_cost, risk_flag, previous_hash, current_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, filename, file_hash, ml_results["expected_freight_cost"], ml_results["risk_flag"], previous_hash, current_hash))
    conn.commit()
    conn.close()
    
    return current_hash

# Initialize the database file on module import
init_db()