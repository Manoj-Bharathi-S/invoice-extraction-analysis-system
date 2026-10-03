from thefuzz import fuzz
from thefuzz import process
import re

# Simulated database of known vendors (this would normally be queried from SQLite)
KNOWN_VENDORS = [
    "Acme Corp", 
    "Global Freight Ltd", 
    "TechLogistics Inc", 
    "FastTrack Shipping",
    "Apex Manufacturing"
]

def parse_text_to_features(raw_text: str) -> dict:
    """
    Parses raw OCR text to extract structured ML features using fuzzy matching and regex.
    """
    lines = raw_text.split('\n')
    
    extracted_data = {
        "vendor_name": "Unknown",
        "quantity": 1,         # Default fallback
        "invoice_value": 0.0   # Default fallback
    }
    
    # 1. Fuzzy Match the Vendor Name
    # Compares the raw text against our known vendor list
    best_match = process.extractOne(raw_text, KNOWN_VENDORS, scorer=fuzz.token_set_ratio)
    if best_match and best_match[1] > 75:  # 75% confidence threshold
        extracted_data["vendor_name"] = best_match[0]
        
    # 2. Extract Quantity and Value using Regex + Fuzzy logic
    for line in lines:
        line_lower = line.lower()
        
        # Find Invoice Value / Total Amount
        if fuzz.partial_ratio("total", line_lower) > 80 or fuzz.partial_ratio("invoice value", line_lower) > 80:
            # Extract numbers (handles decimals like 1250.50)
            numbers = re.findall(r'\d+\.\d+|\d+', line)
            if numbers:
                extracted_data["invoice_value"] = float(numbers[-1])
                
        # Find Quantity / Qty
        if fuzz.partial_ratio("quantity", line_lower) > 80 or fuzz.partial_ratio("qty", line_lower) > 80:
            # Extract integers
            numbers = re.findall(r'\d+', line)
            if numbers:
                extracted_data["quantity"] = int(numbers[-1])
                
    return extracted_data