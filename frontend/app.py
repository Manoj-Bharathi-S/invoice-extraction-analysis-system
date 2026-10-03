import streamlit as st
import requests
import pandas as pd
import sqlite3
from pathlib import Path
import os

# Use the Docker network API_URL if available, otherwise fallback to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/process-invoice")
DB_PATH = Path(__file__).resolve().parent.parent / "governance" / "audit_ledger.db"

# ... (keep the rest of the Streamlit code exactly the same) ...

st.set_page_config(page_title="Enterprise Document Intelligence", layout="wide")

st.title("🏛️ Enterprise Document Intelligence Hub")
st.markdown("Automated AI Ingestion, ML Cost Prediction, and Cryptographic Governance.")

# Create two distinct portals using tabs
tab1, tab2 = st.tabs(["📤 Vendor Portal (Upload)", "Audit"])

with tab1:
    st.header("Submit Vendor Invoice")
    uploaded_file = st.file_uploader("Upload an invoice (PDF, PNG, JPG)", type=['pdf', 'png', 'jpg', 'jpeg'])
    
    if st.button("Process Invoice") and uploaded_file is not None:
        with st.spinner("Processing through OCR and ML Pipeline..."):
            # Send file to FastAPI backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display Results beautifully
                st.success("Invoice Processed and Cryptographically Secured!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Expected Freight Cost", value=f"${result['expected_freight_cost']:,.2f}")
                with col2:
                    if result['risk_flag']:
                        st.error("🚨 HIGH RISK INVOICE: Flagged for Manual Review")
                    else:
                        st.info("✅ Standard Invoice: Auto-Approval Recommended")
                
                with st.expander("View Extracted OCR Text"):
                    st.text(result['extracted_text_preview'])
                    
            else:
                st.error(f"Error processing document: {response.text}")

with tab2:
    st.header("Audit Ledger")
    st.markdown("Tamper-evident record of all AI decisions and document uploads.")
    
    if st.button("Refresh Ledger"):
        try:
            # Replace local SQLite read with an API call
            LEDGER_URL = API_URL.replace("/process-invoice", "/audit-ledger")
            response = requests.get(LEDGER_URL)
            
            if response.status_code == 200 and response.json():
                df = pd.DataFrame(response.json())
                df['expected_freight_cost'] = df['expected_freight_cost'].apply(lambda x: f"${x:,.2f}")
                df['risk_flag'] = df['risk_flag'].apply(lambda x: "🚨 FLAG" if x else "✅ CLEAR")
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No audit logs found. Process an invoice first.")
        except Exception as e:
            st.error(f"Failed to fetch ledger: {e}")