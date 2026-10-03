from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
from extraction.ocr_engine import process_document
from extraction.fuzzy_matcher import parse_text_to_features
# We will import your ML inference and OCR functions here later
# from ml_pipeline.inference.predict import run_inference
# from extraction.ocr_engine import process_document
# Add this import at the top of api/main.py
from ml_pipeline.inference.predict import run_inference
from governance.audit_logger import log_transaction
# ... (keep existing setup and extraction imports) ...



app = FastAPI(
    title="Enterprise Document Intelligence API",
    description="OCR and ML pipeline for vendor invoice processing",
    version="1.0.0"
)

class InvoiceResponse(BaseModel):
    filename: str
    extracted_text_preview: str
    expected_freight_cost: float
    risk_flag: bool

@app.get("/")
def health_check():
    return {"status": "application is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "components": ["api", "ml_pipeline"]}



@app.post("/api/v1/process-invoice", response_model=InvoiceResponse)
async def process_invoice(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload PDF or Image.")

    try:
        file_bytes = await file.read()
        
        # Phase 1: Ingestion & Extraction
        raw_text = process_document(file_bytes, file.filename)
        parsed_features = parse_text_to_features(raw_text)
        
        # Phase 2: Machine Learning Inference
        ml_results = run_inference(parsed_features)
        
        # Phase 3: Governance & Audit Logging
        # Secure the transaction and generate a receipt hash
        transaction_hash = log_transaction(file.filename, file_bytes, ml_results)
        
        # Final API Response
        return {
            "filename": file.filename,
            "extracted_text_preview": raw_text[:100].replace('\n', ' ') + "...",
            "expected_freight_cost": ml_results["expected_freight_cost"],
            "risk_flag": ml_results["risk_flag"],
            # You can also add transaction_hash to the InvoiceResponse schema later!
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






import sys
from pathlib import Path
import uvicorn

# Ensure the root of 'enterprise-doc-intelligence' is in Python's search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root)]
    )