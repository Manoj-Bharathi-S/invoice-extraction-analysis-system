import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io

# Note: Ensure Tesseract-OCR and Poppler are installed on your system path.

def extract_text_from_image(image_bytes: bytes) -> str:
    """Runs OCR on a standard image file."""
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return text.strip()

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Converts a PDF to images, then runs OCR on all pages."""
    # Convert PDF bytes to a list of PIL Images
    pages = convert_from_bytes(pdf_bytes)
    
    full_text = ""
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page)
        full_text += f"\n--- Page {i + 1} ---\n{text}"
        
    return full_text.strip()

def process_document(file_bytes: bytes, filename: str) -> str:
    """Routes the file to the correct extraction method based on extension."""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif filename_lower.endswith(('.png', '.jpg', '.jpeg')):
        return extract_text_from_image(file_bytes)
    else:
        raise ValueError("Unsupported file format for OCR.")