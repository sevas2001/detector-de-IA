import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a uploaded PDF file (Streamlit UploadedFile) or file path.
    Returns the full text as a string.
    """
    try:
        # Check if it's a file path or a stream
        if isinstance(pdf_file, str):
            doc = fitz.open(pdf_file)
        else:
            # Assume it's a bytes stream (like from Streamlit)
            # We need to read the bytes to open it with fitz
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
            
        doc.close()
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

if __name__ == "__main__":
    # Test block
    print("PDF Extractor module ready.")
