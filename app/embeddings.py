from sentence_transformers import SentenceTransformer
import pdfplumber
from typing import Any
from pathlib import Path

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts text from a PDF file
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text


def get_text_embedding(text: str) -> Any:
    """
    Generates embeddings for the given text
    """
    return embedding_model.encode([text])