from sentence_transformers import SentenceTransformer
import pdfplumber
from typing import Any
from pathlib import Path
import numpy as np


class DocumentEmbedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode([text])

