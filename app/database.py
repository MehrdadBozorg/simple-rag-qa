import faiss
import numpy as np
import os
from typing import Any

FAISS_INDEX_PATH = "faiss_index.index"
DOCUMENTS_FILE_PATH = "documents.txt"
VECTOR_DIMENSION = 384  # Adjust based on the embedding model

# Initialize FAISS index
index = faiss.IndexFlatL2(VECTOR_DIMENSION)

# Load index if it exists
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    

def save_index():
    """
    Saves FAISS index to disk
    """
    faiss.write_index(index, FAISS_INDEX_PATH)
    

def add_embedding(embedding: np.ndarray, text: str):
    """
    Adds an embedding and associated document text to the index
    """
    index.add(embedding)
    with open(DOCUMENTS_FILE_PATH, "a") as file:
        file.write(text + "\n")
    save_index()
    

def search_similar(query_embedding: np.ndarray, k: int=3) -> Any:
    """
    Searches for the top-k most similar documents
    """
    _, indices = index.search(query_embedding, k)
    return indices
