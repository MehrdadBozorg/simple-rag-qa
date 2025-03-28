from fastapi import UploadFile, HTTPException
import pdfplumber
import openai
from pathlib import Path
from dotenv import load_dotenv
import os
import faiss
import numpy as np
from typing import Any


# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
FAISS_INDEX_PATH = "faiss_index.index"
DOCUMENTS_FILE_PATH = "documents.txt"
VECTOR_DIMENSION = 384  # Adjust based on the embedding model

# Initialize FAISS index
index = faiss.IndexFlatL2(VECTOR_DIMENSION)

class DocumentHandler:
    def __init__(self):
        self.documents = []

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extracts text from a PDF file
        """
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text

    def save_index(self):
        """
        Saves FAISS index to disk
        """
        faiss.write_index(index, FAISS_INDEX_PATH)
        
    def add_embedding(self, embedding: np.ndarray, text: str):
        """
        Adds an embedding and associated document text to the index
        """
        index.add(embedding)
        with open(DOCUMENTS_FILE_PATH, "a") as file:
            file.write(text + "\n")
        self.save_index()

    async def upload_document(self, file: UploadFile, embedding_model) -> dict:
        if file.content_type not in ["application/pdf", "text/plain"]:
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
        
        content = await file.read()
        
        # Extract text based on file type
        if file.content_type == "application/pdf":
            with open("temp.pdf", "wb") as temp_file:
                temp_file.write(content)
            text = self.extract_text_from_pdf(Path("temp.pdf"))
            os.remove("temp.pdf")
        else:
            text = content.decode("utf-8")
        
        # Create embedding for the document
        embedding = embedding_model.encode(text)
        
        # Store document and its embedding
        self.documents.append((text, embedding))
        
        return {"message": "Document uploaded successfully", "filename": file.filename}
    
    
class QueryHandler:
    def query_openai(self, context: str, user_query: str) -> str:
        """
        Function to query OpenAI's API to get an answer for a given prompt.
        """
        try:
             
            prompt = f"Document: {context}\n\nQuestion: {user_query}"
            message = {
                'role': 'user',
                'content': prompt
            }
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[message]
            )
            return str(response.choices[0].message['content']).strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        

    def find_most_similar_document(self, query_embedding: np.ndarray, k: int=3) -> Any:
        """
        Searches for the top-k most similar documents
        """
        _, indices = index.search(query_embedding, k)
        return indices