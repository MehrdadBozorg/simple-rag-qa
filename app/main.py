from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
from llama_cpp import Llama  
import numpy as np
from collections.abc import Iterator
from typing import Any
import pdfplumber

# Initialize FastAPI
app = FastAPI()

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index to store embeddings
d = 384  # Dimension of embeddings (depends on the model used)
index = faiss.IndexFlatL2(d)
documents_store = []

# Load the local LLM 
llm = Llama(model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

class IngestRequest(BaseModel):
    """
    Store documents
    """
    documents: List[str]

class QueryRequest(BaseModel):
    """
    Query data model
    """
    query: str
    

def extract_text_from_pdf(pdf_file) -> str:
    """
    Read a pdf file, extract its texts (from all pages) and return the text as string
    """
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text


@app.post("/ingest")
async def upload_document(file: UploadFile = File(...))-> dict[str, str]:
    """
    Upload a document in pdf or txt format, read its text, embed the text and store the text.
    
    Return the success message and the file name
    """
    global documents_store
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    content = await file.read()
    
    if file.content_type == "application/pdf":
        text = extract_text_from_pdf(file.file)
    else:
        text = content.decode("utf-8")

    # Embed and store the text
    embedding = embedding_model.encode([text])
    
     # Convert to NumPy array (2D) to make it compatible with the acceptable shape (n_documents, embedding_dim)
    embedding = np.array(embedding) 
    
    if embedding.ndim != 2:  # Check if the embeddings are 2D
        raise HTTPException(status_code=400, detail="Embeddings are not in the correct shape")
    
    # Add embeddings to FAISS index
    try:
        embedding = embedding.astype('float32')  # Acceptable float format for FAISS
        index.add(embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding embeddings to FAISS: {str(e)}")

    # Store the documents for future queries
    documents_store.append(text)
    
    return {"message": "Document uploaded successfully", "filename": file.filename}


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    """
    Embed the query and find the most relevant document to it and the best answer to the question
    """
    query_embedding = embedding_model.encode([request.query])
    D, I = index.search(query_embedding, k=1)
    if len(I[0]) == 0 or I[0][0] == -1:
        raise HTTPException(status_code=404, detail="No relevant document found")
    relevant_doc = documents_store[I[0][0]]

    response = llm(f"Document: {relevant_doc}\nQuery: {request.query}")
    if not isinstance(response, Iterator):
        return {"answer": response["choices"][0]["text"], "document": relevant_doc}

    else:
        return {}