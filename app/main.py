from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import os
from typing import Any
from llama_cpp import Llama 

from app.database import add_embedding, search_similar
from app.embeddings import extract_text_from_pdf, get_text_embedding
from app.models import QueryRequest


app = FastAPI()

# Load the local LLaMA model to answer the question
llm = Llama(model_path="models/llama-7b.ggmlv3.q4_0.bin")


@app.post("/ingest")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str]:
    """
    Upload a document in pdf or txt format, read its text, embed the text and store the text.
    
    Return the success message and the file name
    """
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    content = await file.read()
    if file.content_type == "application/pdf":
        with open("temp.pdf", "wb") as temp_file:
            temp_file.write(content)
        text = extract_text_from_pdf("temp.pdf")
        os.remove("temp.pdf")
    else:
        text = content.decode("utf-8")

    embedding = get_text_embedding(text)
    embedding_np = np.array(embedding, dtype=np.float32)
    
    add_embedding(embedding_np, text)

    return {"message": "Document uploaded successfully", "filename": file.filename}


@app.post("/query")
async def query(request: QueryRequest) -> dict[str, Any]:
    """
    Embed the query and find the most relevant document to it and the best answer to the question
    """
    query_embedding = get_text_embedding(request.query)
    query_embedding_np = np.array(query_embedding, dtype=np.float32)
    indices = search_similar(query_embedding_np, k=3)

    with open("documents.txt", "r") as file:
        docs = file.readlines()

    relevant_docs = [docs[i].strip() for i in indices[0] if i < len(docs)]
    
    # Prepare context for LLM
    context_text = "\n".join(relevant_docs)
    prompt = f"Using the following context:\n{context_text}\nAnswer the question: {request.query}"

    # Generate answer using LLaMA
    response = llm(prompt, max_tokens=200)
    answer = response["choices"][0]["text"].strip()

    return {
        "query": request.query,
        "relevant_documents": relevant_docs,
        "answer": answer
    }