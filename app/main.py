from fastapi import FastAPI, UploadFile, File
from app.models import QueryRequest, QueryResponse
from app.database import DocumentHandler, QueryHandler
from app.embeddings import DocumentEmbedding

app = FastAPI()

# Initialize components
embedding_model = DocumentEmbedding()
document_handler = DocumentHandler()
query_handler = QueryHandler()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process document (PDF/TXT).
    """
    return await document_handler.upload_document(file, embedding_model)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Process a query using document context and LLAMA LLM model.
    """
    # Create embedding for the query
    query_embedding = embedding_model.encode(request.query)
    
    # Find most similar document from the uploaded ones
    index = query_handler.find_most_similar_document(query_embedding)
    
    context = document_handler.documents[index[0]]

    # Get the answer from OpenAI using the context
    answer = query_handler.query_openai(context, request.query)
    
    return QueryResponse(answer=answer, document=document_handler.documents[index[0]][-1])
