from pydantic import BaseModel

class QueryRequest(BaseModel):
    """
    Query data model
    """
    query: str