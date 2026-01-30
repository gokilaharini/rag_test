from pydantic import BaseModel

# class QueryRequest(BaseModel):
#     query: str
#     top_k: int = 5

class IngestTextRequest(BaseModel):
    text: str
    metadata: dict
