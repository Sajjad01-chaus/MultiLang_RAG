from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class IngestDocument(BaseModel):
    doc_path: str = Field(..., description="The path of the document to be ingested")
    id: Optional[str] = Field(None, description="ID of document")
    language: Optional[str] = Field(None, description="Language hint (e.g., 'en', 'hi', 'zh')")

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")
    top_k: Optional[int] = Field(5, description="Number of results to return")

class ProvenanceItem(BaseModel):
    doc_id: str
    page: int
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    provenance: List[Dict[str, Any]]
    latency_ms: Optional[float] = None
    subqueries: Optional[List[str]] = None

class IngestDocResponse(BaseModel):
    status: str
    doc_id: str
    chunks: int
    message: Optional[str] = None
    pages_processed: Optional[int] = None
    seconds: Optional[float] = None

class StatusResponse(BaseModel):
    ok: bool
    details: Optional[Dict[str, Any]] = None