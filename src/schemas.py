from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, description="User question about Agentic AI")


class RetrievedChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunk] = []
    confidence_score: float


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    embedding_model: str
    llm_model: str
    total_chunks: Optional[int] = None
