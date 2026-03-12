"""
Pydantic models for API request/response validation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for the /api/query endpoint."""
    question: str = Field(..., description="User's natural language question", min_length=1)
    top_k: int = Field(5, description="Number of final results after re-ranking", ge=1, le=20)


class CitationResponse(BaseModel):
    """A citation linking answer to source material."""
    source: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    chunk_text: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response body from the /api/query endpoint."""
    answer: str
    citations: List[CitationResponse]
    chunks_used: int
    confidence_score: float
    is_grounded: bool
    metrics: dict


class IngestRequest(BaseModel):
    """Request body for the /api/ingest endpoint."""
    path: str = Field(..., description="Path to file or directory to ingest")


class IngestResponse(BaseModel):
    """Response body from the /api/ingest endpoint."""
    documents_loaded: int
    chunks_created: int
    chunks_stored: int
    message: str


class MetricsResponse(BaseModel):
    """Response body from the /api/metrics endpoint."""
    total_requests: int
    successful_requests: int
    error_count: int
    failure_rate: float
    latency: dict
    cost: dict
    quality: dict
    reranker_scores: dict


class HealthResponse(BaseModel):
    """Response body from the /api/health endpoint."""
    status: str
    vector_store_count: int
    langfuse_enabled: bool


class FeedbackRequest(BaseModel):
    """Request body for the /api/feedback endpoint."""
    question: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="The answer that was generated")
    is_positive: bool = Field(..., description="True for thumbs-up, False for thumbs-down")
    comment: str = Field("", description="Optional text feedback")
    confidence_score: float = Field(0.0, description="Confidence score from the pipeline")
    chunks_used: int = Field(0, description="Number of chunks used")
    is_grounded: bool = Field(True, description="Whether the answer was grounded")
    citations: list = Field(default_factory=list, description="Citation data")


class FeedbackResponse(BaseModel):
    """Response body from the /api/feedback endpoint."""
    feedback_id: int
    message: str


class FeedbackSummaryResponse(BaseModel):
    """Response body from the /api/feedback/summary endpoint."""
    total_feedback: int
    positive: int
    negative: int
    satisfaction_rate: float
    recent_24h: dict
