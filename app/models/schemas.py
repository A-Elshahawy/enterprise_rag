from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for an ingested document."""

    filename: str
    page_count: int
    chunk_count: int
    file_size: int  # bytes


class IngestResponse(BaseModel):
    """Response after document ingestion."""

    document_id: str
    filename: str
    chunks: int
    pages: int
    message: str = "Document ingested successfully"


class ChunkSchema(BaseModel):
    """Schema for a text chunk."""

    chunk_id: str
    document_id: str
    text: str
    page_number: int
    chunk_index: int
    metadata: dict = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    document_id: Optional[str] = None


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page_number: int
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem] = Field(default_factory=list)
    total: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: Optional[str] = None
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)


class SourceCitation(BaseModel):
    source_id: int
    document_id: str
    page_number: int
    text_preview: str
    relevance_score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    model: str


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: Optional[str] = None
