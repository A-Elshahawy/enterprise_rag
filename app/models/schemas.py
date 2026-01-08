from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for an ingested document."""

    filename: str
    page_count: int
    chunk_count: int
    file_size: int  # bytes


class DocumentListItem(BaseModel):
    """Single document in a list."""

    document_id: str
    filename: str


class DocumentListResponse(BaseModel):
    """Response for listing documents."""

    documents: list[DocumentListItem] = Field(default_factory=list)
    total: int = 0


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
    """Request for semantic search."""

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    document_id: Optional[str] = Field(
        default=None, description="Single document filter (deprecated, use document_ids)"
    )
    document_ids: Optional[list[str]] = Field(default=None, description="Filter by multiple document IDs")


class SearchResultItem(BaseModel):
    """Single search result item with position info for highlighting."""

    chunk_id: str
    document_id: str
    text: str
    page_number: int
    score: float
    char_start: int = Field(default=0, description="Start position in page text")
    char_end: int = Field(default=0, description="End position in page text")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response for search queries."""

    query: str
    results: list[SearchResultItem] = Field(default_factory=list)
    total: int


class AskRequest(BaseModel):
    """Request for RAG question answering."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: Optional[str] = Field(
        default=None, description="Single document filter (deprecated, use document_ids)"
    )
    document_ids: Optional[list[str]] = Field(default=None, description="Filter by multiple document IDs")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)


class SourceCitation(BaseModel):
    """Citation source for an answer with position info for highlighting."""

    source_id: int
    document_id: str
    page_number: int
    text_preview: str
    relevance_score: float
    char_start: int = Field(default=0, description="Start position in page text")
    char_end: int = Field(default=0, description="End position in page text")


class AskResponse(BaseModel):
    """Response for RAG questions."""

    question: str
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    model: str


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: Optional[str] = None
