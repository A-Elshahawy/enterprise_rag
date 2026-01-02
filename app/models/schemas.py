from typing import Optional
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


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: Optional[str] = None
