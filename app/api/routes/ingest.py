import logging
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.document_processor import DocumentProcessor
from app.models.schemas import IngestResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Initialize processor
processor = DocumentProcessor()


@router.post(
    "",
    response_model=IngestResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
) -> IngestResponse:
    """
    Ingest a PDF document: extract text and create chunks.

    - Extracts text from all pages
    - Splits into overlapping chunks (1000 chars, 200 overlap)
    - Returns document ID and chunk count

    Note: Step 3 will add embedding and vector storage.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not file.content_type or "pdf" not in file.content_type.lower():
        raise HTTPException(
            status_code=400, detail=f"Invalid content type: {file.content_type}"
        )

    try:
        # Read file content
        pdf_bytes = await file.read()
        file_size = len(pdf_bytes)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")

        # Process PDF
        document_id, chunks, page_count = processor.process_pdf(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No text content found in PDF")

        logger.info(
            f"Ingested '{file.filename}': id={document_id}, "
            f"pages={page_count}, chunks={len(chunks)}"
        )

        # ? Step 3 will add embedding + Qdrant storage here

        return IngestResponse(
            document_id=document_id,
            filename=file.filename,
            chunks=len(chunks),
            pages=page_count,
            message=f"Document processed: {len(chunks)} chunks from {page_count} pages",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to process {file.filename}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process document: {str(e)}"
        )


@router.get("/status")
async def ingestion_status() -> dict:
    """Get ingestion service status."""
    return {
        "status": "ready",
        "chunk_size": processor.chunk_size,
        "chunk_overlap": processor.chunk_overlap,
        "supported_formats": ["pdf"],
    }
