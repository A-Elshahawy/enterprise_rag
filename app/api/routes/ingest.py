"""Document ingestion endpoints."""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.document_processor import DocumentProcessor
from app.core.embeddings import get_embedding_service
from app.core.vector_store import get_vector_store
from app.models.schemas import ErrorResponse, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Initialize services
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
    Ingest a PDF document: extract, chunk, embed, and store.

    Pipeline:
    1. Extract text from PDF pages
    2. Split into overlapping chunks
    3. Generate embeddings (Sentence Transformers)
    4. Store in Qdrant vector database
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not file.content_type or "pdf" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail=f"Invalid content type: {file.content_type}")

    try:
        # Read file
        pdf_bytes = await file.read()
        file_size = len(pdf_bytes)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if file_size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")

        # Step 2: Process PDF → chunks
        document_id, chunks, page_count = processor.process_pdf(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No text content found in PDF")

        # Step 3: Generate embeddings
        embedding_service = get_embedding_service()
        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_service.embed_texts(texts)

        logger.info(f"Generated {len(embeddings)} embeddings for '{file.filename}'")

        # Step 3: Store in Qdrant
        vector_store = get_vector_store()
        stored_count = vector_store.upsert_chunks(chunks, embeddings)

        logger.info(f"Ingested '{file.filename}': id={document_id}, " f"pages={page_count}, chunks={stored_count}")

        return IngestResponse(
            document_id=document_id,
            filename=file.filename,
            chunks=stored_count,
            pages=page_count,
            message=f"Document ingested: {stored_count} chunks embedded and stored",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to process {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.delete("/{document_id}")
async def delete_document(document_id: str) -> dict:
    """Delete a document and all its chunks from the vector store."""
    try:
        vector_store = get_vector_store()
        vector_store.delete_document(document_id)
        return {"message": f"Document '{document_id}' deleted"}
    except Exception as e:
        logger.exception(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def ingestion_status() -> dict:
    """Get ingestion service status and collection info."""
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()

    return {
        "status": "ready",
        "chunk_size": processor.chunk_size,
        "chunk_overlap": processor.chunk_overlap,
        "embedding_model": embedding_service.model_name,
        "embedding_dimension": embedding_service.dimension,
        "collection": vector_store.get_collection_info(),
    }
