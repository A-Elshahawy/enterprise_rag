"""Document ingestion endpoints."""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.document_processor import DocumentProcessor
from app.core.embeddings import get_embedding_service
from app.core.vector_store import get_vector_store
from app.models.schemas import DocumentListItem, DocumentListResponse, ErrorResponse, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Initialize services
processor = DocumentProcessor()

# Maximum file size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024


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
    1. Validate and read PDF file
    2. Extract text from PDF pages
    3. Split into overlapping chunks
    4. Generate embeddings (Sentence Transformers)
    5. Store in Qdrant vector database
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if file.content_type and "pdf" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail=f"Invalid content type: {file.content_type}")

    try:
        # Step 1: Read and validate file
        pdf_bytes = await file.read()
        file_size = len(pdf_bytes)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large (max {MAX_FILE_SIZE // (1024 * 1024)}MB)")

        # Step 2-3: Process PDF → extract text → create chunks
        document_id, chunks, page_count, page_texts = processor.process_pdf(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
        )
        # page_texts can be stored/used for highlighting if needed

        if not chunks:
            raise HTTPException(status_code=400, detail="No text content found in PDF")

        # Step 4: Generate embeddings
        embedding_service = get_embedding_service()
        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_service.embed_texts(texts)

        logger.info(f"Generated {len(embeddings)} embeddings for '{file.filename}'")

        # Step 5: Store in Qdrant
        vector_store = get_vector_store()
        stored_count = vector_store.upsert_chunks(chunks, embeddings)

        logger.info(f"Ingested '{file.filename}': id={document_id}, pages={page_count}, chunks={stored_count}")

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
    if not document_id or not document_id.strip():
        raise HTTPException(status_code=400, detail="Document ID is required")

    try:
        vector_store = get_vector_store()
        vector_store.delete_document(document_id.strip())
        return {"message": f"Document '{document_id}' deleted", "document_id": document_id}
    except Exception as e:
        logger.exception(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_collection() -> dict:
    """Clear all documents from the vector store."""
    try:
        vector_store = get_vector_store()
        vector_store.clear_collection()
        return {"message": "Vector store cleared"}
    except Exception as e:
        logger.exception(f"Failed to clear collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all ingested documents."""
    try:
        vector_store = get_vector_store()
        docs = vector_store.list_documents()
        return DocumentListResponse(documents=[DocumentListItem(**doc) for doc in docs], total=len(docs))
    except Exception as e:
        logger.exception(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/page/{page_number}/text")
async def get_page_text(document_id: str, page_number: int) -> dict:
    """
    Get full text content of a specific page for highlighting.
    Reconstructs page text from stored chunks.
    """
    try:
        vector_store = get_vector_store()
        chunks = vector_store.get_chunks_by_page(document_id, page_number)

        if not chunks:
            raise HTTPException(status_code=404, detail=f"No content found for page {page_number}")

        # Sort by char_start to reconstruct page
        chunks.sort(key=lambda c: c.get("char_start", 0))

        # Reconstruct page text by merging chunks (handling overlaps)
        page_text = ""
        last_end = 0

        for chunk in chunks:
            char_start = chunk.get("char_start", 0)
            char_end = chunk.get("char_end", 0)
            text = chunk.get("text", "")

            if char_start >= last_end:
                if char_start > last_end:
                    page_text += " "
                page_text += text
                last_end = char_end
            elif char_end > last_end:
                overlap = last_end - char_start
                if overlap < len(text):
                    page_text += text[overlap:]
                    last_end = char_end

        return {"document_id": document_id, "page_number": page_number, "text": page_text, "chunk_count": len(chunks)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get page text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def ingestion_status() -> dict:
    """Get ingestion service status and collection info."""
    try:
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
    except Exception as e:
        logger.exception(f"Failed to get status: {e}")
        return {
            "status": "error",
            "error": str(e),
        }
