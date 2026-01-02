import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.core.retriever import get_retriever
from app.models.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    ErrorResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={500: {"model": ErrorResponse}},
)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Semantic search across ingested documents.

    Returns chunks most similar to the query, ranked by relevance score.
    """
    try:
        retriever = get_retriever()
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            document_id=request.document_id,
        )

        return SearchResponse(
            query=request.query,
            results=[
                SearchResultItem(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    text=r.text,
                    page_number=r.page_number,
                    score=r.score,
                    metadata=r.metadata,
                )
                for r in results
            ],
            total=len(results),
        )

    except Exception as e:
        logger.exception(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/search",
    response_model=SearchResponse,
    responses={500: {"model": ErrorResponse}},
)
async def search_get(
    q: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20),
    score_threshold: float = Query(default=0.0, ge=0.0, le=1.0),
    document_id: Optional[str] = Query(default=None),
) -> SearchResponse:
    """GET endpoint for semantic search (convenience)."""
    request = SearchRequest(
        query=q,
        top_k=top_k,
        score_threshold=score_threshold,
        document_id=document_id,
    )
    return await search(request)
