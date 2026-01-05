"""Query endpoints for semantic search and RAG."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.generator import get_generator
from app.core.retriever import get_retriever
from app.models.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SourceCitation,
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

    except HTTPException:
        raise
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


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={500: {"model": ErrorResponse}},
)
async def ask(request: AskRequest) -> AskResponse:
    """
    RAG endpoint: Retrieve relevant context and generate grounded answer.

    Pipeline:
    1. Semantic search for relevant chunks
    2. Generate answer using Gemini with retrieved context
    3. Return answer with source citations
    """
    try:
        # Retrieve context
        retriever = get_retriever()
        context = retriever.search(
            query=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )

        if not context:
            return AskResponse(
                question=request.question,
                answer="No relevant information found in the knowledge base.",
                sources=[],
                model="none",
            )

        # Generate answer
        generator = get_generator()
        result = generator.generate(
            query=request.question,
            context=context,
            temperature=request.temperature,
        )

        return AskResponse(
            question=request.question,
            answer=result.answer,
            sources=[SourceCitation(**src) for src in result.sources],
            model=result.model,
        )

    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Ask failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
