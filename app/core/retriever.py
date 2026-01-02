import logging
from typing import List, Optional
from dataclasses import dataclass

from qdrant_client.http import models

from app.core.embeddings import get_embedding_service
from app.core.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    chunk_id: str
    document_id: str
    text: str
    page_number: int
    score: float
    metadata: dict


class Retriever:
    """Semantic search retriever."""

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        document_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Semantic search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            document_id: Filter by specific document

        Returns:
            List of SearchResult ordered by relevance
        """
        # Embed query
        query_embedding = self.embedding_service.embed_text(query)

        # Build filter
        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )

        # Search Qdrant
        results = self.vector_store.client.search(
            collection_name=self.vector_store.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
        )

        # Convert to SearchResult
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(
                SearchResult(
                    chunk_id=str(hit.id),
                    document_id=payload.get("document_id", ""),
                    text=payload.get("text", ""),
                    page_number=payload.get("page_number", 0),
                    score=hit.score,
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in ("document_id", "text", "page_number")
                    },
                )
            )

        logger.info(f"Search '{query[:50]}...' returned {len(search_results)} results")
        return search_results


def get_retriever() -> Retriever:
    """Get retriever instance."""
    return Retriever()
