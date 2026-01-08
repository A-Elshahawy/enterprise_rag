import logging
from dataclasses import dataclass
from typing import List, Optional

from qdrant_client.http import models

from app.core.embeddings import get_embedding_service
from app.core.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with position info for highlighting."""

    chunk_id: str
    document_id: str
    text: str
    page_number: int
    score: float
    metadata: dict

    # Position tracking for highlighting
    char_start: int = 0
    char_end: int = 0


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
        document_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Semantic search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            document_id: Filter by single document (deprecated, use document_ids)
            document_ids: Filter by multiple documents

        Returns:
            List of SearchResult ordered by relevance
        """
        # Embed query
        query_embedding = self.embedding_service.embed_text(query)

        # Build filter - prioritize document_ids over document_id
        query_filter = self._build_filter(document_id, document_ids)

        # Search Qdrant
        client = self.vector_store.client

        try:
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=self.vector_store.collection_name,
                    query=query_embedding,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                    with_payload=True,
                )
                results = response.points if hasattr(response, "points") else response
            elif hasattr(client, "search"):
                results = client.search(
                    collection_name=self.vector_store.collection_name,
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                    with_payload=True,
                )
            else:
                raise AttributeError("Qdrant client does not support query_points or search")
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            raise

        # Convert to SearchResult
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            search_results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id") or str(hit.id),
                    document_id=payload.get("document_id", ""),
                    text=payload.get("text", ""),
                    page_number=payload.get("page_number", 0),
                    score=hit.score,
                    char_start=payload.get("char_start", 0),
                    char_end=payload.get("char_end", 0),
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in ("chunk_id", "document_id", "text", "page_number", "char_start", "char_end")
                    },
                )
            )

        logger.info(f"Search '{query[:50]}...' returned {len(search_results)} results")
        return search_results

    def _build_filter(
        self,
        document_id: Optional[str],
        document_ids: Optional[List[str]],
    ) -> Optional[models.Filter]:
        """
        Build Qdrant filter for document filtering.

        Args:
            document_id: Single document ID (legacy support)
            document_ids: List of document IDs

        Returns:
            Qdrant Filter or None
        """
        # Normalize and clean document_ids
        ids_to_filter = []

        if document_ids:
            # Clean and validate document_ids
            ids_to_filter = [d.strip() for d in document_ids if d and d.strip()]

        # Fall back to single document_id if document_ids is empty
        if not ids_to_filter and document_id and document_id.strip():
            ids_to_filter = [document_id.strip()]

        # No filter if no IDs provided
        if not ids_to_filter:
            logger.debug("No document filter applied - searching all documents")
            return None

        logger.debug(f"Building filter for document_ids: {ids_to_filter}")

        # Single document - use MatchValue
        if len(ids_to_filter) == 1:
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=ids_to_filter[0]),
                    )
                ]
            )

        # Multiple documents - use MatchAny for proper OR filtering
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=ids_to_filter),
                )
            ]
        )


def get_retriever() -> Retriever:
    """Get retriever instance."""
    return Retriever()
