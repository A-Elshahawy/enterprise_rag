import logging
import uuid
from functools import lru_cache
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import get_settings
from app.core.document_processor import Chunk

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStore:
    """Qdrant vector store for document chunks."""

    def __init__(self) -> None:
        self._client: Optional[QdrantClient] = None
        self.collection_name = settings.qdrant_collection_name
        self.dimension = settings.embedding_dimension

    @property
    def client(self) -> QdrantClient:
        """Lazy load Qdrant client."""
        if self._client is None:
            if settings.is_qdrant_cloud:
                self._client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    timeout=settings.qdrant_timeout,
                )
                logger.info(f"Connected to Qdrant Cloud: {settings.qdrant_url}")
            else:
                self._client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    timeout=settings.qdrant_timeout,
                )
                logger.info(f"Connected to local Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
        return self._client

    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' exists")
        except (UnexpectedResponse, Exception) as e:
            logger.info(f"Creating collection '{self.collection_name}' (reason: {e})")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            # Create payload indexes for filtering
            self._create_payload_indexes()
            logger.info(f"Collection '{self.collection_name}' created")

    def _create_payload_indexes(self) -> None:
        """Create payload indexes for efficient filtering."""
        indexes = [
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("filename", models.PayloadSchemaType.KEYWORD),
            ("page_number", models.PayloadSchemaType.INTEGER),
        ]

        for field_name, field_schema in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as e:
                logger.warning(f"Failed to create index for '{field_name}': {e}")

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store chunks with their embeddings in Qdrant.

        Returns:
            Number of points upserted
        """
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have same length")

        self.ensure_collection()

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    **chunk.metadata,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(f"Upserted {len(points)} chunks to '{self.collection_name}'")
        return len(points)

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document.

        Returns:
            True if deletion was successful
        """
        if not document_id:
            raise ValueError("Document ID is required")

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        logger.info(f"Deleted chunks for document '{document_id}'")
        return True

    def get_chunks_by_page(self, document_id: str, page_number: int) -> List[dict]:
        """
        Get all chunks for a specific document page.

        Args:
            document_id: Document ID
            page_number: Page number (1-indexed)

        Returns:
            List of chunk payloads with text and position info
        """
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        ),
                        models.FieldCondition(
                            key="page_number",
                            match=models.MatchValue(value=page_number),
                        ),
                    ]
                ),
                limit=100,
                with_payload=True,
            )

            return [point.payload for point in results if point.payload]
        except Exception as e:
            logger.error(f"Failed to get chunks for page: {e}")
            return []

    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.warning(f"Failed to delete collection: {e}")

        self.ensure_collection()
        logger.info(f"Recreated collection '{self.collection_name}'")

    def list_documents(self, max_points_to_scan: int = 5000) -> List[dict]:
        """
        List all unique documents in the collection.

        Args:
            max_points_to_scan: Maximum number of points to scan for documents

        Returns:
            List of dicts with document_id and filename
        """
        documents: dict[str, str] = {}

        limit = 256
        scanned = 0
        offset = None

        while scanned < max_points_to_scan:
            try:
                res = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=["document_id", "filename"],
                    with_vectors=False,
                )
            except Exception as e:
                logger.error(f"Failed to scroll collection: {e}")
                break

            if isinstance(res, tuple):
                points, next_offset = res
            else:
                points = getattr(res, "points", [])
                next_offset = getattr(res, "next_page_offset", None)

            if not points:
                break

            scanned += len(points)

            for p in points:
                payload = p.payload or {}
                doc_id = payload.get("document_id")
                if not doc_id or doc_id in documents:
                    continue

                filename = payload.get("filename")
                documents[str(doc_id)] = str(filename) if filename and str(filename).strip() else str(doc_id)

            if not next_offset:
                break
            offset = next_offset

        return [{"document_id": k, "filename": v} for k, v in sorted(documents.items(), key=lambda x: x[1].lower())]

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value if hasattr(info.status, "value") else str(info.status),
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"name": self.collection_name, "error": str(e)}


@lru_cache
def get_vector_store() -> VectorStore:
    """Get cached vector store instance."""
    return VectorStore()
