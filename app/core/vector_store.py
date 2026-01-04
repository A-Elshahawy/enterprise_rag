import logging
from functools import lru_cache
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import get_settings
from app.core.document_processor import Chunk

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStore:
    """Qdrant vector store for document chunks."""

    def __init__(self):
        self._client = None
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
                )
                logger.info(f"Connected to Qdrant Cloud: {settings.qdrant_url}")
            else:
                self._client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                )
                logger.info(f"Connected to local Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
        return self._client

    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' exists")
        except (UnexpectedResponse, Exception):
            logger.info(f"Creating collection '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            # Create payload indexes for filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Collection '{self.collection_name}' created")

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
            raise ValueError("Chunks and embeddings must have same length")

        self.ensure_collection()

        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload={
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
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

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        result = self.client.delete(
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
        return result

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
            }
        except Exception as e:
            return {"error": str(e)}


@lru_cache
def get_vector_store() -> VectorStore:
    """Get cached vector store instance."""
    return VectorStore()
