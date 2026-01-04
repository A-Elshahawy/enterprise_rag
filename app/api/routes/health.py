import logging

from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

settings = get_settings()


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns:
        dict: Health status with Qdrant connection status.
    """
    qdrant_status = "disconnected"

    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=5.0,
        )
        # Check Qdrant connectivity
        client.get_collections()
        qdrant_status = "connected"
        client.close()
    except UnexpectedResponse as e:
        logger.warning(f"Qdrant unexpected response: {e}")
        qdrant_status = "error"
    except Exception as e:
        logger.warning(f"Qdrant connection failed: {e}")
        qdrant_status = "disconnected"

    return {
        "status": "healthy",
        "version": settings.app_version,
        "qdrant": qdrant_status,
    }


@router.get("/health/live")
async def liveness() -> dict:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness() -> dict:
    """
    Kubernetes readiness probe.

    Checks if Qdrant is accessible.
    """
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=5.0,
        )
        client.get_collections()
        client.close()
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")
