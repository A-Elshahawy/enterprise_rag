"""Enterprise RAG Platform.

A production-grade Retrieval-Augmented Generation system.
"""

__version__ = "0.1.0"

# Config
# Routes
from app.api.routes import health, ingest, query
from app.config import Settings, get_settings

# Core services
from app.core import (
    DocumentProcessor,
    EmbeddingService,
    Generator,
    Retriever,
    SearchResult,
    VectorStore,
    get_embedding_service,
    get_generator,
    get_retriever,
    get_vector_store,
)
from app.exceptions import (
    generic_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)

# Middleware & Exceptions
from app.middleware import (
    APIKeyMiddleware,
    LoggingMiddleware,
    RequestIDMiddleware,
    get_request_id,
)

# Models
from app.models import (
    AskRequest,
    AskResponse,
    DocumentMetadata,
    ErrorResponse,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SourceCitation,
)

# Utils
from app.utils import setup_logging

__all__ = [
    # Version
    "__version__",
    # Config
    "Settings",
    "get_settings",
    # Core
    "DocumentProcessor",
    "EmbeddingService",
    "get_embedding_service",
    "VectorStore",
    "get_vector_store",
    "Retriever",
    "SearchResult",
    "get_retriever",
    "Generator",
    "get_generator",
    # Models
    "DocumentMetadata",
    "IngestResponse",
    "ChunkSchema",
    "ErrorResponse",
    "SearchRequest",
    "SearchResultItem",
    "SearchResponse",
    "AskRequest",
    "AskResponse",
    "SourceCitation",
    # Middleware
    "RequestIDMiddleware",
    "LoggingMiddleware",
    "APIKeyMiddleware",
    "get_request_id",
    # Exceptions
    "http_exception_handler",
    "validation_exception_handler",
    "generic_exception_handler",
    # Utils
    "setup_logging",
    # Routes
    "health",
    "ingest",
    "query",
]
