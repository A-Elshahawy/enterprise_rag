import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import health, ingest, query
from app.config import get_settings
from app.exceptions import (
    generic_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.middleware import APIKeyMiddleware, LoggingMiddleware, RequestIDMiddleware
from app.utils import setup_logging

logger = logging.getLogger(__name__)
settings = get_settings()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    setup_logging(debug=settings.debug)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Validate Qdrant connection on startup
    try:
        from app.core.vector_store import get_vector_store  # noqa: PLC0415

        vs = get_vector_store()
        vs.ensure_collection()
        logger.info("Qdrant connection verified")
    except Exception as e:
        logger.warning(f"Qdrant startup check failed: {e}")

    try:
        from app.core.embeddings import get_embedding_service  # noqa: PLC0415

        es = get_embedding_service()
        _ = es.dimension
        es.embed_text("warmup")
        logger.info("Embedding model warmed up")
    except Exception as e:
        logger.warning(f"Embedding model warmup failed: {e}")

    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Enterprise RAG Platform",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    app.state.limiter = limiter

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # API Routers
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)

    # Static Files (Frontend)
    # Ensure the static directory exists
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def read_root():
            return FileResponse(static_dir / "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8000, reload=True)
