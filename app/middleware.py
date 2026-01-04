import logging
import time
import uuid
from typing import Callable, Optional

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add X-Request-ID to all requests for tracing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        request_id = getattr(request.state, "request_id", "unknown")

        # Log request
        logger.info(f"[{request_id}] {request.method} {request.url.path} started")

        response = await call_next(request)

        # Log response
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"completed {response.status_code} in {duration_ms:.2f}ms"
        )

        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optional API key authentication."""

    # Endpoints that don't require auth
    PUBLIC_PATHS = {"/health", "/health/live", "/health/ready", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip if no API key configured
        if not settings.api_key:
            return await call_next(request)

        # Skip public endpoints
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Check API key
        api_key = request.headers.get(settings.api_key_header)
        if api_key != settings.api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key"},
            )

        return await call_next(request)


def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", "unknown")  # noqa: S104
