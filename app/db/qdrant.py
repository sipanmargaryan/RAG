"""Application-scoped async Qdrant client and its connection pool."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends
from httpx import Limits
from qdrant_client import AsyncQdrantClient

from app.config import settings

qdrant_client = AsyncQdrantClient(
    url=str(settings.qdrant.qdrant_url),
    api_key=(
        settings.qdrant.api_key.get_secret_value() if settings.qdrant.api_key else None
    ),
    timeout=settings.qdrant.timeout_seconds,
    limits=Limits(
        max_connections=settings.qdrant.max_connections,
        max_keepalive_connections=settings.qdrant.max_keepalive_connections,
        keepalive_expiry=settings.qdrant.keepalive_expiry_seconds,
    ),
)


@asynccontextmanager
async def qdrant_lifespan() -> AsyncGenerator[None]:
    """Reuse one client and its pool per application worker until shutdown."""
    try:
        yield
    finally:
        await qdrant_client.close()


async def get_qdrant_client() -> AsyncQdrantClient:
    """Return the shared client without closing its pool after each request."""
    return qdrant_client


AsyncQdrantDep = Annotated[AsyncQdrantClient, Depends(get_qdrant_client)]
