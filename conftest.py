"""Mocked API fixtures and opt-in PostgreSQL integration fixtures."""

from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from minio import Minio
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import QueryResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema

from app.config import settings
from app.db import postgres, qdrant
from app.db.postgres import Base, get_session
from app.helpers import minio
from app.main import app
from app.routers.rag import service


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if {"db", "integration_client", "postgres_url"}.intersection(item.fixturenames):
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def mock_engine(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    # API sessions use dependency overrides, never the configured app engine.
    engine = AsyncMock(spec=AsyncEngine)
    monkeypatch.setattr(postgres, "engine", engine)
    return engine


@pytest.fixture
def mock_db() -> AsyncMock:
    session = AsyncMock(spec=AsyncSession)

    async def populate_id():
        for call in session.add.call_args_list:
            if getattr(call.args[0], "id", None) is None:
                call.args[0].id = 1

    session.flush.side_effect = populate_id
    return session


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch: pytest.MonkeyPatch) -> Mock:
    encoder = Mock(
        side_effect=lambda texts: [
            [0.1] * settings.embedding.vector_size for _ in texts
        ]
    )
    monkeypatch.setattr(service, "embed_texts", encoder)
    return encoder


@pytest.fixture(autouse=True)
def mock_minio(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock(spec=Minio)
    client.bucket_exists.return_value = True
    monkeypatch.setattr(minio, "minio_client", client)
    return client


@pytest.fixture(autouse=True)
def mock_qdrant(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    client = AsyncMock(spec=AsyncQdrantClient)
    client.collection_exists.return_value = True
    info = Mock()
    info.config.params.vectors = models.VectorParams(
        size=settings.embedding.vector_size, distance=models.Distance.COSINE
    )
    info.config.metadata = {"embedding_model": settings.embedding.model_name}
    client.get_collection.return_value = info
    result = models.UpdateResult(operation_id=1, status=models.UpdateStatus.COMPLETED)
    client.upsert.return_value = result
    client.delete.return_value = result
    client.query_points.return_value = QueryResponse(points=[])
    monkeypatch.setattr(qdrant, "qdrant_client", client)
    return client


@asynccontextmanager
async def _api_client(session: AsyncSession) -> AsyncGenerator[AsyncClient]:
    async def override_get_session() -> AsyncIterator[AsyncSession]:
        yield session

    previous_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_session] = override_get_session
    try:
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client
    finally:
        app.dependency_overrides = previous_overrides


@pytest.fixture
async def client(mock_db: AsyncMock, anyio_backend: str) -> AsyncIterator[AsyncClient]:
    async with _api_client(mock_db) as client:
        yield client


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    # Import lazily so unit tests do not initialize Testcontainers or Docker.
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer("postgres:17", driver="asyncpg") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture
async def db(postgres_url: str, anyio_backend: str) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(postgres_url, poolclass=NullPool)
    schema = f"test_{uuid4().hex}"
    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()
            try:
                # PostgreSQL rolls back schema creation with the outer transaction.
                await connection.execute(CreateSchema(schema))
                await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
                await connection.run_sync(Base.metadata.create_all)
                async with AsyncSession(
                    bind=connection,
                    expire_on_commit=False,
                    join_transaction_mode="create_savepoint",
                ) as session:
                    yield session
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()


@pytest.fixture
async def integration_client(db: AsyncSession) -> AsyncIterator[AsyncClient]:
    async with _api_client(db) as client:
        yield client
