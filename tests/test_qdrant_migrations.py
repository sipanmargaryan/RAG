from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.config import settings
from app.helpers.exceptions import VectorStorageError
from app.migrations.qdrant import SCHEMA_REVISION, migrate_collection


@pytest.mark.anyio
async def test_migration_creates_collection_once(monkeypatch):
    client = AsyncQdrantClient(":memory:")
    create = AsyncMock(wraps=client.create_collection)
    update = AsyncMock(wraps=client.update_collection)
    monkeypatch.setattr(client, "create_collection", create)
    monkeypatch.setattr(client, "update_collection", update)
    try:
        await migrate_collection(client)
        await migrate_collection(client)
        await migrate_collection(client, check_only=True)
        create.assert_awaited_once()
        update.assert_not_awaited()
        info = await client.get_collection(settings.qdrant.collection_name)
        assert info.config.metadata["schema_revision"] == SCHEMA_REVISION
        assert info.config.params.vectors.size == settings.embedding.vector_size
    finally:
        await client.close()


@pytest.mark.anyio
async def test_check_mode_does_not_create_missing_collection():
    client = AsyncQdrantClient(":memory:")
    try:
        with pytest.raises(VectorStorageError, match="collection is missing"):
            await migrate_collection(client, check_only=True)
        assert not await client.collection_exists(settings.qdrant.collection_name)
    finally:
        await client.close()


@pytest.mark.anyio
async def test_existing_collection_is_adopted_without_changing_points():
    client = AsyncQdrantClient(":memory:")
    name = settings.qdrant.collection_name
    try:
        await client.create_collection(
            name,
            vectors_config=models.VectorParams(
                size=settings.embedding.vector_size, distance=models.Distance.COSINE
            ),
            metadata={
                "embedding_model": settings.embedding.model_name,
                "custom": "keep",
            },
        )
        await client.upsert(
            name,
            points=[
                models.PointStruct(
                    id=1,
                    vector=[1.0] + [0.0] * (settings.embedding.vector_size - 1),
                    payload={"text": "existing content"},
                )
            ],
        )
        with pytest.raises(VectorStorageError, match="revision is missing"):
            await migrate_collection(client, check_only=True)
        assert (
            "schema_revision" not in (await client.get_collection(name)).config.metadata
        )

        await migrate_collection(client)
        await migrate_collection(client)
        info = await client.get_collection(name)
        assert info.config.metadata == {
            "embedding_model": settings.embedding.model_name,
            "custom": "keep",
            "schema_revision": SCHEMA_REVISION,
        }
        points = await client.retrieve(name, ids=[1], with_vectors=True)
        assert points[0].payload == {"text": "existing content"}
        assert points[0].vector[0] == 1.0
        assert (await client.count(name)).count == 1
    finally:
        await client.close()


@pytest.mark.anyio
async def test_unknown_revision_is_not_overwritten():
    client = AsyncQdrantClient(":memory:")
    name = settings.qdrant.collection_name
    try:
        await client.create_collection(
            name,
            vectors_config=models.VectorParams(
                size=settings.embedding.vector_size, distance=models.Distance.COSINE
            ),
            metadata={
                "embedding_model": settings.embedding.model_name,
                "schema_revision": "9999",
            },
        )
        with pytest.raises(VectorStorageError, match="Unsupported"):
            await migrate_collection(client)
        assert (await client.get_collection(name)).config.metadata[
            "schema_revision"
        ] == "9999"
    finally:
        await client.close()
