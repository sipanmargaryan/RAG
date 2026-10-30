from unittest.mock import Mock

import pytest
from httpx import HTTPError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.helpers.exceptions import VectorStorageError
from app.helpers.utils import PDFPage
from app.migrations.qdrant import migrate_collection
from app.routers.rag.models import PDFDocument
from app.routers.rag.services import rag as service
from app.routers.rag.vector_repository import (
    remove_document_chunks,
    save_document_chunks,
)


def document(key="processed/test.pdf", document_id=42):
    return PDFDocument(
        id=document_id,
        bucket_name="processed-pdfs",
        object_key=key,
        original_filename="test.pdf",
        page_count=1,
        chunk_count=3,
    )


@pytest.mark.anyio
async def test_payloads_batching_and_stable_ids_after_migration(monkeypatch):
    monkeypatch.setattr(settings.qdrant, "batch_size", 2)
    client = AsyncQdrantClient(":memory:")
    chunks = ["First chunk", "Second chunk", "Third chunk"]
    vectors = [[1.0] + [0.0] * (settings.embedding.vector_size - 1)] * 3
    try:
        await migrate_collection(client)
        await save_document_chunks(client, document(), chunks, vectors)
        await save_document_chunks(client, document(), chunks, vectors)
        count = await client.count(settings.qdrant.collection_name)
        assert count.count == 3
        points, _ = await client.scroll(
            settings.qdrant.collection_name, with_vectors=True, limit=10
        )
        points.sort(key=lambda point: point.payload["chunk_index"])
        assert [point.payload["text"] for point in points] == chunks
        assert all(point.payload["document_id"] == 42 for point in points)
        assert all(
            point.payload["object_key"] == "processed/test.pdf" for point in points
        )
        assert all(
            len(point.vector) == settings.embedding.vector_size for point in points
        )

        await save_document_chunks(
            client, document("processed/other.pdf", 43), chunks[:1], vectors[:1]
        )
        await remove_document_chunks(client, "processed-pdfs", "processed/test.pdf")
        remaining, _ = await client.scroll(settings.qdrant.collection_name)
        assert len(remaining) == 1
        assert remaining[0].payload["document_id"] == 43
    finally:
        await client.close()


@pytest.mark.anyio
@pytest.mark.parametrize("wrong_model", [False, True])
async def test_incompatible_collection_is_not_modified(wrong_model):
    client = AsyncQdrantClient(":memory:")
    try:
        await client.create_collection(
            settings.qdrant.collection_name,
            vectors_config=models.VectorParams(
                size=settings.embedding.vector_size if wrong_model else 3,
                distance=models.Distance.COSINE,
            ),
            metadata={
                "embedding_model": "other-model"
                if wrong_model
                else settings.embedding.model_name
            },
        )
        with pytest.raises(VectorStorageError, match="does not match"):
            await migrate_collection(client)
        assert (await client.count(settings.qdrant.collection_name)).count == 0
    finally:
        await client.close()


@pytest.mark.anyio
async def test_uploads_do_not_check_or_create_collections(mock_qdrant):
    await save_document_chunks(
        mock_qdrant, document(), ["text"], [[0.1] * settings.embedding.vector_size]
    )

    mock_qdrant.upsert.assert_awaited_once()
    mock_qdrant.collection_exists.assert_not_awaited()
    mock_qdrant.get_collection.assert_not_awaited()
    mock_qdrant.create_collection.assert_not_awaited()
    mock_qdrant.update_collection.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_partial_write_failure_rolls_back_and_cleans_up(
    client, mock_db, mock_minio, mock_qdrant, monkeypatch, cleanup_fails
):
    monkeypatch.setattr(settings.qdrant, "batch_size", 2)
    monkeypatch.setattr(
        service, "_read_pdf", Mock(return_value=[PDFPage(1, "text", False)])
    )
    monkeypatch.setattr(service, "chunk_texts", Mock(return_value=["a", "b", "c"]))
    mock_qdrant.upsert.side_effect = [
        models.UpdateResult(operation_id=1, status=models.UpdateStatus.COMPLETED),
        UnexpectedResponse(
            status_code=500, reason_phrase="Unavailable", content=b"error", headers={}
        ),
    ]
    if cleanup_fails:
        mock_qdrant.delete.side_effect = HTTPError("Connection failed")

    response = await client.post(
        "/rag/pdf", files={"file": ("test.pdf", b"%PDF-1.7", "application/pdf")}
    )

    assert response.status_code == 503
    assert (
        response.json()["message"]
        == "Vector storage is unavailable; please retry later"
    )
    assert mock_qdrant.upsert.await_count == 2
    mock_db.commit.assert_not_awaited()
    mock_db.rollback.assert_awaited_once()
    selector = mock_qdrant.delete.call_args.kwargs["points_selector"]
    bucket, key = mock_minio.fput_object.call_args.args[:2]
    assert selector.filter.must[0].match.value == bucket
    assert selector.filter.must[1].match.value == key
    if cleanup_fails:
        mock_minio.remove_object.assert_not_called()
    else:
        mock_minio.remove_object.assert_called_once_with(bucket, key)
