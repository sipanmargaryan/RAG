import pytest
from httpx import HTTPError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import QueryResponse

from app.config import settings
from app.helpers.exceptions import EmbeddingError
from app.migrations.qdrant import migrate_collection
from app.routers.rag.vector_repository import search_chunks


def scored_point(index):
    return models.ScoredPoint(
        id=index + 1,
        version=0,
        score=1.0 - index * 0.1,
        payload={
            "text": f"Chunk {index}",
            "document_id": 42,
            "chunk_index": index,
            "original_filename": "document.pdf",
            "bucket_name": "processed-pdfs",
            "object_key": "processed/test.pdf",
            "embedding_model": settings.embedding.model_name,
        },
    )


@pytest.mark.anyio
async def test_search_embeds_text_and_returns_at_most_five_chunks(
    client, mock_qdrant, mock_embeddings, mock_db
):
    mock_qdrant.query_points.return_value = QueryResponse(
        points=[scored_point(index) for index in range(7)]
    )
    response = await client.post("/rag/search", json={"text": "  PDF storage  "})

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Search completed"
    assert len(body["data"]) == 5
    assert [chunk["text"] for chunk in body["data"]] == [f"Chunk {i}" for i in range(5)]
    assert body["data"][0] == {
        "text": "Chunk 0",
        "score": 1.0,
        "document_id": 42,
        "chunk_index": 0,
        "original_filename": "document.pdf",
        "bucket_name": "processed-pdfs",
        "object_key": "processed/test.pdf",
    }
    mock_embeddings.assert_called_once_with(["PDF storage"])
    query = mock_qdrant.query_points.call_args.kwargs
    assert query["query"] == [0.1] * settings.embedding.vector_size
    assert query["limit"] == 5
    assert query["collection_name"] == settings.qdrant.collection_name
    assert query["with_vectors"] is False
    mock_qdrant.collection_exists.assert_not_awaited()
    mock_qdrant.create_collection.assert_not_awaited()
    mock_db.commit.assert_not_awaited()


@pytest.mark.anyio
async def test_search_returns_empty_list_when_there_are_no_matches(client):
    response = await client.post("/rag/search", json={"text": "PDF storage"})
    assert response.status_code == 200
    assert response.json() == {"message": "Search completed", "data": []}


@pytest.mark.anyio
@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": " \n\t "}, {"text": None}])
async def test_invalid_search_text_is_rejected(
    client, mock_qdrant, mock_embeddings, body
):
    response = await client.post("/rag/search", json=body)
    assert response.status_code == 422
    assert response.json()["message"] == "Request validation failed"
    mock_embeddings.assert_not_called()
    mock_qdrant.query_points.assert_not_awaited()


@pytest.mark.anyio
async def test_search_reports_vector_store_failure(client, mock_qdrant):
    mock_qdrant.query_points.side_effect = HTTPError("Internal connection details")
    response = await client.post("/rag/search", json={"text": "PDF storage"})
    assert response.status_code == 503
    assert response.json() == {
        "message": "Vector search is unavailable; please retry later",
        "data": None,
    }


@pytest.mark.anyio
async def test_search_reports_embedding_failure(client, mock_embeddings, mock_qdrant):
    mock_embeddings.side_effect = EmbeddingError("Embedding model is unavailable")
    response = await client.post("/rag/search", json={"text": "PDF storage"})
    assert response.status_code == 503
    mock_qdrant.query_points.assert_not_awaited()


@pytest.mark.anyio
async def test_qdrant_search_ranks_five_points_and_filters_model():
    client = AsyncQdrantClient(":memory:")
    vector = [1.0] + [0.0] * (settings.embedding.vector_size - 1)
    try:
        await migrate_collection(client)
        points = [
            models.PointStruct(
                id=index + 1,
                vector=[1.0, float(index)]
                + [0.0] * (settings.embedding.vector_size - 2),
                payload={
                    "text": str(index),
                    "embedding_model": settings.embedding.model_name,
                },
            )
            for index in range(7)
        ]
        points.append(
            models.PointStruct(
                id=99,
                vector=vector,
                payload={"text": "Different model", "embedding_model": "other-model"},
            )
        )
        await client.upsert(settings.qdrant.collection_name, points=points)
        results = await search_chunks(client, vector)
        assert [point.id for point in results] == [1, 2, 3, 4, 5]
        assert [point.score for point in results] == sorted(
            [point.score for point in results], reverse=True
        )
        assert all(point.vector is None for point in results)
    finally:
        await client.close()
