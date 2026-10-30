from unittest.mock import AsyncMock

import pytest
from httpx import HTTPError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import QueryResponse

from app.config import settings
from app.helpers.exceptions import EmbeddingError, QueryExpansionError, RerankingError
from app.migrations.qdrant import migrate_collection
from app.routers.rag.services import rag as service
from app.routers.rag.services.answer_generation import NO_CONTEXT_ANSWER
from app.routers.rag.vector_repository import search_chunks


@pytest.fixture(autouse=True)
def mock_expansion(monkeypatch):
    expansion = AsyncMock(
        return_value=["PDF storage", *[f"PDF storage query {i}" for i in range(4)]]
    )
    monkeypatch.setattr(service, "expand_query", expansion)
    return expansion


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
async def test_search_retrieves_five_per_query_and_reranks_unique_chunks(
    client,
    mock_qdrant,
    mock_embeddings,
    mock_db,
    mock_expansion,
    mock_reranking,
    mock_answer_generation,
):
    mock_qdrant.query_batch_points.return_value = [
        QueryResponse(points=[scored_point(index) for index in range(7)])
    ] * 5
    response = await client.post("/rag/search", json={"text": "  PDF storage  "})

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Answer generated"
    assert body["data"]["answer"] == "Mock RAG answer"
    sources = body["data"]["sources"]
    assert len(sources) == 5
    assert [chunk["text"] for chunk in sources] == [f"Chunk {i}" for i in range(5)]
    mock_answer_generation.assert_awaited_once()
    assert mock_answer_generation.call_args.args[0] == "PDF storage"
    assert len(mock_answer_generation.call_args.args[1]) == 5
    mock_reranking.assert_called_once()
    original, candidates = mock_reranking.call_args.args
    assert original == "PDF storage"
    assert [chunk.chunk_index for chunk in candidates] == list(range(5))
    assert sources[0] == {
        "source_id": 1,
        "text": "Chunk 0",
        "score": 1.0,
        "document_id": 42,
        "chunk_index": 0,
        "original_filename": "document.pdf",
        "bucket_name": "processed-pdfs",
        "object_key": "processed/test.pdf",
    }
    mock_expansion.assert_awaited_once_with("PDF storage")
    mock_embeddings.assert_called_once_with(mock_expansion.return_value)
    mock_qdrant.query_batch_points.assert_awaited_once()
    query = mock_qdrant.query_batch_points.call_args.kwargs
    assert query["collection_name"] == settings.qdrant.collection_name
    assert len(query["requests"]) == 5
    for request in query["requests"]:
        assert request.query == [0.1] * settings.embedding.vector_size
        assert request.limit == 5
        assert request.with_vector is False
        assert request.with_payload is True
        assert request.filter.must[0].match.value == settings.embedding.model_name
    mock_qdrant.collection_exists.assert_not_awaited()
    mock_qdrant.create_collection.assert_not_awaited()
    mock_db.commit.assert_not_awaited()


@pytest.mark.anyio
async def test_search_reports_insufficient_information_when_no_matches(client):
    response = await client.post("/rag/search", json={"text": "PDF storage"})
    assert response.status_code == 200
    assert response.json() == {
        "message": "Answer generated",
        "data": {"answer": NO_CONTEXT_ANSWER, "sources": []},
    }


@pytest.mark.anyio
@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": " \n\t "}, {"text": None}])
async def test_invalid_search_text_is_rejected(
    client, mock_qdrant, mock_embeddings, mock_expansion, body
):
    response = await client.post("/rag/search", json=body)
    assert response.status_code == 422
    assert response.json()["message"] == "Request validation failed"
    mock_embeddings.assert_not_called()
    mock_expansion.assert_not_awaited()
    mock_qdrant.query_batch_points.assert_not_awaited()


@pytest.mark.anyio
async def test_search_reports_vector_store_failure(client, mock_qdrant):
    mock_qdrant.query_batch_points.side_effect = HTTPError(
        "Internal connection details"
    )
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
    mock_qdrant.query_batch_points.assert_not_awaited()


@pytest.mark.anyio
async def test_expansion_failure_stops_before_embeddings_and_vector_search(
    client, mock_expansion, mock_embeddings, mock_qdrant
):
    mock_expansion.side_effect = QueryExpansionError(
        "Query expansion is unavailable; please retry later"
    )
    response = await client.post("/rag/search", json={"text": "PDF storage"})

    assert response.status_code == 503
    assert response.json() == {
        "message": "Query expansion is unavailable; please retry later",
        "data": None,
    }
    mock_embeddings.assert_not_called()
    mock_qdrant.query_batch_points.assert_not_awaited()


@pytest.mark.anyio
async def test_reranking_failure_returns_service_error(
    client, mock_reranking, mock_qdrant
):
    mock_qdrant.query_batch_points.return_value = [
        QueryResponse(points=[scored_point(0)])
    ] * 5
    mock_reranking.side_effect = RerankingError(
        "Reranking model is unavailable; please retry later"
    )
    response = await client.post("/rag/search", json={"text": "PDF storage"})

    assert response.status_code == 503
    assert response.json() == {
        "message": "Reranking model is unavailable; please retry later",
        "data": None,
    }


@pytest.mark.anyio
async def test_batch_results_preserve_each_queries_matches_and_scores(mock_qdrant):
    duplicate = scored_point(0).model_copy(update={"score": 0.95})
    better_duplicate = scored_point(0).model_copy(update={"score": 0.99})
    mock_qdrant.query_batch_points.return_value = [
        QueryResponse(points=[duplicate, scored_point(1), scored_point(2)]),
        QueryResponse(points=[better_duplicate, scored_point(3)]),
        QueryResponse(points=[scored_point(4)]),
        QueryResponse(points=[scored_point(5)]),
        QueryResponse(points=[scored_point(6)]),
    ]

    results = await search_chunks(
        mock_qdrant, [[0.1] * settings.embedding.vector_size] * 5
    )

    assert [point.id for point in results] == [1, 2, 3, 1, 4, 5, 6, 7]
    assert results[0].score == 0.95
    assert results[3].score == 0.99


@pytest.mark.anyio
async def test_qdrant_search_ranks_five_points_per_query_and_filters_model():
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
        results = await search_chunks(client, [vector] * 5)
        assert [point.id for point in results] == [1, 2, 3, 4, 5] * 5
        for start in range(0, 25, 5):
            scores = [point.score for point in results[start : start + 5]]
            assert scores == sorted(scores, reverse=True)
        assert all(point.vector is None for point in results)
    finally:
        await client.close()
