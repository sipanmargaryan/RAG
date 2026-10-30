from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from qdrant_client import models
from qdrant_client.http.models import QueryResponse

from app.config import settings
from app.helpers.exceptions import RerankingError
from app.routers.rag.schemas.api import ChunkSearchResult
from app.routers.rag.services import rag as service
from app.routers.rag.services import reranking


def chunk(index):
    return ChunkSearchResult(
        text=f"Chunk {index}",
        score=1.0 - index * 0.01,
        document_id=42,
        chunk_index=index,
        original_filename="document.pdf",
        bucket_name="processed-pdfs",
        object_key="processed/test.pdf",
    )


@pytest.fixture
def model_factory(monkeypatch):
    monkeypatch.setattr(reranking, "_model", None)
    factory = Mock()
    monkeypatch.setattr(reranking, "CrossEncoder", factory)
    return factory


def test_reranking_uses_original_query_and_returns_top_five_with_new_scores(
    model_factory,
):
    chunks = [chunk(index) for index in range(7)]
    model = model_factory.return_value
    model.predict.return_value = np.array([-5.0, 3.0, 1.0, 9.0, 7.0, -1.0, 5.0])

    result = reranking.rerank_chunks("Original question", chunks)

    assert [item.chunk_index for item in result] == [3, 4, 6, 1, 2]
    assert [item.score for item in result] == [9.0, 7.0, 5.0, 3.0, 1.0]
    assert result[0].model_dump(exclude={"score"}) == chunks[3].model_dump(
        exclude={"score"}
    )
    assert chunks[3].score == 0.97
    model.predict.assert_called_once_with(
        [("Original question", item.text) for item in chunks],
        batch_size=settings.reranking.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    model_factory.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_fewer_than_five_candidates_are_all_returned_and_model_is_reused(model_factory):
    model_factory.return_value.predict.return_value = np.array([-3.0, 2.0])
    chunks = [chunk(0), chunk(1)]

    for _ in range(2):
        result = reranking.rerank_chunks("Question", chunks)
        assert [item.chunk_index for item in result] == [1, 0]
        assert [item.score for item in result] == [2.0, -3.0]

    model_factory.assert_called_once()


def test_empty_candidates_do_not_load_model(model_factory):
    assert reranking.rerank_chunks("Question", []) == []
    model_factory.assert_not_called()


def test_predictions_can_overlap_while_reusing_one_model(model_factory):
    barrier = Barrier(2, timeout=5)

    def predict(pairs, **kwargs):
        # Both predictions must enter before either can complete.
        barrier.wait()
        return np.ones(len(pairs))

    model_factory.return_value.predict.side_effect = predict
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(reranking.rerank_chunks, query, [chunk(0)])
            for query in ("First question", "Second question")
        ]
        assert [future.result(timeout=10)[0].score for future in futures] == [1.0, 1.0]
    model_factory.assert_called_once()


@pytest.mark.parametrize(
    "scores",
    [
        np.array([0.5]),
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
        np.array([[0.5], [0.6]]),
        np.array(0.5),
    ],
)
def test_invalid_model_scores_are_rejected(model_factory, scores):
    model_factory.return_value.predict.return_value = scores
    with pytest.raises(RerankingError, match="Reranking model is unavailable"):
        reranking.rerank_chunks("Question", [chunk(0), chunk(1)])


def test_model_loading_failure_is_reported_without_internal_details(model_factory):
    model_factory.side_effect = OSError("Private model download details")
    with pytest.raises(RerankingError, match="^Reranking model is unavailable"):
        reranking.rerank_chunks("Question", [chunk(0)])


def test_inference_failure_is_reported_without_internal_details(model_factory):
    model_factory.return_value.predict.side_effect = RuntimeError("Private details")
    with pytest.raises(RerankingError, match="^Reranking model is unavailable"):
        reranking.rerank_chunks("Question", [chunk(0)])


@pytest.mark.anyio
async def test_search_deduplicates_25_candidates_then_returns_top_five(
    client, mock_qdrant, monkeypatch, model_factory
):
    original = "Original question"
    monkeypatch.setattr(
        service,
        "expand_query",
        AsyncMock(return_value=[original, *[f"Variation {i}" for i in range(4)]]),
    )
    monkeypatch.setattr(service, "rerank_chunks", reranking.rerank_chunks)

    def point(index):
        candidate = chunk(index)
        return models.ScoredPoint(
            id=index + 1,
            version=0,
            score=candidate.score,
            payload=candidate.model_dump(exclude={"score"}),
        )

    mock_qdrant.query_batch_points.return_value = [
        QueryResponse(points=[point(0), *[point(query * 4 + i) for i in range(1, 5)]])
        for query in range(5)
    ]
    model_factory.return_value.predict.return_value = np.arange(21, dtype=float)
    response = await client.post("/rag/search", json={"text": original})

    assert response.status_code == 200
    results = response.json()["data"]["sources"]
    assert [item["chunk_index"] for item in results] == [20, 19, 18, 17, 16]
    assert [item["score"] for item in results] == [20.0, 19.0, 18.0, 17.0, 16.0]
    model_factory.return_value.predict.assert_called_once_with(
        [(original, f"Chunk {index}") for index in range(21)],
        batch_size=settings.reranking.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    assert len(mock_qdrant.query_batch_points.call_args.kwargs["requests"]) == 5
