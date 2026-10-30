"""Rerank retrieved PDF chunks against the original query in a worker thread."""

import logging
from math import isfinite
from threading import Lock

from sentence_transformers import CrossEncoder

from app.config import settings
from app.helpers.exceptions import RerankingError
from app.routers.rag.schemas.api import ChunkSearchResult

logger = logging.getLogger(__name__)
_initialization_lock = Lock()
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        with _initialization_lock:
            if _model is None:
                _model = CrossEncoder(settings.reranking.model_name)
    return _model


def rerank_chunks(
    query: str, chunks: list[ChunkSearchResult]
) -> list[ChunkSearchResult]:
    """Score unique candidates with the original query and return the best five."""
    if not chunks:
        return []
    try:
        scores = _get_model().predict(
            [(query, chunk.text) for chunk in chunks],
            batch_size=settings.reranking.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if (
            scores.ndim != 1
            or len(scores) != len(chunks)
            or not all(isfinite(score) for score in scores)
        ):
            raise ValueError("Reranker must return one finite score per chunk")
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("Failed to rerank PDF chunks")
        raise RerankingError(
            "Reranking model is unavailable; please retry later"
        ) from exc

    ranked = [
        chunk.model_copy(update={"score": float(score)})
        for chunk, score in zip(chunks, scores, strict=True)
    ]
    return sorted(ranked, key=lambda chunk: chunk.score, reverse=True)[:5]
