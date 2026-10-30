"""Generate RAG embeddings in a worker thread using a cached local model."""

import logging
from math import isfinite
from threading import Lock

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.helpers.exceptions import EmbeddingError

logger = logging.getLogger(__name__)
_initialization_lock = Lock()
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _initialization_lock:
            if _model is None:
                _model = SentenceTransformer(settings.embedding.model_name)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one normalized vector per chunk without blocking the event loop."""
    if not texts:
        return []
    try:
        vectors = (
            _get_model()
            .encode(
                texts,
                batch_size=settings.embedding.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            .tolist()
        )
        if len(vectors) != len(texts) or any(
            len(vector) != settings.embedding.vector_size
            or not all(isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise ValueError(
                "Embedding output does not match the configured dimensions"
            )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("Failed to generate chunk embeddings")
        raise EmbeddingError(
            "Embedding model is unavailable; please retry later"
        ) from exc
    return vectors
