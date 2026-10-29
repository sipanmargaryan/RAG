"""Generate dense embeddings in a worker thread using a cached local model."""

import logging
from functools import cache
from math import isfinite
from threading import Lock

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.helpers.exceptions import EmbeddingError

logger = logging.getLogger(__name__)
_model_lock = Lock()


@cache
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding.model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one normalized vector per chunk without blocking the event loop."""
    if not texts:
        return []
    try:
        # Serialize initialization and inference across request worker threads.
        with _model_lock:
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
