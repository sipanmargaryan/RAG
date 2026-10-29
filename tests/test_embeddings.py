from unittest.mock import Mock

import numpy as np
import pytest

from app.config import settings
from app.helpers import embeddings
from app.helpers.exceptions import EmbeddingError


@pytest.fixture
def model_factory(monkeypatch):
    embeddings._get_model.cache_clear()
    factory = Mock()
    monkeypatch.setattr(embeddings, "SentenceTransformer", factory)
    yield factory
    embeddings._get_model.cache_clear()


def test_embeddings_match_chunks_and_reuse_model(model_factory):
    model = model_factory.return_value
    model.encode.return_value = np.full((2, settings.embedding.vector_size), 0.1)
    texts = ["one", "two"]

    assert len(embeddings.embed_texts(texts)) == 2
    embeddings.embed_texts(texts)

    model_factory.assert_called_once_with(settings.embedding.model_name)
    model.encode.assert_called_with(
        texts,
        batch_size=settings.embedding.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def test_empty_input_does_not_load_embedding_model(model_factory):
    assert embeddings.embed_texts([]) == []
    model_factory.assert_not_called()


@pytest.mark.parametrize("output", [np.zeros((1, 3)), np.full((1, 384), np.nan)])
def test_bad_vectors_are_rejected(model_factory, output):
    model_factory.return_value.encode.return_value = output
    with pytest.raises(EmbeddingError):
        embeddings.embed_texts(["text"])


def test_model_failure_is_reported_without_internal_details(model_factory):
    model_factory.side_effect = OSError("Internal download details")
    with pytest.raises(EmbeddingError, match="Embedding model is unavailable"):
        embeddings.embed_texts(["text"])
