from unittest.mock import Mock

import pytest

from app.helpers import processing


@pytest.fixture
def splitter_factory(monkeypatch):
    processing._get_splitter.cache_clear()
    factory = Mock()
    monkeypatch.setattr(processing, "SentenceTransformersTokenTextSplitter", factory)
    yield factory
    processing._get_splitter.cache_clear()


def test_empty_input_does_not_load_model(splitter_factory):
    assert processing.chunk_texts([]) == []
    splitter_factory.assert_not_called()


def test_inputs_split_separately_and_model_is_reused(splitter_factory):
    splitter = splitter_factory.return_value
    splitter.split_text.side_effect = [["one", "two"], ["three"], ["four"]]

    assert processing.chunk_texts(["first", "second"]) == ["one", "two", "three"]
    assert processing.chunk_texts(["third"]) == ["four"]
    splitter_factory.assert_called_once_with(
        tokens_per_chunk=254,
        chunk_overlap=32,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    assert [call.args[0] for call in splitter.split_text.call_args_list] == [
        "first",
        "second",
        "third",
    ]
