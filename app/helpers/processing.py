from functools import cache

from langchain_text_splitters import SentenceTransformersTokenTextSplitter

from app.config import settings


@cache
def _get_splitter() -> SentenceTransformersTokenTextSplitter:
    return SentenceTransformersTokenTextSplitter(
        # MiniLM's 256-token input includes two special tokens.
        tokens_per_chunk=254,
        chunk_overlap=32,
        model_name=settings.embedding.model_name,
    )


def chunk_texts(texts: list[str]) -> list[str]:
    if not texts:
        return []

    splitter = _get_splitter()
    chunks: list[str] = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks
