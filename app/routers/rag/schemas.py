"""Request and response payloads for RAG endpoints."""

from pydantic import BaseModel, ConfigDict, Field


class ChunkSearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1, examples=["How are PDFs stored?"])


class ChunkSearchResult(BaseModel):
    text: str
    score: float
    document_id: int
    chunk_index: int
    original_filename: str | None = None
    bucket_name: str
    object_key: str
