"""Request and response payloads for RAG endpoints."""

from pydantic import BaseModel, ConfigDict, Field


class ChunkSearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1, examples=["How are PDFs stored?"])


class ChunkSearchResult(BaseModel):
    text: str
    score: float = Field(description="CrossEncoder relevance score; higher is better.")
    document_id: int
    chunk_index: int
    original_filename: str | None = None
    bucket_name: str
    object_key: str


class RAGSource(ChunkSearchResult):
    source_id: int = Field(ge=1)


class RAGResult(BaseModel):
    answer: str
    sources: list[RAGSource]
