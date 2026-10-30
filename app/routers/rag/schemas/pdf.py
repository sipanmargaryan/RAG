from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessedPDF:
    bucket_name: str
    object_key: str
    page_count: int
    chunks: list[str]
    vectors: list[list[float]]
