import logging
from uuid import NAMESPACE_URL, uuid5

from httpx import HTTPError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.config import settings
from app.helpers.exceptions import VectorStorageError
from app.routers.rag.models import PDFDocument

logger = logging.getLogger(__name__)
_CLIENT_ERRORS = (
    UnexpectedResponse,
    ResponseHandlingException,
    HTTPError,
    TimeoutError,
)


async def search_chunks(
    client: AsyncQdrantClient, vector: list[float]
) -> list[models.ScoredPoint]:
    """Return up to five nearest chunks without returning their vectors."""
    try:
        result = await client.query_points(
            collection_name=settings.qdrant.collection_name,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="embedding_model",
                        match=models.MatchValue(value=settings.embedding.model_name),
                    )
                ]
            ),
            limit=5,
            with_payload=True,
            with_vectors=False,
        )
    except _CLIENT_ERRORS as exc:
        logger.exception("Failed to search PDF chunks in Qdrant")
        raise VectorStorageError(
            "Vector search is unavailable; please retry later"
        ) from exc
    return result.points[:5]


async def save_document_chunks(
    client: AsyncQdrantClient,
    document: PDFDocument,
    chunks: list[str],
    vectors: list[list[float]],
) -> None:
    """Upsert batches with stable IDs and wait for confirmed writes."""
    if not chunks or len(chunks) != len(vectors):
        raise VectorStorageError("Each PDF chunk requires an embedding")
    try:
        batch_size = settings.qdrant.batch_size
        for start in range(0, len(chunks), batch_size):
            points = [
                models.PointStruct(
                    id=str(
                        uuid5(
                            NAMESPACE_URL,
                            f"{document.bucket_name}/{document.object_key}/{index}",
                        )
                    ),
                    vector=vectors[index],
                    payload={
                        "document_id": document.id,
                        "bucket_name": document.bucket_name,
                        "object_key": document.object_key,
                        "original_filename": document.original_filename,
                        "chunk_index": index,
                        "text": chunks[index],
                        "embedding_model": settings.embedding.model_name,
                    },
                )
                for index in range(start, min(start + batch_size, len(chunks)))
            ]
            result = await client.upsert(
                collection_name=settings.qdrant.collection_name,
                points=points,
                wait=True,
            )
            if result.status != models.UpdateStatus.COMPLETED:
                raise VectorStorageError("Qdrant did not confirm the chunk writes")
    except _CLIENT_ERRORS as exc:
        logger.exception("Failed to store PDF chunks in Qdrant")
        raise VectorStorageError(
            "Vector storage is unavailable; please retry later"
        ) from exc


async def remove_document_chunks(
    client: AsyncQdrantClient, bucket: str, object_key: str
) -> None:
    """Remove only this upload's points, including partially written batches."""
    try:
        result = await client.delete(
            collection_name=settings.qdrant.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="bucket_name", match=models.MatchValue(value=bucket)
                        ),
                        models.FieldCondition(
                            key="object_key", match=models.MatchValue(value=object_key)
                        ),
                    ]
                )
            ),
            wait=True,
        )
        if result.status != models.UpdateStatus.COMPLETED:
            raise VectorStorageError("Qdrant did not confirm chunk cleanup")
    except UnexpectedResponse as exc:
        if exc.status_code != 404:
            raise VectorStorageError("Failed to remove PDF chunks") from exc
    except _CLIENT_ERRORS as exc:
        raise VectorStorageError("Failed to remove PDF chunks") from exc
