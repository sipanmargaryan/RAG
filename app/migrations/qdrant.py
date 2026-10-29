"""Create or validate the PDF chunk collection outside the upload path."""

import argparse
import asyncio
import logging

from httpx import HTTPError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.config import settings
from app.helpers.exceptions import VectorStorageError

SCHEMA_REVISION = "0001"
logger = logging.getLogger(__name__)


async def migrate_collection(
    client: AsyncQdrantClient, *, check_only: bool = False
) -> None:
    """Apply revision 0001 idempotently, or check it without changing anything.

    Existing compatible collections are adopted by marking their revision.
    Incompatible or newer schemas are rejected without modifying their data.
    """
    try:
        await _run_migration(client, check_only=check_only)
    except (
        UnexpectedResponse,
        ResponseHandlingException,
        HTTPError,
        TimeoutError,
    ) as exc:
        logger.exception("Qdrant collection migration failed")
        raise VectorStorageError("Could not migrate the Qdrant collection") from exc


async def _run_migration(client: AsyncQdrantClient, *, check_only: bool) -> None:
    name = settings.qdrant.collection_name
    if not await client.collection_exists(name):
        if check_only:
            raise VectorStorageError("Qdrant collection is missing; run qdrant-migrate")
        try:
            await client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=settings.embedding.vector_size, distance=models.Distance.COSINE
                ),
                metadata={
                    "embedding_model": settings.embedding.model_name,
                    "schema_revision": SCHEMA_REVISION,
                },
            )
        except UnexpectedResponse as exc:
            if exc.status_code not in (400, 409) or not await client.collection_exists(
                name
            ):
                raise

    info = await client.get_collection(name)
    vectors = info.config.params.vectors
    metadata = info.config.metadata or {}
    if (
        not isinstance(vectors, models.VectorParams)
        or vectors.size != settings.embedding.vector_size
        or vectors.distance != models.Distance.COSINE
        or metadata.get("embedding_model") != settings.embedding.model_name
    ):
        raise VectorStorageError(
            "Qdrant collection does not match the embedding model configuration"
        )

    revision = metadata.get("schema_revision")
    if revision == SCHEMA_REVISION:
        return
    if revision is not None:
        raise VectorStorageError(f"Unsupported Qdrant schema revision: {revision}")
    if check_only:
        raise VectorStorageError(
            "Qdrant schema revision is missing; run qdrant-migrate"
        )

    updated = await client.update_collection(
        collection_name=name,
        metadata={**metadata, "schema_revision": SCHEMA_REVISION},
    )
    if not updated:
        raise VectorStorageError("Qdrant did not confirm the migration")


async def _main(check_only: bool) -> None:
    client = AsyncQdrantClient(
        url=str(settings.qdrant.qdrant_url),
        api_key=(
            settings.qdrant.api_key.get_secret_value()
            if settings.qdrant.api_key
            else None
        ),
        timeout=settings.qdrant.timeout_seconds,
    )
    try:
        await migrate_collection(client, check_only=check_only)
        print(
            f"Qdrant collection {settings.qdrant.collection_name}: "
            f"revision {SCHEMA_REVISION} verified"
        )
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without changes")
    args = parser.parse_args()
    try:
        asyncio.run(_main(args.check))
    except VectorStorageError as exc:
        parser.exit(1, f"Qdrant migration failed: {exc}\n")


if __name__ == "__main__":
    main()
