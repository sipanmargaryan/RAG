"""Shared MinIO client and storage for successfully processed PDFs."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends
from minio import Minio
from minio.error import MinioException, S3Error
from urllib3 import PoolManager, Retry, Timeout
from urllib3.exceptions import HTTPError

from app.config import settings
from app.helpers.exceptions import StorageError

_http_pool = PoolManager(
    timeout=Timeout(
        connect=settings.minio.timeout_seconds, read=settings.minio.timeout_seconds
    ),
    retries=Retry(total=2, backoff_factor=0.2),
)
minio_client = Minio(
    settings.minio.endpoint,
    access_key=settings.minio.access_key.get_secret_value(),
    secret_key=settings.minio.secret_key.get_secret_value(),
    secure=settings.minio.secure,
    region=settings.minio.region,
    http_client=_http_pool,
)


def save_pdf(client: Minio, path: Path) -> str:
    """Store PDF bytes under a unique key, creating the bucket if necessary.

    Run in a worker thread. Concurrent workers may create the same bucket;
    only an already-owned bucket is treated as a successful creation race.
    """
    bucket = settings.minio.bucket_name
    object_name = f"processed/{uuid4().hex}.pdf"
    try:
        if not client.bucket_exists(bucket):
            try:
                client.make_bucket(bucket, location=settings.minio.region)
            except S3Error as exc:
                if exc.code != "BucketAlreadyOwnedByYou":
                    raise
        client.fput_object(
            bucket, object_name, str(path), content_type="application/pdf"
        )
    except (MinioException, HTTPError) as exc:
        raise StorageError("PDF storage is unavailable; please retry later") from exc
    return object_name


def remove_pdf(client: Minio, bucket: str, object_name: str) -> None:
    """Remove an uploaded PDF after a failed metadata write."""
    try:
        client.remove_object(bucket, object_name)
    except (MinioException, HTTPError) as exc:
        raise StorageError("Failed to clean up the uploaded PDF") from exc


@asynccontextmanager
async def minio_lifespan() -> AsyncGenerator[None]:
    """Release the shared HTTP connections when the application shuts down."""
    try:
        yield
    finally:
        _http_pool.clear()


async def get_minio_client() -> Minio:
    return minio_client


MinioDep = Annotated[Minio, Depends(get_minio_client)]
