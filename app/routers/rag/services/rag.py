import logging
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import BinaryIO

from minio import Minio
from pydantic import ValidationError
from pypdf.errors import PyPdfError
from qdrant_client import AsyncQdrantClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.helpers.exceptions import (
    DatabaseError,
    InvalidPDFError,
    StorageError,
    VectorStorageError,
)
from app.helpers.minio import remove_pdf, save_pdf
from app.helpers.processing import chunk_texts
from app.helpers.utils import _read_pdf
from app.routers.rag.models import PDFDocument
from app.routers.rag.repository import pdf_document_repository
from app.routers.rag.schemas.api import ChunkSearchResult, RAGResult
from app.routers.rag.schemas.pdf import ProcessedPDF
from app.routers.rag.services.answer_generation import generate_rag_answer
from app.routers.rag.services.embeddings import embed_texts
from app.routers.rag.services.query_expansion import expand_query
from app.routers.rag.services.reranking import rerank_chunks
from app.routers.rag.vector_repository import (
    remove_document_chunks,
    save_document_chunks,
    search_chunks,
)

logger = logging.getLogger(__name__)


async def create_rag_answer(text: str, qdrant: AsyncQdrantClient) -> RAGResult:
    chunks = await search_pdf_chunks(text, qdrant)
    return await generate_rag_answer(text, chunks)


async def search_pdf_chunks(
    text: str, qdrant: AsyncQdrantClient
) -> list[ChunkSearchResult]:
    """Retrieve five chunks per query, deduplicate, and rerank to the top five."""
    queries = await expand_query(text)
    vectors = await run_in_threadpool(embed_texts, queries)
    points = await search_chunks(qdrant, vectors)
    unique_points = {}
    for point in points:
        unique_points.setdefault(point.id, point)
    try:
        chunks = [
            ChunkSearchResult.model_validate(
                {**(point.payload or {}), "score": point.score}
            )
            for point in unique_points.values()
        ]
    except ValidationError as exc:
        logger.exception("Qdrant search returned invalid chunk metadata")
        raise VectorStorageError("Stored search results could not be read") from exc
    return await run_in_threadpool(rerank_chunks, text, chunks)


def process_pdf(stream: BinaryIO, storage: Minio) -> ProcessedPDF:
    """Parse, tokenize, and store an upload; always remove its temporary file.

    This function performs blocking work and should run in a worker thread.
    """
    stream.seek(0)
    if b"%PDF-" not in stream.read(1024):
        raise InvalidPDFError("Upload must contain a PDF file")
    stream.seek(0)

    with TemporaryDirectory(prefix="rag-pdf-") as directory:
        path = Path(directory) / "upload.pdf"
        with path.open("wb") as destination:
            copyfileobj(stream, destination)
        try:
            pages = _read_pdf(path)
        except (PyPdfError, ValueError) as exc:
            raise InvalidPDFError(
                "PDF is unreadable, encrypted, or contains no extractable text"
            ) from exc

        chunks = chunk_texts([page.text for page in pages])
        if not chunks:
            raise InvalidPDFError("No chunks found in the PDF")
        vectors = embed_texts(chunks)
        object_key = save_pdf(storage, path)

    return ProcessedPDF(
        settings.minio.bucket_name, object_key, len(pages), chunks, vectors
    )


async def _cleanup_upload(
    processed: ProcessedPDF, storage: Minio, qdrant: AsyncQdrantClient | None = None
) -> None:
    if qdrant is not None:
        try:
            await remove_document_chunks(
                qdrant, processed.bucket_name, processed.object_key
            )
        except VectorStorageError:
            logger.exception("Chunk cleanup failed; preserving PDF for reconciliation")
            return
    try:
        await run_in_threadpool(
            remove_pdf, storage, processed.bucket_name, processed.object_key
        )
    except StorageError:
        logger.exception("Failed to remove PDF after ingestion failure")


async def create_pdf_document(
    stream: BinaryIO,
    storage: Minio,
    session: AsyncSession,
    original_filename: str | None,
    qdrant: AsyncQdrantClient,
) -> PDFDocument:
    """Process and embed in a worker, write chunks, then commit PDF metadata."""
    processed = await run_in_threadpool(process_pdf, stream, storage)
    document = PDFDocument(
        bucket_name=processed.bucket_name,
        object_key=processed.object_key,
        original_filename=original_filename,
        page_count=processed.page_count,
        chunk_count=len(processed.chunks),
    )
    try:
        document = await pdf_document_repository.save_and_commit(session, document)
    except SQLAlchemyError as exc:
        logger.exception("Failed to save PDF metadata")
        await _cleanup_upload(processed, storage)
        raise DatabaseError(
            "PDF metadata could not be saved; please retry later"
        ) from exc

    try:
        await save_document_chunks(
            qdrant, document, processed.chunks, processed.vectors
        )
    except VectorStorageError:
        await _cleanup_upload(processed, storage, qdrant)
        await pdf_document_repository.delete_and_commit(session, document)
        raise

    return document
