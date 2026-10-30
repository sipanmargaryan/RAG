from io import BytesIO
from unittest.mock import Mock

import pytest
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from urllib3.exceptions import HTTPError

from app.helpers.utils import PDFPage
from app.routers.rag.models import PDFDocument
from app.routers.rag.services import rag as service


@pytest.fixture
def parsed_pdf(monkeypatch):
    monkeypatch.setattr(
        service, "_read_pdf", Mock(return_value=[PDFPage(1, "PDF text", False)])
    )
    monkeypatch.setattr(service, "chunk_texts", Mock(return_value=["PDF text"]))


@pytest.mark.anyio
async def test_metadata_write_failure_rolls_back_and_removes_pdf(
    client, mock_db, mock_minio, parsed_pdf
):
    mock_db.flush.side_effect = SQLAlchemyError("Database unavailable")

    response = await client.post(
        "/rag/pdf", files={"file": ("document.pdf", b"%PDF-1.7", "application/pdf")}
    )

    assert response.status_code == 503
    assert response.json() == {
        "message": "PDF metadata could not be saved; please retry later",
        "data": None,
    }
    bucket, key = mock_minio.fput_object.call_args.args[:2]
    mock_minio.remove_object.assert_called_once_with(bucket, key)
    mock_db.rollback.assert_awaited_once()
    mock_db.commit.assert_not_awaited()


@pytest.mark.anyio
async def test_uncertain_commit_preserves_pdf(client, mock_db, mock_minio, parsed_pdf):
    mock_db.commit.side_effect = SQLAlchemyError("Connection lost during commit")
    response = await client.post(
        "/rag/pdf", files={"file": ("document.pdf", b"%PDF-1.7", "application/pdf")}
    )

    assert response.status_code == 503
    mock_minio.fput_object.assert_called_once()
    mock_minio.remove_object.assert_not_called()
    mock_db.rollback.assert_awaited_once()


@pytest.mark.anyio
async def test_cleanup_failure_preserves_metadata_error(
    client, mock_db, mock_minio, parsed_pdf
):
    mock_db.flush.side_effect = SQLAlchemyError("Database unavailable")
    mock_minio.remove_object.side_effect = HTTPError("Storage unavailable")
    response = await client.post(
        "/rag/pdf", files={"file": ("document.pdf", b"%PDF-1.7", "application/pdf")}
    )

    assert response.status_code == 503
    assert (
        response.json()["message"]
        == "PDF metadata could not be saved; please retry later"
    )


@pytest.mark.anyio
async def test_document_identifier_persisted_in_postgres(
    db, mock_minio, mock_qdrant, parsed_pdf
):
    document = await service.create_pdf_document(
        BytesIO(b"%PDF-1.7"), mock_minio, db, "document.pdf", mock_qdrant
    )
    assert document.id is not None
    db.expunge_all()
    saved = await db.scalar(select(PDFDocument).where(PDFDocument.id == document.id))
    assert saved is not None
    assert saved.object_key == mock_minio.fput_object.call_args.args[1]
    assert saved.bucket_name == mock_minio.fput_object.call_args.args[0]
    assert saved.original_filename == "document.pdf"
    assert saved.created_at is not None
    assert saved.page_count == 1
    assert saved.chunk_count == 1
