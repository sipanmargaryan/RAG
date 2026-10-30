from unittest.mock import Mock

import pytest
from minio import Minio
from minio.error import S3Error
from urllib3.exceptions import HTTPError

from app.config import settings
from app.helpers.exceptions import StorageError
from app.helpers.minio import save_pdf


def test_stores_bytes_with_unique_object_names(tmp_path):
    client = Mock(spec=Minio)
    client.bucket_exists.return_value = True
    path = tmp_path / "upload.pdf"
    path.write_bytes(b"%PDF-example")

    first = save_pdf(client, path)
    second = save_pdf(client, path)

    assert first != second
    assert first.startswith("processed/") and first.endswith(".pdf")
    client.fput_object.assert_any_call(
        settings.minio.bucket_name, first, str(path), content_type="application/pdf"
    )
    client.make_bucket.assert_not_called()


@pytest.mark.parametrize("creation_race", [False, True])
def test_creates_bucket_and_handles_creation_race(tmp_path, creation_race):
    client = Mock(spec=Minio)
    client.bucket_exists.return_value = False
    if creation_race:
        client.make_bucket.side_effect = S3Error(
            response=Mock(),
            code="BucketAlreadyOwnedByYou",
            message="Already owned",
            resource=None,
            request_id=None,
            host_id=None,
        )

    save_pdf(client, tmp_path / "upload.pdf")

    client.make_bucket.assert_called_once_with(
        settings.minio.bucket_name, location=settings.minio.region
    )
    client.fput_object.assert_called_once()


def test_bucket_owned_by_someone_else_is_not_ignored(tmp_path):
    client = Mock(spec=Minio)
    client.bucket_exists.return_value = False
    client.make_bucket.side_effect = S3Error(
        response=Mock(),
        code="BucketAlreadyExists",
        message="Unavailable",
        resource=None,
        request_id=None,
        host_id=None,
    )
    with pytest.raises(StorageError):
        save_pdf(client, tmp_path / "upload.pdf")
    client.fput_object.assert_not_called()


@pytest.mark.anyio
async def test_storage_failure_returns_503(client, monkeypatch, mock_minio):
    from app.helpers.utils import PDFPage
    from app.routers.rag.services import rag as service

    paths = []

    def read_pdf(path):
        paths.append(path)
        return [PDFPage(1, "PDF text", False)]

    monkeypatch.setattr(service, "_read_pdf", read_pdf)
    monkeypatch.setattr(service, "chunk_texts", Mock(return_value=["PDF text"]))
    mock_minio.fput_object.side_effect = HTTPError("Connection failed")

    response = await client.post(
        "/rag/pdf", files={"file": ("document.pdf", b"%PDF-1.7", "application/pdf")}
    )

    assert response.status_code == 503
    assert response.json() == {
        "message": "PDF storage is unavailable; please retry later",
        "data": None,
    }
    assert not paths[0].parent.exists()


@pytest.mark.anyio
async def test_tokenization_failure_does_not_store_pdf(client, monkeypatch, mock_minio):
    from httpx import ASGITransport, AsyncClient

    from app.helpers.utils import PDFPage
    from app.main import app
    from app.routers.rag.services import rag as service

    monkeypatch.setattr(
        service, "_read_pdf", Mock(return_value=[PDFPage(1, "PDF text", False)])
    )
    monkeypatch.setattr(
        service, "chunk_texts", Mock(side_effect=RuntimeError("Model failure"))
    )
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as http:
        response = await http.post(
            "/rag/pdf", files={"file": ("document.pdf", b"%PDF-1.7", "application/pdf")}
        )

    assert response.status_code == 500
    assert response.json() == {"message": "Internal server error", "data": None}
    mock_minio.fput_object.assert_not_called()
