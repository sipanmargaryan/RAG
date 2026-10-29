from pathlib import Path
from unittest.mock import Mock

import pytest

from app.helpers.utils import PDFPage
from app.routers.rag import service


@pytest.mark.anyio
async def test_pdf_upload_extracts_text_before_tokenizing(
    client, monkeypatch, mock_minio, mock_db
):
    paths = []

    def read_pdf(path):
        assert path.read_bytes() == b"%PDF-1.7\nexample"
        paths.append(path)
        return [PDFPage(1, "Digital text.", False), PDFPage(3, "OCR text.", True)]

    tokenizer = Mock(return_value=["Digital text.", "OCR text."])
    monkeypatch.setattr(service, "_read_pdf", read_pdf)
    monkeypatch.setattr(service, "chunk_texts", tokenizer)

    def upload(bucket, object_name, path, **kwargs):
        tokenizer.assert_called_once_with(["Digital text.", "OCR text."])
        assert Path(path).read_bytes() == b"%PDF-1.7\nexample"
        assert object_name.startswith("processed/")
        assert object_name.endswith(".pdf")
        assert kwargs["content_type"] == "application/pdf"

    mock_minio.fput_object.side_effect = upload
    response = await client.post(
        "/rag/pdf",
        files={"file": ("../../example.pdf", b"%PDF-1.7\nexample", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "PDF processed successfully",
        "data": None,
    }
    tokenizer.assert_called_once_with(["Digital text.", "OCR text."])
    mock_minio.fput_object.assert_called_once()
    document = mock_db.add.call_args.args[0]
    assert document.bucket_name == mock_minio.fput_object.call_args.args[0]
    assert document.object_key == mock_minio.fput_object.call_args.args[1]
    assert document.original_filename == "../../example.pdf"
    assert document.page_count == 2
    assert document.chunk_count == 2
    mock_db.commit.assert_awaited_once()
    assert paths[0].name == "upload.pdf"
    assert not paths[0].parent.exists()


@pytest.mark.anyio
@pytest.mark.parametrize("content", [b"", b"plain text", b"%PDF-1.7\ninvalid"])
async def test_invalid_pdf_does_not_reach_tokenizer(
    client, monkeypatch, content, mock_minio
):
    tokenizer = Mock()
    monkeypatch.setattr(service, "chunk_texts", tokenizer)
    response = await client.post(
        "/rag/pdf", files={"file": ("example.pdf", content, "application/pdf")}
    )
    assert response.status_code == 422
    tokenizer.assert_not_called()
    mock_minio.fput_object.assert_not_called()


@pytest.mark.anyio
async def test_unreadable_pdf_cleans_up_temporary_file(client, monkeypatch):
    paths = []

    def read_pdf(path):
        paths.append(path)
        raise ValueError("No text found")

    monkeypatch.setattr(service, "_read_pdf", read_pdf)
    response = await client.post(
        "/rag/pdf", files={"file": ("blank.pdf", b"%PDF-1.7", "application/pdf")}
    )
    assert response.status_code == 422
    assert not paths[0].parent.exists()


@pytest.mark.anyio
async def test_pdf_upload_requires_file(client):
    response = await client.post("/rag/pdf")
    assert response.status_code == 422


@pytest.mark.anyio
async def test_health_uses_shared_response_model(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "Service is healthy", "data": None}
