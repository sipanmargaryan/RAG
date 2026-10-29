import shutil
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from app.helpers import utils
from app.helpers.utils import PDFPage, _read_pdf


def write_pdf(path: Path, *, digital_text: bool = False) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=150)
    if digital_text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        content = DecodedStreamObject()
        content.set_data(b"BT /F1 24 Tf 20 80 Td (Digital text) Tj ET")
        page[NameObject("/Contents")] = content
    writer.write(path)


def test_embedded_text_does_not_use_ocr(tmp_path, monkeypatch):
    path = tmp_path / "digital.pdf"
    write_pdf(path, digital_text=True)
    ocr = Mock(side_effect=AssertionError("OCR should not run"))
    monkeypatch.setattr(utils.pytesseract, "image_to_string", ocr)

    assert _read_pdf(path) == [PDFPage(1, "Digital text", False)]


def test_fallback_preserves_page_numbers_and_options(tmp_path, monkeypatch):
    path = tmp_path / "mixed.pdf"
    write_pdf(path, digital_text=True)
    writer = PdfWriter(clone_from=path)
    writer.add_blank_page(width=300, height=150)
    writer.add_blank_page(width=300, height=150)
    writer.write(path)
    ocr = Mock(side_effect=["  ", " Scanned text\n"])
    monkeypatch.setattr(utils.pytesseract, "image_to_string", ocr)

    assert _read_pdf(path, ocr_language="eng+fra", ocr_timeout=12) == [
        PDFPage(1, "Digital text", False),
        PDFPage(3, "Scanned text", True),
    ]
    assert ocr.call_count == 2
    assert ocr.call_args.kwargs == {"lang": "eng+fra", "timeout": 12}


def test_force_ocr_replaces_existing_text(tmp_path, monkeypatch):
    path = tmp_path / "digital.pdf"
    write_pdf(path, digital_text=True)
    monkeypatch.setattr(utils.pytesseract, "image_to_string", Mock(return_value="OCR"))

    assert _read_pdf(path, force_ocr=True) == [PDFPage(1, "OCR", True)]


def test_no_text_is_explicit(tmp_path, monkeypatch):
    path = tmp_path / "blank.pdf"
    write_pdf(path)
    monkeypatch.setattr(utils.pytesseract, "image_to_string", Mock(return_value=""))

    with pytest.raises(ValueError, match="No text found"):
        _read_pdf(path)


def test_missing_tesseract_has_installation_hint(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    write_pdf(path)
    monkeypatch.setattr(
        utils.pytesseract,
        "image_to_string",
        Mock(side_effect=utils.pytesseract.TesseractNotFoundError()),
    )

    with pytest.raises(RuntimeError, match="Install tesseract-ocr"):
        _read_pdf(path)


def test_ocr_timeout_propagates(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    write_pdf(path)
    monkeypatch.setattr(
        utils.pytesseract,
        "image_to_string",
        Mock(side_effect=RuntimeError("Tesseract process timeout")),
    )

    with pytest.raises(RuntimeError, match="timeout"):
        _read_pdf(path)


def test_invalid_timeout():
    with pytest.raises(ValueError, match="ocr_timeout"):
        _read_pdf("unused.pdf", ocr_timeout=0)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract required")
def test_real_ocr_of_image_only_pdf(tmp_path):
    path = tmp_path / "scan.pdf"
    with Image.new("RGB", (1200, 300), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.text(
            (60, 80),
            "Scanned document 12345",
            font=ImageFont.load_default(size=60),
            fill="black",
        )
        image.save(path, "PDF", resolution=150)

    # Ensure this fixture really has no embedded text layer.
    assert not PdfReader(BytesIO(path.read_bytes())).pages[0].extract_text()
    pages = _read_pdf(path)
    assert len(pages) == 1
    assert pages[0].used_ocr
    assert "Scanned document 12345" in pages[0].text
