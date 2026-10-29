from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import BinaryIO

import pypdfium2 as pdfium
import pytesseract
from PIL import Image
from pypdf import PdfReader

_PDFIUM_LOCK = Lock()


def _render_page(stream: BinaryIO, page_index: int) -> Image.Image:
    # PDFium is not thread-safe, even for separate documents. Copy the pixels
    # before releasing native resources so OCR can run outside the lock.
    with _PDFIUM_LOCK:
        stream.seek(0)
        with (
            pdfium.PdfDocument(stream, autoclose=False) as document,
            closing(document[page_index]) as page,
            closing(page.render(scale=300 / 72)) as bitmap,
            bitmap.to_pil() as image,
        ):
            return image.copy()


@dataclass(frozen=True)
class PDFPage:
    page_number: int
    text: str
    used_ocr: bool


def _read_pdf(
    filename: str | Path,
    *,
    ocr_language: str = "eng",
    force_ocr: bool = False,
    ocr_timeout: float = 60,
) -> list[PDFPage]:
    """Read nonempty pages, using OCR when a page has no extractable text.

    Page numbers are one-based and remain accurate when empty pages are skipped.
    Set force_ocr for scans with an incomplete or unusable existing text layer.
    Tesseract and the requested language data must be installed for OCR.
    Raises ValueError if no text is found; file, PDF, and OCR errors propagate.
    Run this synchronous function in a worker thread in async endpoints.
    """
    if ocr_timeout <= 0:
        raise ValueError("ocr_timeout must be greater than zero")

    pages: list[PDFPage] = []
    with ExitStack() as stack:
        stream = stack.enter_context(Path(filename).open("rb"))
        reader = PdfReader(stream)

        for page_number, page in enumerate(reader.pages, start=1):
            text = "" if force_ocr else (page.extract_text() or "").strip()
            used_ocr = not text
            if used_ocr:
                with _render_page(stream, page_number - 1) as image:
                    try:
                        text = pytesseract.image_to_string(
                            image, lang=ocr_language, timeout=ocr_timeout
                        ).strip()
                    except pytesseract.TesseractNotFoundError as exc:
                        raise RuntimeError(
                            "PDF OCR requires Tesseract. Install tesseract-ocr "
                            "and the requested language data, and ensure "
                            "tesseract is on PATH."
                        ) from exc

            if text:
                pages.append(PDFPage(page_number, text, used_ocr))

    if not pages:
        raise ValueError("No text found in the PDF after text extraction and OCR")
    return pages
