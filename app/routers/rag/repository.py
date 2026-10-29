from app.helpers.repository import BaseRepository
from app.routers.rag.models import PDFDocument


class PDFDocumentRepository(BaseRepository[PDFDocument]):
    pass


pdf_document_repository = PDFDocumentRepository(PDFDocument)
