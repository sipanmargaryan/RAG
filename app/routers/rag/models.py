from sqlalchemy import BigInteger, Identity, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import BaseDBModel


class PDFDocument(BaseDBModel):
    __tablename__ = "pdf_documents"
    __table_args__ = (
        UniqueConstraint("bucket_name", "object_key", name="uq_pdf_documents_object"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    bucket_name: Mapped[str] = mapped_column(String(63))
    object_key: Mapped[str] = mapped_column(String(1024))
    original_filename: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer)
