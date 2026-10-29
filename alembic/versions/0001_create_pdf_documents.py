"""Create PostgreSQL records identifying PDFs stored in MinIO.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pdf_documents",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("bucket_name", sa.String(63), nullable=False),
        sa.Column("object_key", sa.String(1024), nullable=False),
        sa.Column("original_filename", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "bucket_name", "object_key", name="uq_pdf_documents_object"
        ),
    )


def downgrade() -> None:
    op.drop_table("pdf_documents")
