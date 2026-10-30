from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.postgres import database_lifespan
from app.db.qdrant import qdrant_lifespan
from app.helpers.exceptions import register_exception_handlers
from app.helpers.minio import minio_lifespan
from app.routers.health import router as health_router
from app.routers.rag.routes import router as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    async with database_lifespan(), qdrant_lifespan(), minio_lifespan():
        yield


app = FastAPI(
    title="RAG API",
    version="0.1.0",
    description=(
        "Upload a PDF to extract its text and create token-sized chunks. "
        "Processing includes OCR for pages without embedded text. "
        "Successfully processed PDFs are stored in MinIO. "
        "Chunk text and MiniLM embeddings are indexed in Qdrant and linked "
        "to PostgreSQL document metadata. "
        "Search the original text and four variations, then rerank unique matches "
        "to select up to five relevant chunks and generate an answer with sources. "
        "All responses contain a message and optional data."
    ),
    docs_url="/docs",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "health", "description": "Check API availability."},
        {
            "name": "rag",
            "description": "Process documents for retrieval and embedding.",
        },
    ],
    swagger_ui_parameters={
        "tryItOutEnabled": True,
        "displayRequestDuration": True,
    },
    lifespan=lifespan,
)
register_exception_handlers(app)
app.include_router(health_router)
app.include_router(rag_router)
