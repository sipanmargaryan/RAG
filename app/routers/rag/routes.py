from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from app.db.postgres import AsyncSessionDep
from app.db.qdrant import AsyncQdrantDep
from app.helpers.exceptions import ERROR_RESPONSES
from app.helpers.minio import MinioDep
from app.helpers.responses import APIResponse
from app.routers.rag.schemas import ChunkSearchRequest, ChunkSearchResult
from app.routers.rag.service import create_pdf_document, search_pdf_chunks

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/search",
    response_model=APIResponse[list[ChunkSearchResult]],
    summary="Search PDF chunks by text",
    responses={
        **ERROR_RESPONSES,
        422: {
            "model": APIResponse[object],
            "description": "Search text is required and must not be blank.",
        },
    },
)
async def search_pdf(
    request: ChunkSearchRequest, qdrant: AsyncQdrantDep
) -> APIResponse[list[ChunkSearchResult]]:
    """Embed the text and return at most five chunks ranked by cosine similarity.

    Results include the chunk text, score, and PDF source metadata. No matches
    return an empty list. The Qdrant collection must already be migrated.
    """
    chunks = await search_pdf_chunks(request.text, qdrant)
    return APIResponse[list[ChunkSearchResult]](message="Search completed", data=chunks)


@router.post(
    "/pdf",
    response_model=APIResponse[None],
    status_code=status.HTTP_200_OK,
    summary="Process a PDF",
    responses={
        **ERROR_RESPONSES,
        200: {
            "description": "PDF stored, metadata committed, and chunks indexed.",
            "content": {
                "application/json": {
                    "example": {"message": "PDF processed successfully", "data": None}
                }
            },
        },
    },
)
async def process_pdf_upload(
    file: Annotated[
        UploadFile,
        File(
            description="PDF to extract and tokenize",
            json_schema_extra={"format": "binary"},
        ),
    ],
    storage: MinioDep,
    session: AsyncSessionDep,
    qdrant: AsyncQdrantDep,
) -> APIResponse[None]:
    """Upload a PDF using multipart form field **file**.

    Extract text, use OCR where necessary, and create chunks using the MiniLM
    tokenizer (254 content tokens with 32-token overlap). Processing finishes before
    this endpoint responds. The original PDF is saved to MinIO only after
    processing succeeds. Its bucket and object key are recorded in PostgreSQL.
    Chunk text and normalized MiniLM embeddings are stored in Qdrant, linked
    to the document ID. Chunks are not included in the HTTP response.
    OCR requires Tesseract on the server. The tokenizer model loads on first
    use and may need to download files, making the first request slower.
    """
    await create_pdf_document(file.file, storage, session, file.filename, qdrant)
    return APIResponse[None](message="PDF processed successfully")
