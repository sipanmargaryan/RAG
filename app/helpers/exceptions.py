"""Application errors, shared HTTP handlers, and OpenAPI error responses."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.helpers.responses import APIResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """An expected application failure with a client-safe message."""

    status_code = status.HTTP_400_BAD_REQUEST


class InvalidPDFError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class StorageError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class DatabaseError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class VectorStorageError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class EmbeddingError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


def register_exception_handlers(app: FastAPI) -> None:
    """Apply the common response envelope without changing HTTP status codes."""

    @app.exception_handler(AppError)
    async def application_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=APIResponse[None](message=str(exc)).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            headers=exc.headers,
            content=APIResponse[object](
                message=str(exc.detail)
                if isinstance(exc.detail, str)
                else "HTTP error",
                data=None if isinstance(exc.detail, str) else exc.detail,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=APIResponse[object](
                message="Request validation failed",
                data={"errors": jsonable_encoder(exc.errors())},
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled request error", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse[None](message="Internal server error").model_dump(),
        )


ERROR_RESPONSES = {
    422: {
        "model": APIResponse[object],
        "description": "Request validation failed or the PDF cannot be processed.",
        "content": {
            "application/json": {
                "examples": {
                    "invalid_pdf": {
                        "summary": "File is not a PDF",
                        "value": {
                            "message": "Upload must contain a PDF file",
                            "data": None,
                        },
                    },
                    "missing_file": {
                        "summary": "Required upload is missing",
                        "value": {
                            "message": "Request validation failed",
                            "data": {
                                "errors": [
                                    {
                                        "type": "missing",
                                        "loc": ["body", "file"],
                                        "msg": "Field required",
                                    }
                                ]
                            },
                        },
                    },
                }
            }
        },
    },
    503: {
        "model": APIResponse[None],
        "description": "PDF, metadata, embedding, or vector storage is unavailable.",
        "content": {
            "application/json": {
                "example": {
                    "message": "PDF storage is unavailable; please retry later",
                    "data": None,
                }
            }
        },
    },
    500: {
        "model": APIResponse[None],
        "description": "Unexpected processing or server failure.",
        "content": {
            "application/json": {
                "example": {"message": "Internal server error", "data": None}
            }
        },
    },
}
