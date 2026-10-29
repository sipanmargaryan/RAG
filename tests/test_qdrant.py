"""Qdrant client reuse and lifecycle cleanup without a live database."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.db.qdrant import AsyncQdrantDep
from app.main import lifespan


@pytest.mark.anyio
async def test_requests_share_client_until_shutdown(mock_qdrant: AsyncMock) -> None:
    app = FastAPI(lifespan=lifespan)
    clients = []

    @app.get("/collections")
    async def collections(
        client: AsyncQdrantDep,
    ):
        clients.append(client)
        await client.get_collections()
        return {"status": "ok"}

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http:
            for _ in range(2):
                response = await http.get("/collections")
                assert response.status_code == 200
        assert clients == [mock_qdrant, mock_qdrant]
        assert mock_qdrant.get_collections.await_count == 2
        mock_qdrant.close.assert_not_awaited()

    mock_qdrant.close.assert_awaited_once()


@pytest.mark.anyio
async def test_client_closed_on_lifespan_error(mock_qdrant: AsyncMock) -> None:
    app = FastAPI(lifespan=lifespan)
    with pytest.raises(RuntimeError, match="operation failed"):
        async with app.router.lifespan_context(app):
            raise RuntimeError("operation failed")

    mock_qdrant.close.assert_awaited_once()
