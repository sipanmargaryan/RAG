"""Request session isolation and application resource cleanup."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db import postgres
from app.db.postgres import AsyncSessionDep
from app.main import lifespan


@pytest.mark.anyio
async def test_requests_receive_separate_sessions(
    monkeypatch: pytest.MonkeyPatch, mock_engine: AsyncMock
) -> None:
    # Unbound sessions perform no network IO in this endpoint.
    monkeypatch.setattr(
        postgres, "AsyncSessionLocal", async_sessionmaker(expire_on_commit=False)
    )
    app = FastAPI(lifespan=lifespan)
    sessions = []

    @app.get("/session")
    async def session_endpoint(session: AsyncSessionDep):
        sessions.append(session)
        return {"status": "ok"}

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            for _ in range(2):
                response = await client.get("/session")
                assert response.status_code == 200
        assert len(sessions) == 2
        assert sessions[0] is not sessions[1]
        mock_engine.dispose.assert_not_awaited()
    mock_engine.dispose.assert_awaited_once()


@pytest.mark.anyio
async def test_engine_disposed_if_qdrant_cleanup_fails(
    mock_engine: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    mock_qdrant.close.side_effect = RuntimeError("Qdrant close failed")
    app = FastAPI(lifespan=lifespan)

    with pytest.raises(RuntimeError, match="Qdrant close failed"):
        async with app.router.lifespan_context(app):
            pass

    mock_engine.dispose.assert_awaited_once()


@pytest.mark.anyio
async def test_resources_closed_on_lifespan_error(
    mock_engine: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    app = FastAPI(lifespan=lifespan)

    with pytest.raises(RuntimeError, match="operation failed"):
        async with app.router.lifespan_context(app):
            raise RuntimeError("operation failed")

    mock_qdrant.close.assert_awaited_once()
    mock_engine.dispose.assert_awaited_once()
