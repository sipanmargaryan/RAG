import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from app.helpers.exceptions import register_exception_handlers


@pytest.mark.anyio
async def test_http_errors_use_shared_response_and_preserve_headers():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/private")
    async def private():
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/private")
        missing = await client.get("/missing")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"message": "Authentication required", "data": None}
    assert missing.status_code == 404
    assert missing.json() == {"message": "Not Found", "data": None}
