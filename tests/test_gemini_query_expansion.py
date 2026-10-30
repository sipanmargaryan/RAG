import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from app.config import QueryExpansionSettings, settings
from app.helpers.exceptions import QueryExpansionError
from app.routers.rag.services import llm, query_expansion

ORIGINAL = "How are PDFs stored?"
VARIATIONS = [
    "PDF document storage process",
    "Where are uploaded PDF files saved?",
    "Persistence of processed PDF documents",
    "Storage location for original PDF uploads",
]


def completed_response(queries=VARIATIONS):
    return {
        "status": "completed",
        "steps": [
            {"type": "thought", "content": [{"type": "text", "text": "Reasoning"}]},
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps({"queries": queries})}],
            },
        ],
    }


@pytest.fixture
def gemini_transport(monkeypatch):
    monkeypatch.setattr(settings.query_expansion, "provider", "gemini")
    monkeypatch.setattr(settings.gemini, "api_key", SecretStr("gemini-test-key"))
    monkeypatch.setattr(settings.openai, "api_key", None)
    monkeypatch.setattr(llm, "sleep", AsyncMock())
    real_client = httpx.AsyncClient

    def install(handler):
        monkeypatch.setattr(
            llm.httpx,
            "AsyncClient",
            lambda **kwargs: real_client(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        )

    return install


@pytest.mark.anyio
async def test_gemini_request_uses_structured_output_and_preserves_original(
    gemini_transport,
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completed_response())

    gemini_transport(handler)
    original = "  How are PDFs STORED?  "
    assert await query_expansion.expand_query(original) == [original, *VARIATIONS]
    assert len(requests) == 1
    request = requests[0]
    assert (
        str(request.url)
        == "https://generativelanguage.googleapis.com/v1beta/interactions"
    )
    assert request.headers["x-goog-api-key"] == "gemini-test-key"
    assert "authorization" not in request.headers
    assert request.extensions["timeout"]["read"] == settings.gemini.timeout_seconds
    body = json.loads(request.content)
    assert body["model"] == settings.gemini.model
    assert body["input"] == original
    assert body["store"] is False
    assert body["system_instruction"] == query_expansion._INSTRUCTIONS
    assert body["generation_config"]["thinking_level"] == "low"
    assert body["response_format"]["mime_type"] == "application/json"
    schema = body["response_format"]["schema"]["properties"]["queries"]
    assert schema["minItems"] == schema["maxItems"] == 4


@pytest.mark.anyio
async def test_gemini_function_can_be_called_directly(gemini_transport):
    gemini_transport(lambda request: httpx.Response(200, json=completed_response()))
    assert await query_expansion.expand_query_gemini(ORIGINAL) == [
        ORIGINAL,
        *VARIATIONS,
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("provider", ["openai", "gemini", "claude"])
async def test_dispatch_calls_only_selected_provider(monkeypatch, provider):
    monkeypatch.setattr(settings.query_expansion, "provider", provider)
    generators = {
        name: AsyncMock(return_value=[ORIGINAL, *VARIATIONS])
        for name in ("openai", "gemini", "claude")
    }
    for name, generator in generators.items():
        monkeypatch.setattr(query_expansion, f"expand_query_{name}", generator)
    assert await query_expansion.expand_query(ORIGINAL) == [ORIGINAL, *VARIATIONS]
    for name, generator in generators.items():
        if name == provider:
            generator.assert_awaited_once_with(ORIGINAL)
        else:
            generator.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("key", [None, SecretStr("")])
async def test_gemini_missing_key_fails_without_fallback(
    gemini_transport, monkeypatch, key
):
    def handler(request):
        pytest.fail("No provider should be called without the selected provider key")

    gemini_transport(handler)
    monkeypatch.setattr(settings.gemini, "api_key", key)
    monkeypatch.setattr(settings.openai, "api_key", SecretStr("available-openai-key"))
    with pytest.raises(QueryExpansionError, match="not configured"):
        await query_expansion.expand_query(ORIGINAL)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "queries",
    [
        VARIATIONS[:3],
        [*VARIATIONS, "Extra query"],
        ["  ", *VARIATIONS[1:]],
        [VARIATIONS[0], VARIATIONS[0].upper(), *VARIATIONS[2:]],
        [" HOW  ARE PDFs stored? ", *VARIATIONS[1:]],
        [None, *VARIATIONS[1:]],
    ],
)
async def test_gemini_invalid_variations_are_rejected(gemini_transport, queries):
    gemini_transport(
        lambda request: httpx.Response(200, json=completed_response(queries))
    )
    with pytest.raises(QueryExpansionError, match="unavailable"):
        await query_expansion.expand_query(ORIGINAL)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {"status": "incomplete", "steps": []},
        {"status": "failed", "steps": []},
        {"status": "completed", "steps": []},
        {
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "refusal"}]}],
        },
        {
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "not json"}],
                }
            ],
        },
        {"unexpected": "response"},
    ],
)
async def test_gemini_incomplete_blocked_or_malformed_response(gemini_transport, body):
    gemini_transport(lambda request: httpx.Response(200, json=body))
    with pytest.raises(QueryExpansionError, match="unavailable"):
        await query_expansion.expand_query(ORIGINAL)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "message"),
    [
        (400, "Gemini rejected"),
        (401, "Gemini authentication failed"),
        (403, "Gemini access denied"),
        (404, "Configured Gemini model or endpoint is unavailable"),
        (429, "Gemini query expansion quota or rate limit reached"),
        (503, "Gemini is temporarily overloaded"),
    ],
)
async def test_gemini_http_errors_hide_provider_details(
    gemini_transport, status, message
):
    gemini_transport(
        lambda request: httpx.Response(status, json={"error": "Private details"})
    )
    with pytest.raises(QueryExpansionError, match=f"^{message}") as caught:
        await query_expansion.expand_query(ORIGINAL)
    assert "Private details" not in str(caught.value)


@pytest.mark.anyio
async def test_gemini_timeout_is_a_service_failure(gemini_transport):
    def handler(request):
        raise httpx.ReadTimeout("Private details", request=request)

    gemini_transport(handler)
    with pytest.raises(QueryExpansionError, match="timed out"):
        await query_expansion.expand_query(ORIGINAL)


@pytest.mark.anyio
async def test_search_expands_with_gemini_before_embedding_and_batch_search(
    client, gemini_transport, mock_embeddings, mock_qdrant
):
    def handler(request):
        mock_embeddings.assert_not_called()
        mock_qdrant.query_batch_points.assert_not_awaited()
        return httpx.Response(200, json=completed_response())

    gemini_transport(handler)
    response = await client.post("/rag/search", json={"text": ORIGINAL})
    assert response.status_code == 200
    mock_embeddings.assert_called_once_with([ORIGINAL, *VARIATIONS])
    mock_qdrant.query_batch_points.assert_awaited_once()
    assert len(mock_qdrant.query_batch_points.call_args.kwargs["requests"]) == 5


def test_unknown_provider_is_rejected():
    with pytest.raises(ValidationError):
        QueryExpansionSettings(provider="unsupported")


@pytest.mark.anyio
async def test_temporary_overload_retries_with_backoff_then_succeeds(gemini_transport):
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) < 3:
            return httpx.Response(503, json={"error": {"code": "service_unavailable"}})
        return httpx.Response(200, json=completed_response())

    gemini_transport(handler)
    assert await query_expansion.expand_query(ORIGINAL) == [ORIGINAL, *VARIATIONS]
    assert len(attempts) == 3
    delays = [call.args[0] for call in llm.sleep.await_args_list]
    assert len(delays) == 2
    assert 1 <= delays[0] <= 1.25
    assert 2 <= delays[1] <= 2.25


@pytest.mark.anyio
async def test_overload_retries_stop_at_configured_limit(gemini_transport, monkeypatch):
    attempts = []
    monkeypatch.setattr(settings.gemini, "max_retries", 1)

    def handler(request):
        attempts.append(request)
        return httpx.Response(503, json={"error": {"code": "service_unavailable"}})

    gemini_transport(handler)
    with pytest.raises(QueryExpansionError, match="temporarily overloaded"):
        await query_expansion.expand_query(ORIGINAL)
    assert len(attempts) == 2


@pytest.mark.anyio
async def test_daily_quota_errors_are_not_retried(gemini_transport):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(429, json={"error": {"code": "quota_exceeded"}})

    gemini_transport(handler)
    with pytest.raises(QueryExpansionError, match="quota or rate limit reached"):
        await query_expansion.expand_query(ORIGINAL)
    assert len(attempts) == 1
    llm.sleep.assert_not_awaited()


@pytest.mark.anyio
async def test_overall_deadline_bounds_request_duration(gemini_transport, monkeypatch):
    monkeypatch.setattr(settings.gemini, "timeout_seconds", 0.01)

    async def handler(request):
        await asyncio.Event().wait()

    gemini_transport(handler)
    with pytest.raises(QueryExpansionError, match="timed out"):
        await query_expansion.expand_query(ORIGINAL)


@pytest.mark.anyio
async def test_api_reports_quota_error_before_embedding_or_vector_search(
    client, gemini_transport, mock_embeddings, mock_qdrant
):
    gemini_transport(lambda request: httpx.Response(429, json={"error": "private"}))
    response = await client.post("/rag/search", json={"text": ORIGINAL})
    assert response.status_code == 503
    assert "quota or rate limit reached" in response.json()["message"]
    assert "private" not in response.json()["message"]
    mock_embeddings.assert_not_called()
    mock_qdrant.query_batch_points.assert_not_awaited()
