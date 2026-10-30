import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr

from app.config import settings
from app.helpers.exceptions import QueryExpansionError
from app.routers.rag.schemas.answer_generation import GeneratedAnswer
from app.routers.rag.schemas.query_expansion import ExpandedQueries
from app.routers.rag.services import llm, query_expansion

ORIGINAL = "Where are PDFs stored?"
VARIATIONS = [f"PDF storage variation {index}" for index in range(4)]


def completed_response(queries=VARIATIONS, stop_reason="end_turn"):
    return {
        "stop_reason": stop_reason,
        "content": [
            {"type": "thinking", "thinking": "Internal reasoning"},
            {"type": "text", "text": json.dumps({"queries": queries})},
        ],
    }


@pytest.fixture
def claude_transport(monkeypatch):
    monkeypatch.setattr(settings.query_expansion, "provider", "claude")
    monkeypatch.setattr(settings.claude, "api_key", SecretStr("test-claude-key"))
    monkeypatch.setattr(settings.claude, "workspace_id", None)
    monkeypatch.setattr(settings.openai, "api_key", None)
    monkeypatch.setattr(settings.gemini, "api_key", None)
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
async def test_claude_structured_request_and_original_query(claude_transport):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completed_response())

    claude_transport(handler)
    assert await query_expansion.expand_query(ORIGINAL) == [ORIGINAL, *VARIATIONS]
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["authorization"] == "Bearer test-claude-key"
    assert request.headers["anthropic-version"] == settings.claude.api_version
    assert "anthropic-workspace-id" not in request.headers
    assert request.extensions["timeout"]["read"] == settings.claude.timeout_seconds
    body = json.loads(request.content)
    assert body["system"] == query_expansion._INSTRUCTIONS
    assert body["messages"] == [{"role": "user", "content": ORIGINAL}]
    assert body["model"] == settings.claude.model
    assert body["max_tokens"] == 2048
    assert body["output_config"]["format"]["type"] == "json_schema"
    wire = body["output_config"]["format"]["schema"]["properties"]["queries"]
    assert wire["minItems"] == 1
    assert "maxItems" not in wire
    assert "minLength" not in wire["items"]
    assert '"minItems": 4' in wire["description"]
    # Wire adaptation must not weaken the application's validation schema.
    original_schema = ExpandedQueries.model_json_schema()["properties"]["queries"]
    assert original_schema["minItems"] == original_schema["maxItems"] == 4


@pytest.mark.anyio
async def test_direct_claude_function_includes_optional_workspace(
    claude_transport, monkeypatch
):
    monkeypatch.setattr(settings.claude, "workspace_id", "wrkspc_test")

    def handler(request):
        assert request.headers["anthropic-workspace-id"] == "wrkspc_test"
        return httpx.Response(200, json=completed_response())

    claude_transport(handler)
    assert await query_expansion.expand_query_claude(ORIGINAL) == [
        ORIGINAL,
        *VARIATIONS,
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "queries",
    [
        VARIATIONS[:3],
        [*VARIATIONS, "extra"],
        [" ", *VARIATIONS[1:]],
        [VARIATIONS[0], VARIATIONS[0].upper(), *VARIATIONS[2:]],
        [" WHERE are PDFs stored? ", *VARIATIONS[1:]],
        [None, *VARIATIONS[1:]],
    ],
)
async def test_claude_expansion_is_fully_validated_locally(claude_transport, queries):
    claude_transport(
        lambda request: httpx.Response(200, json=completed_response(queries))
    )
    with pytest.raises(QueryExpansionError, match="unavailable"):
        await query_expansion.expand_query(ORIGINAL)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        completed_response(stop_reason="max_tokens"),
        completed_response(stop_reason="refusal"),
        completed_response(stop_reason="tool_use"),
        {**completed_response(), "stop_details": {"type": "refusal"}},
        {"stop_reason": "end_turn", "content": []},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "not JSON"}]},
        {"unexpected": "response"},
    ],
)
async def test_claude_rejects_incomplete_refused_and_malformed_outputs(
    claude_transport, body
):
    claude_transport(lambda request: httpx.Response(200, json=body))
    with pytest.raises(QueryExpansionError, match="unavailable"):
        await query_expansion.expand_query(ORIGINAL)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status, message",
    [
        (400, "Claude rejected"),
        (401, "Claude authentication failed"),
        (403, "Claude access denied"),
        (404, "Configured Claude model or workspace"),
        (429, "Claude query expansion quota or rate limit reached"),
        (529, "Claude is temporarily overloaded"),
    ],
)
async def test_claude_errors_are_client_safe(claude_transport, status, message):
    claude_transport(
        lambda request: httpx.Response(status, json={"error": "private details"})
    )
    with pytest.raises(QueryExpansionError, match=message) as caught:
        await query_expansion.expand_query(ORIGINAL)
    assert "private details" not in str(caught.value)


@pytest.mark.anyio
async def test_claude_retries_transient_overload(claude_transport):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) < 3:
            return httpx.Response(529, json={"error": {"type": "overloaded_error"}})
        return httpx.Response(200, json=completed_response())

    claude_transport(handler)
    assert await query_expansion.expand_query(ORIGINAL) == [ORIGINAL, *VARIATIONS]
    assert len(requests) == 3
    assert llm.sleep.await_count == 2


@pytest.mark.anyio
async def test_claude_does_not_retry_quota_failure(claude_transport):
    claude_transport(lambda request: httpx.Response(429, json={"error": "rate limit"}))
    with pytest.raises(QueryExpansionError, match="quota or rate limit"):
        await query_expansion.expand_query(ORIGINAL)
    llm.sleep.assert_not_awaited()


@pytest.mark.anyio
async def test_claude_overload_retry_limit_is_bounded(claude_transport, monkeypatch):
    monkeypatch.setattr(settings.claude, "max_retries", 1)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(529, json={"error": "overload"})

    claude_transport(handler)
    with pytest.raises(QueryExpansionError, match="temporarily overloaded"):
        await query_expansion.expand_query(ORIGINAL)
    assert len(requests) == 2


@pytest.mark.anyio
async def test_claude_overall_deadline_is_bounded(claude_transport, monkeypatch):
    monkeypatch.setattr(settings.claude, "timeout_seconds", 0.01)

    async def handler(request):
        await asyncio.Event().wait()

    claude_transport(handler)
    with pytest.raises(QueryExpansionError, match="timed out"):
        await query_expansion.expand_query(ORIGINAL)


@pytest.mark.anyio
@pytest.mark.parametrize("key", [None, SecretStr("")])
async def test_claude_missing_credentials_never_fall_back(
    claude_transport, monkeypatch, key
):
    monkeypatch.setattr(settings.claude, "api_key", key)
    monkeypatch.setattr(settings.openai, "api_key", SecretStr("available-other-key"))

    def handler(request):
        pytest.fail("No provider request should be made without Claude credentials")

    claude_transport(handler)
    with pytest.raises(QueryExpansionError, match="not configured"):
        await query_expansion.expand_query(ORIGINAL)


def test_claude_answer_schema_adaptation_keeps_original_validation():
    wire = llm._claude_schema(GeneratedAnswer)
    assert "minLength" not in wire["properties"]["answer"]
    assert "minimum" not in wire["properties"]["source_ids"]["items"]
    original = GeneratedAnswer.model_json_schema()
    assert original["properties"]["answer"]["minLength"] == 1
    assert original["properties"]["source_ids"]["items"]["minimum"] == 1
