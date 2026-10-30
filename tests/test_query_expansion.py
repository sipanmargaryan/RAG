import json

import httpx
import pytest
from pydantic import SecretStr

from app.config import settings
from app.helpers.exceptions import QueryExpansionError
from app.routers.rag.services import llm, query_expansion

ORIGINAL = "How are PDFs stored?"
VARIATIONS = [
    "PDF document storage process",
    "Where are uploaded PDF files saved?",
    "Persistence of processed PDF documents",
    "Storage location for original PDF uploads",
]
QUERIES = [ORIGINAL, *VARIATIONS]


def completed_response(queries=VARIATIONS):
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning"},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": json.dumps({"queries": queries})}
                ],
            },
        ],
    }


@pytest.fixture
def openai_transport(monkeypatch):
    monkeypatch.setattr(settings.query_expansion, "provider", "openai")
    monkeypatch.setattr(settings.openai, "api_key", SecretStr("test-api-key"))
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
async def test_expansion_sends_structured_request_and_returns_five_queries(
    openai_transport,
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completed_response())

    openai_transport(handler)
    assert await query_expansion.expand_query("How are PDFs stored?") == QUERIES

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == "Bearer test-api-key"
    assert request.extensions["timeout"]["read"] == settings.openai.timeout_seconds
    body = json.loads(request.content)
    assert body["input"] == "How are PDFs stored?"
    assert body["model"] == settings.openai.model
    assert body["store"] is False
    format_ = body["text"]["format"]
    assert format_["strict"] is True
    schema = format_["schema"]["properties"]["queries"]
    assert schema["minItems"] == schema["maxItems"] == 4


@pytest.mark.anyio
@pytest.mark.parametrize("key", [None, SecretStr("")])
async def test_missing_key_fails_without_http_request(
    monkeypatch, openai_transport, key
):
    def handler(request):
        pytest.fail("OpenAI must not be called without credentials")

    openai_transport(handler)
    monkeypatch.setattr(settings.openai, "api_key", key)
    with pytest.raises(QueryExpansionError, match="not configured"):
        await query_expansion.expand_query("PDF storage")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "queries",
    [
        VARIATIONS[:3],
        [*VARIATIONS, "Extra query"],
        ["  ", *VARIATIONS[1:]],
        [VARIATIONS[0], VARIATIONS[0].upper(), *VARIATIONS[2:]],
        [None, *VARIATIONS[1:]],
    ],
)
async def test_invalid_expansion_is_rejected(openai_transport, queries):
    openai_transport(
        lambda request: httpx.Response(200, json=completed_response(queries))
    )
    with pytest.raises(QueryExpansionError, match="unavailable"):
        await query_expansion.expand_query("PDF storage")


@pytest.mark.anyio
@pytest.mark.parametrize("repeated", [ORIGINAL, "  HOW  are PDFs STORED?  "])
async def test_generated_query_cannot_repeat_original(openai_transport, repeated):
    openai_transport(
        lambda request: httpx.Response(
            200, json=completed_response([repeated, *VARIATIONS[1:]])
        )
    )
    with pytest.raises(QueryExpansionError, match="unavailable"):
        await query_expansion.expand_query(ORIGINAL)


@pytest.mark.anyio
async def test_original_query_is_preserved_without_rewriting(openai_transport):
    original = "  How are PDFs STORED?  "
    openai_transport(lambda request: httpx.Response(200, json=completed_response()))

    assert await query_expansion.expand_query(original) == [original, *VARIATIONS]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {"status": "incomplete", "output": []},
        {"status": "completed", "output": []},
        {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "refusal"}]}],
        },
        {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "bad"}]}
            ],
        },
        {"unexpected": "response"},
    ],
)
async def test_incomplete_refused_or_malformed_responses_are_rejected(
    openai_transport, body
):
    openai_transport(lambda request: httpx.Response(200, json=body))
    with pytest.raises(QueryExpansionError, match="unavailable"):
        await query_expansion.expand_query("PDF storage")


@pytest.mark.anyio
@pytest.mark.parametrize("status", [401, 429, 500])
async def test_http_errors_hide_provider_details(openai_transport, status):
    openai_transport(
        lambda request: httpx.Response(
            status, json={"error": "Private provider details"}
        )
    )
    with pytest.raises(QueryExpansionError, match="^Query expansion is unavailable"):
        await query_expansion.expand_query("PDF storage")


@pytest.mark.anyio
async def test_openai_timeout_is_reported_as_service_failure(openai_transport):
    def handler(request):
        raise httpx.ReadTimeout("Private connection details", request=request)

    openai_transport(handler)
    with pytest.raises(QueryExpansionError, match="unavailable"):
        await query_expansion.expand_query("PDF storage")


@pytest.mark.anyio
async def test_search_expands_via_openai_before_batch_search(
    client, openai_transport, mock_embeddings, mock_qdrant
):
    def handler(request):
        mock_embeddings.assert_not_called()
        mock_qdrant.query_batch_points.assert_not_awaited()
        return httpx.Response(200, json=completed_response())

    openai_transport(handler)
    vectors = [[float(index)] * settings.embedding.vector_size for index in range(5)]
    mock_embeddings.side_effect = lambda texts: vectors
    response = await client.post("/rag/search", json={"text": "How are PDFs stored?"})

    assert response.status_code == 200
    mock_embeddings.assert_called_once_with(QUERIES)
    mock_qdrant.query_batch_points.assert_awaited_once()
    requests = mock_qdrant.query_batch_points.call_args.kwargs["requests"]
    assert len(requests) == 5
    assert [request.query for request in requests] == vectors


@pytest.mark.anyio
async def test_search_without_key_stops_before_embedding_and_vector_search(
    client, monkeypatch, mock_embeddings, mock_qdrant
):
    monkeypatch.setattr(settings.query_expansion, "provider", "openai")
    monkeypatch.setattr(settings.openai, "api_key", None)
    response = await client.post("/rag/search", json={"text": "PDF storage"})

    assert response.status_code == 503
    assert response.json() == {
        "message": "Query expansion is not configured",
        "data": None,
    }
    mock_embeddings.assert_not_called()
    mock_qdrant.query_batch_points.assert_not_awaited()
