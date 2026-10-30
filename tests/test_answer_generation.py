import json
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from qdrant_client import models
from qdrant_client.http.models import QueryResponse

from app.config import AnswerGenerationSettings, QueryExpansionSettings, settings
from app.helpers.exceptions import AnswerGenerationError
from app.routers.rag.schemas.api import ChunkSearchResult
from app.routers.rag.services import answer_generation, llm
from app.routers.rag.services import rag as service

ORIGINAL = "Where are PDFs stored?"


@pytest.mark.parametrize(
    "schema, variable",
    [
        (QueryExpansionSettings, "QUERY_EXPANSION_PROVIDER"),
        (AnswerGenerationSettings, "ANSWER_PROVIDER"),
    ],
)
def test_provider_must_be_explicit(schema, variable, monkeypatch):
    monkeypatch.delenv(variable, raising=False)
    with pytest.raises(ValidationError):
        schema(_env_file=None)
    with pytest.raises(ValidationError):
        schema(provider="inherit", _env_file=None)


def chunk(index):
    return ChunkSearchResult(
        text=f"Source text {index}: original PDFs are stored in MinIO.",
        score=0.9 - index * 0.1,
        document_id=42,
        chunk_index=index,
        original_filename="guide.pdf",
        bucket_name="pdfs",
        object_key="guide.pdf",
    )


def provider_response(provider, output, status="completed", refusal=False):
    text = json.dumps(output)
    if provider == "claude":
        return {
            "stop_reason": "refusal"
            if refusal
            else ("end_turn" if status == "completed" else "max_tokens"),
            "content": [{"type": "text", "text": text}],
        }
    if provider == "openai":
        content = (
            [{"type": "refusal"}]
            if refusal
            else [{"type": "output_text", "text": text}]
        )
        return {"status": status, "output": [{"type": "message", "content": content}]}
    content = [{"type": "refusal"}] if refusal else [{"type": "text", "text": text}]
    return {"status": status, "steps": [{"type": "model_output", "content": content}]}


def request_input(body):
    return body["messages"][0]["content"] if "messages" in body else body["input"]


@pytest.fixture(params=["openai", "gemini", "claude"])
def provider(request, monkeypatch):
    name = request.param
    monkeypatch.setattr(settings.query_expansion, "provider", name)
    monkeypatch.setattr(settings.answer, "provider", name)
    monkeypatch.setattr(
        settings.openai,
        "api_key",
        SecretStr("test-openai-key") if name == "openai" else None,
    )
    monkeypatch.setattr(
        settings.gemini,
        "api_key",
        SecretStr("test-gemini-key") if name == "gemini" else None,
    )
    monkeypatch.setattr(
        settings.claude,
        "api_key",
        SecretStr("test-claude-key") if name == "claude" else None,
    )
    monkeypatch.setattr(llm, "sleep", AsyncMock())
    return name


@pytest.fixture
def transport(monkeypatch):
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
async def test_original_question_and_context_go_to_selected_provider(
    provider, transport
):
    requests = []
    chunks = [chunk(0), chunk(1)]

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=provider_response(
                provider,
                {"answer": "Original PDFs are stored in MinIO [2].", "source_ids": [2]},
            ),
        )

    transport(handler)
    result = await answer_generation.generate_rag_answer(ORIGINAL, chunks)

    assert result.answer == "Original PDFs are stored in MinIO [2]."
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.source_id == 2
    assert source.model_dump(exclude={"source_id"}) == chunks[1].model_dump()
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    prompt = json.loads(request_input(body))
    assert prompt["question"] == ORIGINAL
    assert [source["text"] for source in prompt["sources"]] == [
        item.text for item in chunks
    ]
    assert [source["source_id"] for source in prompt["sources"]] == [1, 2]
    assert "bucket_name" not in prompt["sources"][0]
    if provider != "claude":
        assert body["store"] is False
    else:
        assert "store" not in body
    if provider == "openai":
        assert requests[0].headers["authorization"] == "Bearer test-openai-key"
        assert body["instructions"] == answer_generation._INSTRUCTIONS
        assert body["model"] == settings.openai.model
        schema = body["text"]["format"]["schema"]
    elif provider == "gemini":
        assert requests[0].headers["x-goog-api-key"] == "test-gemini-key"
        assert body["system_instruction"] == answer_generation._INSTRUCTIONS
        assert body["model"] == settings.gemini.model
        schema = body["response_format"]["schema"]
    else:
        assert requests[0].headers["authorization"] == "Bearer test-claude-key"
        assert requests[0].headers["anthropic-version"] == settings.claude.api_version
        assert body["system"] == answer_generation._INSTRUCTIONS
        assert body["model"] == settings.claude.model
        schema = body["output_config"]["format"]["schema"]
    assert set(schema["required"]) == {"answer", "source_ids"}


@pytest.mark.anyio
async def test_empty_retrieval_skips_provider_even_without_credentials(monkeypatch):
    openai = AsyncMock()
    gemini = AsyncMock()
    claude = AsyncMock()
    monkeypatch.setattr(answer_generation, "generate_openai_text", openai)
    monkeypatch.setattr(answer_generation, "generate_gemini_text", gemini)
    monkeypatch.setattr(answer_generation, "generate_claude_text", claude)
    result = await answer_generation.generate_rag_answer(ORIGINAL, [])
    assert result.answer == answer_generation.NO_CONTEXT_ANSWER
    assert result.sources == []
    openai.assert_not_awaited()
    gemini.assert_not_awaited()
    claude.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "output",
    [
        {"answer": "", "source_ids": []},
        {"answer": "Bad citation [3].", "source_ids": [3]},
        {"answer": "Duplicate [1].", "source_ids": [1, 1]},
        {"answer": "Missing citation.", "source_ids": [1]},
        {"answer": "Unlisted citation [1].", "source_ids": []},
        {"answer": "Invalid source [0].", "source_ids": [0]},
        {"answer": "Boolean source [1].", "source_ids": [True]},
        {"answer": "Missing source_ids"},
    ],
)
async def test_invalid_answers_and_unknown_citations_are_rejected(
    provider, transport, output
):
    transport(
        lambda request: httpx.Response(200, json=provider_response(provider, output))
    )
    with pytest.raises(AnswerGenerationError, match="could not be validated"):
        await answer_generation.generate_rag_answer(ORIGINAL, [chunk(0), chunk(1)])


@pytest.mark.anyio
async def test_insufficient_evidence_can_return_an_uncited_answer(provider, transport):
    output = {"answer": "The documents do not specify this.", "source_ids": []}
    transport(
        lambda request: httpx.Response(200, json=provider_response(provider, output))
    )
    result = await answer_generation.generate_rag_answer(ORIGINAL, [chunk(0)])
    assert result.answer == output["answer"]
    assert result.sources == []


@pytest.mark.anyio
@pytest.mark.parametrize("refusal", [False, True])
async def test_incomplete_or_refused_answer_is_not_returned(
    provider, transport, refusal
):
    transport(
        lambda request: httpx.Response(
            200,
            json=provider_response(
                provider,
                {"answer": "Partial [1].", "source_ids": [1]},
                status="completed" if refusal else "incomplete",
                refusal=refusal,
            ),
        )
    )
    with pytest.raises(AnswerGenerationError, match="Answer generation is unavailable"):
        await answer_generation.generate_rag_answer(ORIGINAL, [chunk(0)])


@pytest.mark.anyio
@pytest.mark.parametrize(
    "expansion, selected",
    [
        ("gemini", "openai"),
        ("openai", "claude"),
        ("claude", "gemini"),
    ],
)
async def test_answer_provider_can_differ_from_expansion(
    monkeypatch, expansion, selected
):
    monkeypatch.setattr(settings.query_expansion, "provider", expansion)
    monkeypatch.setattr(settings.answer, "provider", selected)
    monkeypatch.setattr(settings.answer, "max_output_tokens", 512)
    generators = {
        name: AsyncMock(
            return_value=json.dumps({"answer": "MinIO [1].", "source_ids": [1]})
        )
        for name in ("openai", "gemini", "claude")
    }
    for name, generator in generators.items():
        monkeypatch.setattr(answer_generation, f"generate_{name}_text", generator)
    await answer_generation.generate_rag_answer(ORIGINAL, [chunk(0)])
    for name, generator in generators.items():
        if name == selected:
            generator.assert_awaited_once()
            assert "model" not in generator.await_args.kwargs
            assert generator.await_args.kwargs["max_output_tokens"] == 512
        else:
            generator.assert_not_awaited()


@pytest.mark.anyio
async def test_endpoint_retrieves_reranks_then_generates_using_only_final_chunks(
    client, provider, transport, monkeypatch, mock_qdrant, mock_reranking
):
    monkeypatch.setattr(
        service, "generate_rag_answer", answer_generation.generate_rag_answer
    )
    mock_qdrant.query_batch_points.return_value = [
        QueryResponse(
            points=[
                models.ScoredPoint(
                    id=index + 1,
                    version=0,
                    score=chunk(index).score,
                    payload=chunk(index).model_dump(exclude={"score"}),
                )
                for index in range(7)
            ]
        )
    ] * 5
    mock_reranking.side_effect = lambda query, chunks: list(reversed(chunks))[:5]
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if provider == "openai":
            schema = body["text"]["format"]["schema"]
        elif provider == "gemini":
            schema = body["response_format"]["schema"]
        else:
            schema = body["output_config"]["format"]["schema"]
        if "queries" in schema["properties"]:
            assert request_input(body) == ORIGINAL
            return httpx.Response(
                200,
                json=provider_response(
                    provider,
                    {
                        "queries": [
                            f"PDF storage variation {index}" for index in range(4)
                        ]
                    },
                ),
            )
        mock_reranking.assert_called_once()
        prompt = json.loads(request_input(body))
        assert prompt["question"] == ORIGINAL
        assert [source["chunk_index"] for source in prompt["sources"]] == [
            4,
            3,
            2,
            1,
            0,
        ]
        return httpx.Response(
            200,
            json=provider_response(
                provider, {"answer": "PDFs are stored in MinIO [2].", "source_ids": [2]}
            ),
        )

    transport(handler)
    response = await client.post("/rag/search", json={"text": ORIGINAL})
    assert response.status_code == 200
    assert len(requests) == 2
    result = response.json()["data"]
    assert result["answer"] == "PDFs are stored in MinIO [2]."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_id"] == 2
    assert result["sources"][0]["chunk_index"] == 3


@pytest.mark.anyio
async def test_answer_failure_is_returned_as_service_error(
    client, provider, transport, monkeypatch
):
    monkeypatch.setattr(
        service, "search_pdf_chunks", AsyncMock(return_value=[chunk(0)])
    )
    monkeypatch.setattr(
        service, "generate_rag_answer", answer_generation.generate_rag_answer
    )
    transport(
        lambda request: httpx.Response(429, json={"error": "private provider details"})
    )
    response = await client.post("/rag/search", json={"text": ORIGINAL})
    assert response.status_code == 503
    assert "private provider details" not in response.text
    if provider in {"gemini", "claude"}:
        assert (
            "answer generation quota or rate limit reached"
            in response.json()["message"]
        )
    else:
        assert (
            response.json()["message"]
            == "Answer generation is unavailable; please retry later"
        )


@pytest.mark.anyio
async def test_missing_answer_credentials_fail_without_request(
    provider, transport, monkeypatch
):
    monkeypatch.setattr(getattr(settings, provider), "api_key", None)

    def handler(request):
        pytest.fail("An answer provider without credentials must not be called")

    transport(handler)
    with pytest.raises(
        AnswerGenerationError, match="Answer generation is not configured"
    ):
        await answer_generation.generate_rag_answer(ORIGINAL, [chunk(0)])


@pytest.mark.anyio
async def test_answer_timeout_is_reported_without_partial_answer(provider, transport):
    def handler(request):
        raise httpx.ReadTimeout("Private connection details", request=request)

    transport(handler)
    with pytest.raises(AnswerGenerationError) as caught:
        await answer_generation.generate_rag_answer(ORIGINAL, [chunk(0)])
    assert "Private connection details" not in str(caught.value)
    assert "answer generation" in str(caught.value).lower()
