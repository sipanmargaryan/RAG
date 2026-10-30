"""Shared structured-output requests for OpenAI, Gemini, and Claude."""

import asyncio
import json
import logging
from asyncio import sleep
from random import uniform

import httpx
from pydantic import BaseModel

from app.config import settings
from app.helpers.exceptions import ProviderError
from app.routers.rag.schemas.providers import (
    ClaudeResponse,
    GeminiResponse,
    OpenAIResponse,
)

logger = logging.getLogger(__name__)


def _claude_schema(schema: type[BaseModel]) -> dict:
    """Remove unsupported wire constraints; callers validate the original schema."""
    result = schema.model_json_schema()
    unsupported = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "maxItems",
        "uniqueItems",
        "contains",
        "minContains",
        "maxContains",
    }

    def adapt(node: dict) -> None:
        constraints = {key: node.pop(key) for key in unsupported if key in node}
        if node.get("minItems", 0) > 1:
            constraints["minItems"] = node["minItems"]
            node["minItems"] = 1
        if constraints:
            node["description"] = (
                node.get("description", "")
                + " Constraints: "
                + json.dumps(constraints, sort_keys=True)
            ).strip()
        if node.get("type") == "object":
            node["additionalProperties"] = False
        for key in ("properties", "$defs", "definitions"):
            for child in node.get(key, {}).values():
                adapt(child)
        if isinstance(node.get("items"), dict):
            adapt(node["items"])
        for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
            for child in node.get(key, []):
                adapt(child)

    adapt(result)
    return result


async def generate_openai_text(
    text: str,
    instructions: str,
    schema: type[BaseModel],
    *,
    operation: str,
    max_output_tokens: int = 2048,
) -> str:
    if not settings.openai.api_key or not settings.openai.api_key.get_secret_value():
        raise ProviderError(f"{operation.capitalize()} is not configured")

    try:
        async with httpx.AsyncClient(timeout=settings.openai.timeout_seconds) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": (
                        f"Bearer {settings.openai.api_key.get_secret_value()}"
                    ),
                },
                json={
                    "model": settings.openai.model,
                    "instructions": instructions,
                    "input": text,
                    "store": False,
                    "max_output_tokens": max_output_tokens,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": schema.__name__,
                            "strict": True,
                            "schema": schema.model_json_schema(),
                        }
                    },
                },
            )
            response.raise_for_status()
        result = OpenAIResponse.model_validate(response.json())
        if result.status != "completed":
            raise ValueError("Model response did not complete")
        content = [
            part
            for item in result.output
            if item.type == "message"
            for part in item.content
        ]
        if any(part.type == "refusal" for part in content):
            raise ValueError("Model response was refused")
        output_text = "".join(
            part.text for part in content if part.type == "output_text"
        )
        return output_text
    except (httpx.HTTPError, ValueError) as exc:
        # Avoid logging response bodies, search text, or authorization headers.
        logger.warning("OpenAI %s failed (%s)", operation, type(exc).__name__)
        raise ProviderError(
            f"{operation.capitalize()} is unavailable; please retry later"
        ) from exc


async def generate_gemini_text(
    text: str,
    instructions: str,
    schema: type[BaseModel],
    *,
    operation: str,
    max_output_tokens: int = 2048,
) -> str:
    if not settings.gemini.api_key or not settings.gemini.api_key.get_secret_value():
        raise ProviderError(f"{operation.capitalize()} is not configured")

    try:
        async with (
            asyncio.timeout(settings.gemini.timeout_seconds),
            httpx.AsyncClient(timeout=settings.gemini.timeout_seconds) as client,
        ):
            for attempt in range(settings.gemini.max_retries + 1):
                response = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/interactions",
                    headers={
                        "x-goog-api-key": settings.gemini.api_key.get_secret_value()
                    },
                    json={
                        "model": settings.gemini.model,
                        "system_instruction": instructions,
                        "input": text,
                        "store": False,
                        "generation_config": {
                            "max_output_tokens": max_output_tokens,
                            "thinking_level": "low",
                        },
                        "response_format": {
                            "type": "text",
                            "mime_type": "application/json",
                            "schema": schema.model_json_schema(),
                        },
                    },
                )
                if (
                    response.status_code != 503
                    or attempt == settings.gemini.max_retries
                ):
                    break
                logger.warning("Gemini is overloaded; retrying %s", operation)
                await sleep(2**attempt + uniform(0, 0.25))
            response.raise_for_status()
        result = GeminiResponse.model_validate(response.json())
        if result.status != "completed":
            raise ValueError("Model response did not complete")
        content = [
            part
            for step in result.steps
            if step.type == "model_output"
            for part in step.content
        ]
        if any(part.type == "refusal" for part in content):
            raise ValueError("Model response was refused")
        output_text = "".join(part.text for part in content if part.type == "text")
        return output_text
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        logger.warning("Gemini %s failed (HTTP %s)", operation, status_code)
        messages = {
            400: (
                f"Gemini rejected the {operation} request; "
                "check the configured model and request settings"
            ),
            401: "Gemini authentication failed; check GEMINI_API_KEY",
            403: (
                "Gemini access denied; check API key permissions "
                "and your project's free-tier eligibility"
            ),
            404: (
                "Configured Gemini model or endpoint is unavailable; check GEMINI_MODEL"
            ),
            503: "Gemini is temporarily overloaded; please retry later",
            429: (
                f"Gemini {operation} quota or rate limit reached; "
                "check your Google AI Studio limits and retry after the limit resets"
            ),
        }
        raise ProviderError(
            messages.get(
                status_code,
                f"{operation.capitalize()} is unavailable; please retry later",
            )
        ) from exc
    except (httpx.TimeoutException, TimeoutError) as exc:
        logger.warning("Gemini %s timed out (%s)", operation, type(exc).__name__)
        raise ProviderError(
            f"Gemini {operation} timed out; retry or increase GEMINI_TIMEOUT_SECONDS"
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Gemini %s failed (%s)", operation, type(exc).__name__)
        raise ProviderError(
            f"{operation.capitalize()} is unavailable; please retry later"
        ) from exc


async def generate_claude_text(
    text: str,
    instructions: str,
    schema: type[BaseModel],
    *,
    operation: str,
    max_output_tokens: int = 2048,
) -> str:
    if not settings.claude.api_key or not settings.claude.api_key.get_secret_value():
        raise ProviderError(f"{operation.capitalize()} is not configured")

    headers = {
        "Authorization": f"Bearer {settings.claude.api_key.get_secret_value()}",
        "anthropic-version": settings.claude.api_version,
    }
    if settings.claude.workspace_id:
        headers["anthropic-workspace-id"] = settings.claude.workspace_id
    try:
        async with (
            asyncio.timeout(settings.claude.timeout_seconds),
            httpx.AsyncClient(timeout=settings.claude.timeout_seconds) as client,
        ):
            for attempt in range(settings.claude.max_retries + 1):
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json={
                        "model": settings.claude.model,
                        "max_tokens": max_output_tokens,
                        "system": instructions,
                        "messages": [{"role": "user", "content": text}],
                        "output_config": {
                            "format": {
                                "type": "json_schema",
                                "schema": _claude_schema(schema),
                            }
                        },
                    },
                )
                if (
                    response.status_code not in {502, 503, 504, 529}
                    or attempt == settings.claude.max_retries
                ):
                    break
                logger.warning(
                    "Claude is temporarily unavailable; retrying %s", operation
                )
                await sleep(2**attempt + uniform(0, 0.25))
            response.raise_for_status()
        result = ClaudeResponse.model_validate(response.json())
        if result.stop_reason != "end_turn":
            raise ValueError("Claude response was incomplete or refused")
        if result.stop_details and result.stop_details.type == "refusal":
            raise ValueError("Claude response was refused")
        if any(part.type == "refusal" for part in result.content):
            raise ValueError("Claude response was refused")
        output = "".join(part.text for part in result.content if part.type == "text")
        if not output.strip():
            raise ValueError("Claude returned no text")
        return output
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        logger.warning("Claude %s failed (HTTP %s)", operation, status_code)
        messages = {
            400: (
                f"Claude rejected the {operation} request; "
                "check model and workspace settings"
            ),
            401: "Claude authentication failed; check CLAUDE_API_KEY",
            403: "Claude access denied; check API key permissions and model access",
            404: (
                "Configured Claude model or workspace is unavailable; "
                "check Claude settings"
            ),
            429: (
                f"Claude {operation} quota or rate limit reached; "
                "check your Claude API limits"
            ),
            529: "Claude is temporarily overloaded; please retry later",
            503: "Claude is temporarily unavailable; please retry later",
        }
        raise ProviderError(
            messages.get(
                status_code,
                f"{operation.capitalize()} is unavailable; please retry later",
            )
        ) from exc
    except (httpx.TimeoutException, TimeoutError) as exc:
        logger.warning("Claude %s timed out (%s)", operation, type(exc).__name__)
        raise ProviderError(
            f"Claude {operation} timed out; retry or increase CLAUDE_TIMEOUT_SECONDS"
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Claude %s failed (%s)", operation, type(exc).__name__)
        raise ProviderError(
            f"{operation.capitalize()} is unavailable; please retry later"
        ) from exc
