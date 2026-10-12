"""Triage service modes, OpenRouter calls, and bounded transport retries."""

from __future__ import annotations

from pathlib import Path
import random
import time

from openai import APIError, APIStatusError, APITimeoutError
from pydantic import ValidationError

from config import Settings, get_settings
from llm.client import create_openrouter_client
from schemas import TriageRequest, TriageResponse


class ServiceUnavailableError(RuntimeError):
    """Raised when the current LLM mode cannot produce a triage result."""


class ProviderError(RuntimeError):
    """Raised for safe-to-report OpenRouter request or response failures."""


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "triage-v1.md"


def _load_system_prompt() -> str:
    """Read the versioned triage instructions from their single source."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def _retry_delay_seconds(retry_number: int) -> float:
    """Return bounded exponential backoff with a small amount of jitter."""
    return min(0.25 * (2**retry_number), 2.0) + random.uniform(0, 0.1)


def _is_retryable(error: APIError) -> bool:
    if isinstance(error, APITimeoutError):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code == 429 or 500 <= error.status_code <= 599
    return False


def _request_completion(client: object, settings: Settings, user_text: str) -> str:
    """Request one completion, retrying only bounded transient failures."""
    messages = [
        {"role": "system", "content": _load_system_prompt()},
        {"role": "user", "content": user_text},
    ]

    for attempt in range(settings.llm_max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=settings.openrouter_model,
                messages=messages,
            )
            try:
                content = completion.choices[0].message.content
            except (AttributeError, IndexError, TypeError) as exc:
                raise ProviderError("LLM provider returned no usable content.") from exc
            if not isinstance(content, str) or not content.strip():
                raise ProviderError("LLM provider returned no usable content.")
            return content
        except ProviderError:
            raise
        except APIError as exc:
            if not _is_retryable(exc) or attempt == settings.llm_max_retries:
                raise ProviderError("LLM provider request failed.") from exc
            time.sleep(_retry_delay_seconds(attempt))

    raise ProviderError("LLM provider request failed.")


def triage_support_message(request: TriageRequest) -> TriageResponse:
    """Triage a validated support message according to the selected mode.

    Stub mode has priority so local deterministic testing remains available
    even when the live LLM kill switch is enabled.
    """
    settings = get_settings(include_api_key=False)

    if settings.llm_stub:
        return TriageResponse(
            category="other",
            urgency="normal",
            confidence=1.0,
            reason="Deterministic stub response for local testing.",
        )

    if not settings.llm_enabled:
        raise ServiceUnavailableError("LLM service is disabled.")

    settings = get_settings()
    if not settings.openrouter_api_key:
        raise ServiceUnavailableError("Live LLM service is not configured.")

    content = _request_completion(
        create_openrouter_client(settings), settings, request.text
    )
    try:
        return TriageResponse.model_validate_json(content)
    except ValidationError as exc:
        raise ProviderError("LLM provider returned invalid structured output.") from exc
