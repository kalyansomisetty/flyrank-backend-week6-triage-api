"""OpenRouter client construction kept outside the HTTP layer."""

from __future__ import annotations

from openai import OpenAI

from config import Settings


def create_openrouter_client(settings: Settings) -> OpenAI:
    """Create an OpenAI-compatible client with explicit request behavior.

    Callers must check that an API key is configured before calling this
    function. SDK retries are disabled because the service owns retry policy.
    """
    if not settings.openrouter_api_key:
        raise ValueError("An OpenRouter API key is required for live mode.")

    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )
