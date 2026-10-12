"""Configuration loading for the support-triage service.

Values come from environment variables. A local .env file is supported for
development, but it is deliberately excluded from Git.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(ValueError):
    """Raised when an environment setting has an invalid value."""


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean value such as true or false.")


def _read_positive_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc
    if parsed < minimum:
        raise ConfigError(f"{name} must be at least {minimum}.")
    return parsed


@dataclass(frozen=True)
class Settings:
    """Validated application settings with safe defaults for local setup."""

    openrouter_api_key: str | None
    openrouter_base_url: str
    openrouter_model: str
    llm_enabled: bool
    llm_stub: bool
    llm_timeout_seconds: int
    llm_max_retries: int
    app_log_level: str


def get_settings(*, include_api_key: bool = True) -> Settings:
    """Load settings without printing or otherwise exposing secret values.

    A missing API key is allowed here so local stub and disabled modes work
    without secrets. The live-model service must fail safely before it creates
    or calls an LLM client when live access has no configured key.

    Set ``include_api_key`` to False for decisions that do not need the key,
    such as stub mode. This avoids reading the secret unnecessarily.
    """
    load_dotenv()

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    model = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()
    log_level = os.getenv("APP_LOG_LEVEL", "INFO").strip().upper()

    if not base_url:
        raise ConfigError("OPENROUTER_BASE_URL must not be blank.")
    if not model:
        raise ConfigError("OPENROUTER_MODEL must not be blank.")
    if not log_level:
        raise ConfigError("APP_LOG_LEVEL must not be blank.")

    api_key = os.getenv("OPENROUTER_API_KEY") if include_api_key else None
    return Settings(
        openrouter_api_key=api_key.strip() if api_key else None,
        openrouter_base_url=base_url,
        openrouter_model=model,
        llm_enabled=_read_bool("LLM_ENABLED", True),
        llm_stub=_read_bool("LLM_STUB", False),
        llm_timeout_seconds=_read_positive_int("LLM_TIMEOUT_SECONDS", 30),
        llm_max_retries=_read_positive_int("LLM_MAX_RETRIES", 2, minimum=0),
        app_log_level=log_level,
    )
