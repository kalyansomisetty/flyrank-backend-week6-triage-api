"""Safe local storage for model outputs that fail schema validation twice."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("flyrank.llm.quarantine")
QUARANTINE_PATH = Path(__file__).resolve().parent.parent / "logs" / "triage-quarantine.jsonl"
MAX_OUTPUT_CHARS = 4_000


def _truncate(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) <= MAX_OUTPUT_CHARS:
        return value
    return f"{value[:MAX_OUTPUT_CHARS]}...[truncated]"


def quarantine_failed_output(
    *,
    prompt_version: str,
    model: str,
    initial_failure: list[dict[str, Any]],
    initial_output: str,
    repaired_output: str | None = None,
    repair_failure: str | None = None,
) -> bool:
    """Append a bounded JSONL record without ever raising to the API path."""
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "prompt_version": prompt_version,
        "model": model,
        "initial_validation_failure": initial_failure,
        "initial_output": _truncate(initial_output),
        "repaired_output": _truncate(repaired_output),
        "repair_failure": repair_failure,
    }
    try:
        QUARANTINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with QUARANTINE_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except OSError:
        # Logging metadata only keeps a quarantine filesystem failure private.
        LOGGER.warning("Unable to write LLM quarantine record")
        return False
