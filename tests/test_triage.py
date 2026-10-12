"""Stage 2 tests for the local FastAPI input boundary."""

from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def default_to_stub_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep endpoint tests deterministic and restore environment changes."""
    monkeypatch.setenv("LLM_STUB", "1")
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_triage_returns_stub_response() -> None:
    response = client.post("/triage", json={"text": "I was charged twice."})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"category", "urgency", "confidence", "reason"}
    assert body["category"] in {"billing", "bug", "feature", "other"}
    assert body["urgency"] in {"low", "normal", "high"}
    assert 0 <= body["confidence"] <= 1
    assert body["reason"].strip()


def test_missing_text_returns_field_focused_400() -> None:
    response = client.post("/triage", json={})

    assert response.status_code == 400
    assert response.json()["detail"][0]["field"] == "text"


def test_non_string_text_returns_field_focused_400() -> None:
    response = client.post("/triage", json={"text": 123})

    assert response.status_code == 400
    assert response.json()["detail"][0]["field"] == "text"


@pytest.mark.parametrize("text", ["", "   \t\n"])
def test_blank_text_returns_400(text: str) -> None:
    response = client.post("/triage", json={"text": text})

    assert response.status_code == 400
    assert response.json()["detail"][0]["field"] == "text"


def test_text_over_2000_characters_returns_field_focused_400() -> None:
    response = client.post("/triage", json={"text": "a" * 2_001})

    assert response.status_code == 400
    assert response.json()["detail"][0]["field"] == "text"


def test_placeholder_endpoint_makes_no_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def network_call_not_allowed(*args: object, **kwargs: object) -> None:
        raise AssertionError("Stage 2 must not make network calls.")

    monkeypatch.setattr(socket, "create_connection", network_call_not_allowed)

    response = client.post("/triage", json={"text": "Just checking in."})

    assert response.status_code == 200


def test_stub_mode_works_without_an_api_key() -> None:
    response = client.post("/triage", json={"text": "No API key is needed."})

    assert response.status_code == 200
    assert response.json()["reason"] == "Deterministic stub response for local testing."


def test_stub_mode_takes_precedence_over_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_STUB", "1")
    monkeypatch.setenv("LLM_ENABLED", "false")

    response = client.post("/triage", json={"text": "Use the stub."})

    assert response.status_code == 200
    assert response.json()["category"] == "other"


def test_disabled_llm_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_STUB", "0")
    monkeypatch.setenv("LLM_ENABLED", "false")

    response = client.post("/triage", json={"text": "Service should be disabled."})

    assert response.status_code == 503
    assert response.json()["detail"] == "LLM service is disabled."


def test_unimplemented_live_llm_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_STUB", "0")
    monkeypatch.setenv("LLM_ENABLED", "true")

    response = client.post("/triage", json={"text": "Live service is unavailable."})

    assert response.status_code == 503
    assert response.json()["detail"] == "Live LLM service is not yet implemented."


@pytest.mark.parametrize(
    ("stub", "enabled", "expected_status"),
    [("1", "true", 200), ("0", "false", 503), ("0", "true", 503)],
)
def test_all_stage_three_modes_make_no_network_calls(
    monkeypatch: pytest.MonkeyPatch, stub: str, enabled: str, expected_status: int
) -> None:
    def network_call_not_allowed(*args: object, **kwargs: object) -> None:
        raise AssertionError("Stage 3 must not make network calls.")

    monkeypatch.setattr(socket, "create_connection", network_call_not_allowed)
    monkeypatch.setenv("LLM_STUB", stub)
    monkeypatch.setenv("LLM_ENABLED", enabled)

    response = client.post("/triage", json={"text": "No network access allowed."})

    assert response.status_code == expected_status
