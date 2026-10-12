"""Stage 2 tests for the local FastAPI input boundary."""

from __future__ import annotations

import socket
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIStatusError, APITimeoutError

from main import app
from llm import service


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


def test_live_llm_without_api_key_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_STUB", "0")
    monkeypatch.setenv("LLM_ENABLED", "true")

    response = client.post("/triage", json={"text": "Live service is unavailable."})

    assert response.status_code == 503
    assert response.json()["detail"] == "Live LLM service is not configured."


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


class FakeClient:
    """Small fake for the OpenAI chat-completions interface used by the service."""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = iter(outcomes)
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))]
        )


def _status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return APIStatusError("provider error", response=response, body=None)


def _timeout_error() -> APITimeoutError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return APITimeoutError(request=request)


@pytest.fixture
def live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_STUB", "0")
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-a-real-secret")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")


def _use_fake_client(monkeypatch: pytest.MonkeyPatch, fake_client: FakeClient) -> None:
    monkeypatch.setattr(service, "create_openrouter_client", lambda settings: fake_client)
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(service, "_retry_delay_seconds", lambda retry_number: 0.0)


def test_stub_mode_does_not_instantiate_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def client_must_not_be_created(*args: object, **kwargs: object) -> object:
        raise AssertionError("Stub mode must not create a client.")

    monkeypatch.setattr(service, "create_openrouter_client", client_must_not_be_created)

    response = client.post("/triage", json={"text": "Stay deterministic."})

    assert response.status_code == 200


def test_disabled_mode_does_not_instantiate_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_STUB", "0")
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setattr(
        service,
        "create_openrouter_client",
        lambda settings: pytest.fail("Disabled mode must not create a client."),
    )

    response = client.post("/triage", json={"text": "Stay disabled."})

    assert response.status_code == 503


def test_live_mode_without_key_does_not_instantiate_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_STUB", "0")
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        service,
        "create_openrouter_client",
        lambda settings: pytest.fail("A missing key must prevent client creation."),
    )

    response = client.post("/triage", json={"text": "No key is configured."})

    assert response.status_code == 503


def test_mocked_openrouter_response_returns_valid_schema(
    monkeypatch: pytest.MonkeyPatch, live_mode: None
) -> None:
    fake_client = FakeClient(
        [
            '{"category":"billing","urgency":"high",'
            '"confidence":0.96,"reason":"The customer reports a duplicate charge."}'
        ]
    )
    _use_fake_client(monkeypatch, fake_client)

    response = client.post("/triage", json={"text": "I was charged twice."})

    assert response.status_code == 200
    assert response.json()["category"] == "billing"
    assert response.json()["urgency"] == "high"
    assert fake_client.calls == 1


@pytest.mark.parametrize("error_factory", [_timeout_error, lambda: _status_error(429), lambda: _status_error(500)])
def test_transient_provider_failures_are_retried(
    monkeypatch: pytest.MonkeyPatch, live_mode: None, error_factory: object
) -> None:
    fake_client = FakeClient(
        [
            error_factory(),
            '{"category":"bug","urgency":"normal",'
            '"confidence":0.8,"reason":"The feature is not working."}',
        ]
    )
    _use_fake_client(monkeypatch, fake_client)

    response = client.post("/triage", json={"text": "It is broken."})

    assert response.status_code == 200
    assert fake_client.calls == 2


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_client_errors_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, live_mode: None, status_code: int
) -> None:
    fake_client = FakeClient([_status_error(status_code)])
    _use_fake_client(monkeypatch, fake_client)

    response = client.post("/triage", json={"text": "Do not retry client errors."})

    assert response.status_code == 502
    assert fake_client.calls == 1


def test_retries_stop_after_configured_maximum(
    monkeypatch: pytest.MonkeyPatch, live_mode: None
) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    fake_client = FakeClient([_status_error(503), _status_error(503), _status_error(503)])
    _use_fake_client(monkeypatch, fake_client)

    response = client.post("/triage", json={"text": "Retry only twice."})

    assert response.status_code == 502
    assert fake_client.calls == 3


def test_malformed_model_output_returns_controlled_error(
    monkeypatch: pytest.MonkeyPatch, live_mode: None
) -> None:
    fake_client = FakeClient(["not valid JSON"])
    _use_fake_client(monkeypatch, fake_client)

    response = client.post("/triage", json={"text": "Return malformed output."})

    assert response.status_code == 502
    assert response.json()["detail"] == "LLM provider returned invalid structured output."


def test_missing_model_content_returns_controlled_error(
    monkeypatch: pytest.MonkeyPatch, live_mode: None
) -> None:
    fake_client = FakeClient([None])
    _use_fake_client(monkeypatch, fake_client)

    response = client.post("/triage", json={"text": "Return no content."})

    assert response.status_code == 502
    assert response.json()["detail"] == "LLM provider returned no usable content."
