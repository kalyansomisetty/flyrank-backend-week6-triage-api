"""Stage 2 tests for the local FastAPI input boundary."""

from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_triage_returns_placeholder_response() -> None:
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
