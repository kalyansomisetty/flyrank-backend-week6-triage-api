"""Tests for the offline-safe evaluation runner."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from evals import run_eval


def test_dataset_has_eight_valid_cases_and_full_label_coverage() -> None:
    cases = run_eval.load_cases(run_eval.DEFAULT_DATASET)

    assert len(cases) == 8
    assert {case.expected_category for case in cases} == {"billing", "bug", "feature", "other"}
    assert {case.expected_urgency for case in cases} == {"low", "normal", "high"}


def test_offline_runner_forces_stub_without_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = run_eval.load_cases(run_eval.DEFAULT_DATASET)

    monkeypatch.setattr(
        "llm.service.create_openrouter_client",
        lambda settings: pytest.fail("Offline evaluation must not create a client."),
    )
    results = run_eval.evaluate_cases(cases, live=False)

    assert len(results) == 8
    assert all(result.status == "ok" for result in results)
    assert {result.predicted_category for result in results} == {"other"}


def test_metrics_handle_controlled_predictions_and_invalid_cases() -> None:
    results = [
        run_eval.CaseResult("one", "billing", "high", "billing", "high", "ok"),
        run_eval.CaseResult("two", "bug", "normal", "bug", "low", "ok"),
        run_eval.CaseResult("three", "other", "low", None, None, "invalid"),
    ]

    metrics = run_eval.calculate_metrics(results)

    assert metrics["category_accuracy"] == pytest.approx(2 / 3)
    assert metrics["urgency_accuracy"] == pytest.approx(1 / 3)
    assert metrics["combined_accuracy"] == pytest.approx(1 / 3)
    assert metrics["failed_or_invalid_cases"] == 1


def test_invalid_dataset_row_fails_clearly(tmp_path: Path) -> None:
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text('{"id":"bad","text":"","expected_category":"unknown"}\n')

    with pytest.raises(run_eval.EvaluationError, match="Dataset line 1 is invalid"):
        run_eval.load_cases(dataset)


def test_invalid_prediction_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = run_eval.load_cases(run_eval.DEFAULT_DATASET)[:1]
    monkeypatch.setattr(run_eval, "triage_support_message", lambda request: object())

    results = run_eval.evaluate_cases(cases, live=False)

    assert results[0].status == "invalid"
    assert run_eval.calculate_metrics(results)["failed_or_invalid_cases"] == 1


def test_live_mode_refuses_missing_configuration_without_service_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = run_eval.load_cases(run_eval.DEFAULT_DATASET)[:1]
    monkeypatch.setenv("LLM_STUB", "0")
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        run_eval,
        "triage_support_message",
        lambda request: pytest.fail("Live preflight must stop before service calls."),
    )

    with pytest.raises(run_eval.EvaluationError, match="OPENROUTER_API_KEY"):
        run_eval.evaluate_cases(cases, live=True)


def test_default_cli_mode_is_offline(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["run_eval.py"])
    monkeypatch.setattr(
        "llm.service.create_openrouter_client",
        lambda settings: pytest.fail("Default CLI mode must not create a client."),
    )

    assert run_eval.main() == 0
    assert "OFFLINE STUB BASELINE" in capsys.readouterr().out
