"""Offline-by-default evaluation runner for the support-triage service."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, ValidationError

from config import get_settings
from llm.service import ProviderError, ServiceUnavailableError, triage_support_message
from schemas import Category, TriageRequest, TriageResponse, Urgency


DEFAULT_DATASET = Path(__file__).resolve().parent / "cases.jsonl"


class EvaluationError(RuntimeError):
    """Raised when an evaluation dataset or invocation is not usable."""


class EvaluationCase(BaseModel):
    """One labelled, synthetic input row from the JSONL dataset."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    expected_category: Category
    expected_urgency: Urgency


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    expected_category: str
    expected_urgency: str
    predicted_category: str | None
    predicted_urgency: str | None
    status: str


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load strict JSONL data and reuse the API request validation contract."""
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError("Unable to read the evaluation dataset.") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EvaluationError(f"Dataset line {line_number} must not be blank.")
        try:
            case = EvaluationCase.model_validate_json(line)
            TriageRequest(text=case.text)
        except ValidationError as exc:
            raise EvaluationError(f"Dataset line {line_number} is invalid.") from exc
        if not case.id.strip() or case.id in seen_ids:
            raise EvaluationError(f"Dataset line {line_number} has an invalid or duplicate id.")
        seen_ids.add(case.id)
        cases.append(case)

    if not cases:
        raise EvaluationError("The evaluation dataset must contain at least one case.")
    return cases


@contextmanager
def offline_stub_mode() -> Iterator[None]:
    """Force safe stub mode and restore the caller's environment afterward."""
    previous_stub = os.environ.get("LLM_STUB")
    previous_enabled = os.environ.get("LLM_ENABLED")
    os.environ["LLM_STUB"] = "1"
    os.environ["LLM_ENABLED"] = "false"
    try:
        yield
    finally:
        if previous_stub is None:
            os.environ.pop("LLM_STUB", None)
        else:
            os.environ["LLM_STUB"] = previous_stub
        if previous_enabled is None:
            os.environ.pop("LLM_ENABLED", None)
        else:
            os.environ["LLM_ENABLED"] = previous_enabled


def validate_live_configuration() -> None:
    """Fail before evaluation if explicit live-mode requirements are not met."""
    settings = get_settings()
    if settings.llm_stub:
        raise EvaluationError("Live evaluation requires LLM_STUB=0.")
    if not settings.llm_enabled:
        raise EvaluationError("Live evaluation requires LLM_ENABLED=true.")
    if not settings.openrouter_api_key:
        raise EvaluationError("Live evaluation requires OPENROUTER_API_KEY.")


def evaluate_cases(cases: list[EvaluationCase], *, live: bool) -> list[CaseResult]:
    """Run cases through the existing service, never reimplementing triage."""
    if live:
        validate_live_configuration()
        mode = _no_environment_change()
    else:
        mode = offline_stub_mode()

    results: list[CaseResult] = []
    with mode:
        for case in cases:
            try:
                prediction = triage_support_message(TriageRequest(text=case.text))
                if not isinstance(prediction, TriageResponse):
                    raise TypeError("Service returned an invalid prediction object.")
                results.append(
                    CaseResult(
                        case_id=case.id,
                        expected_category=case.expected_category,
                        expected_urgency=case.expected_urgency,
                        predicted_category=prediction.category,
                        predicted_urgency=prediction.urgency,
                        status="ok",
                    )
                )
            except (ProviderError, ServiceUnavailableError, ValidationError, TypeError):
                results.append(
                    CaseResult(
                        case_id=case.id,
                        expected_category=case.expected_category,
                        expected_urgency=case.expected_urgency,
                        predicted_category=None,
                        predicted_urgency=None,
                        status="invalid",
                    )
                )
    return results


@contextmanager
def _no_environment_change() -> Iterator[None]:
    yield


def calculate_metrics(results: list[CaseResult]) -> dict[str, int | float]:
    """Calculate accuracy while treating invalid predictions as failed cases."""
    total = len(results)
    valid = [result for result in results if result.status == "ok"]
    category_correct = sum(
        result.predicted_category == result.expected_category for result in valid
    )
    urgency_correct = sum(
        result.predicted_urgency == result.expected_urgency for result in valid
    )
    combined_correct = sum(
        result.predicted_category == result.expected_category
        and result.predicted_urgency == result.expected_urgency
        for result in valid
    )
    return {
        "total_cases": total,
        "valid_cases": len(valid),
        "failed_or_invalid_cases": total - len(valid),
        "category_accuracy": category_correct / total if total else 0.0,
        "urgency_accuracy": urgency_correct / total if total else 0.0,
        "combined_accuracy": combined_correct / total if total else 0.0,
    }


def print_report(results: list[CaseResult], metrics: dict[str, int | float], *, live: bool) -> None:
    label = "LIVE MODEL EVALUATION" if live else "OFFLINE STUB BASELINE"
    print(label)
    for result in results:
        predicted = (
            f"{result.predicted_category}/{result.predicted_urgency}"
            if result.status == "ok"
            else "invalid"
        )
        print(
            f"{result.case_id}: expected={result.expected_category}/{result.expected_urgency} "
            f"predicted={predicted} status={result.status}"
        )
    print("Summary")
    print(f"total_cases={metrics['total_cases']}")
    print(f"failed_or_invalid_cases={metrics['failed_or_invalid_cases']}")
    for name in ("category_accuracy", "urgency_accuracy", "combined_accuracy"):
        print(f"{name}={metrics[name]:.1%}")


def _write_output(path: Path, results: list[CaseResult], metrics: dict[str, int | float], *, live: bool) -> None:
    payload = {
        "mode": "live" if live else "stub_baseline",
        "metrics": metrics,
        "results": [asdict(result) for result in results],
    }
    try:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise EvaluationError("Unable to write evaluation output.") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate support-triage labels.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--live", action="store_true", help="Allow live OpenRouter requests.")
    parser.add_argument("--output", type=Path, help="Optional JSON results file.")
    args = parser.parse_args()

    try:
        cases = load_cases(args.dataset)
        results = evaluate_cases(cases, live=args.live)
        metrics = calculate_metrics(results)
        print_report(results, metrics, live=args.live)
        if args.output:
            _write_output(args.output, results, metrics, live=args.live)
    except EvaluationError as exc:
        print(f"Evaluation failed: {exc}")
        return 1
    except Exception:
        print("Evaluation failed unexpectedly.")
        return 1

    return 1 if metrics["failed_or_invalid_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
