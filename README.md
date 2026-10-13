# FlyRank Support Message Triage API

A FastAPI backend that classifies synthetic customer-support messages into a
category and urgency level. It demonstrates safe LLM integration patterns with
OpenRouter, schema validation, bounded retries, a deterministic offline mode,
and a reproducible evaluation baseline.

## Features

- `POST /triage` validates input before any service work.
- Closed labels: `billing`, `bug`, `feature`, `other`; and `low`, `normal`,
  `high` urgency.
- OpenRouter integration through the official `openai` Python SDK.
- `LLM_STUB=1` deterministic offline mode and `LLM_ENABLED=false` kill switch.
- A 30-second default request timeout, SDK retries disabled, and bounded custom
  retries for timeouts, HTTP 429, and HTTP 5xx only.
- JSON and Pydantic output validation, followed by exactly one repair request
  for malformed or schema-invalid model output.
- Safe JSONL quarantine records for responses that remain invalid after repair.
- Structured operational logs without API keys, customer text, or raw outputs.

## Project structure

```text
main.py                 FastAPI app and request-validation handler
routes/triage.py        HTTP endpoint and safe error mapping
schemas.py              Request and validated response schemas
config.py               Environment configuration
llm/client.py           OpenRouter SDK client construction
llm/service.py          Mode selection, retries, validation, and repair
llm/quarantine.py       Bounded JSONL failed-output storage
prompts/triage-v1.md    Versioned system prompt
evals/cases.jsonl       Eight synthetic labelled evaluation cases
evals/run_eval.py       Offline-by-default evaluation runner
tests/                  API, resilience, repair, and evaluation tests
```

## Setup

Requires Python 3.11+ and an OpenRouter API key only for live mode.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Keep `.env` local. It is ignored by Git; do not place a real API key in source
code, tests, or committed files.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | empty | Required only for live requests. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL. |
| `OPENROUTER_MODEL` | `openrouter/free` | Model route used in live mode. |
| `LLM_ENABLED` | `true` | Kill switch for all live model calls. |
| `LLM_STUB` | `0` | Deterministic local mode; takes precedence over the kill switch. |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-provider-request timeout. |
| `LLM_MAX_RETRIES` | `2` | Additional transient transport attempts after the initial request. |
| `APP_LOG_LEVEL` | `INFO` | Application logging level. |

## Run the API

For an offline, safe local session:

```bash
LLM_STUB=1 LLM_ENABLED=false uvicorn main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok"}
```

Triage request:

```bash
curl -X POST http://127.0.0.1:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"text":"I was charged twice for my subscription."}'
```

Stub-mode response shape:

```json
{
  "category": "other",
  "urgency": "normal",
  "confidence": 1.0,
  "reason": "Deterministic stub response for local testing."
}
```

Live responses use the same shape but are model-generated and validated.

### Validation and error behavior

`text` is required, must be a non-blank string, and is limited to 2,000
characters. Invalid requests return HTTP `400` with a field-focused JSON
error. Live output that remains invalid after its one repair attempt returns
HTTP `422`. Disabled or missing live configuration returns `503`; controlled
provider failures return `502`.

## Live OpenRouter mode

Set `LLM_STUB=0`, `LLM_ENABLED=true`, and configure `OPENROUTER_API_KEY` only
in your local `.env`. The service uses the configured base URL and model.
Stub mode wins over the kill switch, while `LLM_ENABLED=false` blocks every
live request when stub mode is off.

The SDK retry setting is explicitly `0`. The service retries only timeouts,
HTTP 429, and HTTP 5xx errors, using bounded exponential backoff with jitter.
With `LLM_MAX_RETRIES=2`, there are at most three attempts per provider
request. This transport policy is separate from the single schema-repair
request, which is allowed only after invalid initial model output.

Failed initial-and-repair outputs are written to the ignored local file
`logs/triage-quarantine.jsonl`. Records contain bounded invalid output plus
safe metadata, but not customer input, keys, or HTTP objects. Each provider
attempt emits a structured `llm_call` log event with prompt version, model,
operation, duration, usage when supplied, repair count, and safe outcome data.

## Tests

```bash
source .venv/bin/activate
python -m pytest -q
```

## Evaluation

The eight-case JSONL set covers every category and all urgency values. The
runner prints expected and predicted labels, category accuracy, urgency
accuracy, combined accuracy, and invalid/failed cases.

Offline evaluation is the default and forcibly uses the production stub; it
cannot create an OpenRouter client:

```bash
python -m evals.run_eval
```

Actual offline stub-baseline result from this repository:

| Metric | Result |
| --- | --- |
| Cases | 8 |
| Category accuracy | 25.0% (2/8) |
| Urgency accuracy | 37.5% (3/8) |
| Combined accuracy | 12.5% (1/8) |
| Failed/invalid cases | 0 |

These numbers are a deterministic **stub baseline**, not LLM quality.

To intentionally run a live evaluation—which can consume provider quota—set
`LLM_STUB=0`, `LLM_ENABLED=true`, and a local API key, then run:

```bash
python -m evals.run_eval --live
```

The runner refuses live mode before making a request if the key, kill switch,
or stub setting is unsuitable. To save reproducible, case-level results to an
ignored file:

```bash
python -m evals.run_eval --output evals/results.json
```

### Live model evaluation — October 13, 2026

A live evaluation completed successfully on the same eight synthetic cases.
The cases cover `billing`, `bug`, `feature`, and `other`, plus `low`,
`normal`, and `high` urgency levels.

| Metric | Result |
| --- | --- |
| Sample size | 8 cases |
| Failed/invalid cases | 0 |
| Category accuracy | 100.0% (8/8) |
| Urgency accuracy | 50.0% (4/8) |
| Combined category-plus-urgency accuracy | 50.0% (4/8) |

Category classification was correct for all eight cases in this run. Urgency
classification needs improvement. This is an initial benchmark on a small,
synthetic dataset; it is not a production-quality performance claim and does
not establish statistical significance, generalisation, or guaranteed future
accuracy.

## Cost estimate

The live evaluation did not record token totals or measured cost. For a chosen
live model, estimate each request as:

```text
(input_tokens / 1,000,000 × input_price_per_million)
+ (output_tokens / 1,000,000 × output_price_per_million)
```

Multiply by the number of initial requests, plus occasional repair requests
and retry attempts. The `openrouter/free` route may have zero listed price at
a given time, but availability, routing, limits, and terms can change; it is
not a guarantee of uninterrupted free service. Check the selected model's
current OpenRouter pricing before budgeting.

## Limitations and next steps

- Stub-baseline scores are not representative of live model performance.
- Reason quality is constrained by prompt and length checks; one-sentence
  semantics are not fully machine-enforced.
- Quarantine records can contain bounded model output, so protect the local
  ignored `logs/` directory in operational environments.
- Add a larger, reviewed evaluation set before making production quality
  claims.

## Author
Venkata Naga Sri Kalyan Somisetty

Backend AI Engineer Intern – FlyRank AI