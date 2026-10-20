## v0.8 additions

`POST /api/compose_equations` body `{ "equation_ids": [1,2], "x": 0.375 }` returns
`schema, supported, value, initial_x, equation_ids, steps, disclosure`; stopped
chains also return `reason`. Grounded Answer Studio recognizes
`chain equations 1 2 at 0.375`. Persisted equations use
`method_version=compositional-bic-v2`; prior records remain readable.

# Operator Atlas — CLI and Local HTTP Contracts

The GUI reads the server's authoritative graph and operator metadata; graph node dragging does not modify execution.

## CLI

`./launch.sh gui`, `./launch.sh serve [--port 8765] [--model PATH]`, `./launch.sh train [--kind factorized|dense] [--steps N] [--rank R] [--hidden H] [--out PATH]`, `./launch.sh generate 'PROMPT' [--tokens N]`, `./launch.sh simulate`, `./launch.sh diagnose`.

All CLI commands use Python package modules; none call shellout model executables or an external API.

## JSON API (localhost only)

| Method | Route | Request | Result |
|---|---|---|---|
| GET | `/api/status` | — | model identity, operator specifications, training metrics, document count, control metrics |
| GET | `/api/graph` | — | typed visual nodes and edges |
| GET | `/api/operator/{input,recurrent,head}` | — | operator shapes, rank, bytes and sampled weights |
| POST | `/api/generate` | `{"prompt":"The atlas", "max_new":40,"seed":12,"temperature":0.7}` | continuation, per-token probabilities, traced activation vectors |
| POST | `/api/fit_equation` | `{"points":[[0,1],[1,2],...]}` (≥8 finite pairs) | selected grammar terms, fitted coefficients and validation error |
| POST | `/api/ingest` | `{"title":"Book excerpt","text":"..."}` | content hash, source ID and chunk count |
| POST | `/api/search` | `{"query":"matrix"}` | matching source passages with source ID and hash |
| POST | `/api/experiments` | `{"seed":41}` | synthetic representation errors, equation fit and host-local operation latencies |

Failures use JSON `{ "error": "..." }` and non-200 HTTP codes. The HTTP API supports loopback only and is not an authenticated public API.

## Semantic guarantees

`/api/generate` returns a **sampled character continuation**, not a factual or cited answer. `/api/search` returns source text, not a hallucination-free generative response. `/api/fit_equation` searches only a named grammar; it does not learn entirely new function families. `/api/experiments` performs host-local CPU microbenchmarks, not model inference throughput testing.

## Version rules

- Artifacts have `schema: 1` and a content-dependent checkpoint ID.
- Client must treat the server's model artifact and operator metadata as source of truth.
- Model training saves a candidate checkpoint by default. It does not hot-swap or promote. Start `serve --model PATH` to inspect a candidate; restart the server to change a model.
- Source ingestion cannot modify model weights; only explicit offline training can generate a new artifact.

## v0.3 additions (same service, no alternate model or source pipelines)

CLI: `./launch.sh import-file PATH`, `./launch.sh migrate-from PREVIOUS_PROJECT_DIRECTORY`, `./launch.sh logs --limit 40`, `./launch.sh report [--out FILE]`, and `./launch.sh probe --characters 400 --tolerance 0.05 [--write-candidate]`.

| Method | Route | Request | Result |
|---|---|---|---|
| GET | `/api/events` | — | latest 100 sanitized local diagnostic events |
| POST | `/api/ingest_file` | `{ "filename":"book.txt", "base64":"..." }` | source ID, text SHA-256, original SHA-256, chunk count |
| POST | `/api/sources` | `{ "limit": 40 }` | source list and provenance metadata, without document text |
| POST | `/api/ask_sources` | `{ "question":"...", "limit":3 }` | cited lexical passages or abstention |
| POST | `/api/probe` | `{ "characters":400, "allowed_increase":0.05 }` | validation-gated per-layer candidates and held-out loss; saved run ID |

Size controls: general API 250 KB; pasted source text 1.7 MB; file JSON uploads 7 MB (decoded file ≤5 MB, extracted text ≤1.5M characters). Files accepted: `.txt`, `.md`, `.csv`, `.json`, `.html`, `.htm`, `.pdf`; PDF decoding requires optional `pypdf`. Files never undergo implicit execution or model training.

Note: `/api/probe` is for local experimental inspection; computation is CPU-based. Results are not verified general-model acceleration. `--write-candidate` is CLI-only and exports into a new checkpoint directory without mutating `models/starter` or `models/dense-control`.


## v0.4 command additions

- `./launch.sh doctor [--quick] [--out PATH]`: isolated release integrity, operator, API, evidence, equation and trained-checkpoint diagnostics; full mode runs probe and pytest regression. Optional PyTorch and browser tests run when their dependencies are available. The ZIP includes `doctor.json`. Nonzero exit code denotes a failing check, but report generation is attempted regardless.
- `./launch.sh train --layers N`: supports 1–4 recurrent layers, same `Operator.apply` and manifest-based checkpoint reader. Default remains one.
- `./launch.sh study --steps N --seeds S1 S2 ...`: runs/reuses 2-layer fixed comparison checkpoints and records results through the existing experiment authority. Requires optional PyTorch.
- `/api/graph`: number of layer/operator nodes depends on the actually selected checkpoint. Still display-only.

The new model manifests add optional `layers` and training `initializer` fields without changing the existing `schema:1` and storage authority.


## v0.5 probe selection contract

The pre-existing `probe` action and CLI command accept two additional
optional fields: `objective: "storage"|"latency_guarded"` (default `storage`)
and `max_slowdown: float` between 0 and 1 (default `0.05`). The response
adds `decision` and retains previous `operators`, `timing`, and payload
fields. `decision.accepted=false` means actual execution is the original dense
checkpoint; rejected pre-gate options remain in `operators[].pre_guard_selected`
for inspection. No sealed test measurement participates in the choice.
`--write-candidate` refuses to export a no-change fallback.
