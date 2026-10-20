# Operator Atlas v0.3 — Locked implementation plan (2026-10-19)

## Architectural decision

Continue v0.2 as a research runtime rather than starting a second stack. Preserve `atlas/operators.py`, the trained checkpoint schema, the service endpoint conventions, the SQLite source authority and the browser graph. **Model parameters do not automatically change on document import.** Graph edges are observations, not executable editing authority.

### Compared approaches

| Problem | Option rejected | Chosen approach | Why |
|---|---|---|---|
| File import | Separate document service/database | Format parsing in `atlas/importers.py`, then existing `EvidenceStore.ingest` | One source authority; portable CLI and local UI |
| Diagnostics | User-content logging scattered through handlers | Small structured append-only JSONL event authority, no prompt/content by default | Privacy, reproducibility and single diagnostic surface |
| Compression progress | More synthetic SVD-only plots | Task-level held-out cross-entropy probe on a real shipped checkpoint | Connect compression to prediction behavior |
| Browser UI | Rebuild in frontend framework | Extend the existing thin JS client/graph | No second UI or model authority |
| Model changes | Retrain on every uploaded document | Source-backed search first; explicit future training candidates | Prevent forgetting and fabricated learning claims |

## Files and responsibilities

**Existing preserved:**
- `atlas/operators.py`: only numerical operator implementations.
- `atlas/model.py`: only character recurrent execution; extend with `score_text` delegating to the same `step`.
- `atlas/memory.py`: only document source truth, immutable text and provenance, source index.
- `atlas/selection.py`: existing synthetic selector; retain and test.
- `atlas/discovery.py`: existing bounded symbolic search; retain.
- `atlas/api.py`: instrument actions with named sanitized events; local-only upload and inspection routes.
- `atlas/cli.py`: add import / logs / probe to existing CLI; do not duplicate service logic.
- `atlas/graph.py`: existing graph remains an inspector, never a second compute path.
- `web/index.html`, `web/app.js`, `web/style.css`: input file control, source exploration, probe, run log display; no client-side inference.
- `launch.sh`: always prefer local .venv and support subcommands.

**New justified modules:**
- `atlas/importers.py`: format-specific local file decoding only; strict size/page/character bounds.
- `atlas/diagnostics.py`: one privacy-preserving event log (JSONL) with log tail and immutable experiment summaries.
- `atlas/probe.py`: compare actual language-model validation/test loss for trial operator substitutions; never automatically promote a candidate.

**Documentation:** README, `docs/ARCHITECTURE.md`, `docs/INTERFACES.md`, `docs/RESEARCH.md`, `docs/TEST_REPORT.md`, and continuation checkpoint. Tests cover all new contracts, corrupt files, deduplication, observation privacy, held-out separation, invalid launches and API behavior.

## Real-world workflow

User launches `./launch.sh` → GUI opens → may type text, import `.txt/.md/.csv/.json/.html` (PDF if optional `pypdf` installed), search cited passages, generate from trained toy model, inspect operation graph, evaluate compression on real checkpoint, and inspect/export diagnostics. All imported content remains local; no implicit training or external calls. CLI supports full-size local document import for books up to declared limits.

## Scientific decision gates

1. Task-level model loss must be reported next to actual stored bytes, not inferred from matrix reconstruction error.
2. Compression candidate selection sees validation only, never test data; no automatic promotion.
3. Compare against dense baseline and quantization, not just low-rank rank changes.
4. Limit claims to tiny templated character corpus and local CPU execution.
5. No production deployment claim, general intelligence claim, or unrestricted document understanding.

## Stage gates

Gate A: characterize existing v0.2 behavior before edits.
Gate B: implement canonical file ingestion, logs, task-level probe.
Gate C: integrate CLI and GUI with no independent domain pipelines.
Gate D: unit/integration tests, browser smoke, clean extraction, package and handoff.

The larger research agenda (multi-layer trainable operators, input-distribution-aware structure, stronger language corpora, GPU kernels, multi-seed hardware comparisons) remains a future stage, not silently represented as complete here.

## Implementation audit after integration

All stage gates passed for the v0.3 local research workbench. An additional `report` CLI action was justified by real-world bug-report workflow: it exports just sanitized system summaries, recent events and run metadata without the source DB or text. Existing `atlas/diagnostics.py` implements it; no second logging owner was introduced. Staged candidates remain isolated from the active model.


## v0.4 architectural review — implemented

**Issue: better field learning without duplicate inference architecture.** Compared writing an independent multi-layer engine with extending `CharacterModel.step` and the existing `train_model` checkpoint exporter. Chosen: extend the existing canonical recurrence, preserve legacy one-layer semantics, add optional layer count and additional named operators. Reject new model subsystem and separate GUI executor.

**Issue: poor factorized language results.** Compared increasing rank immediately with correcting effective initialization variance. Chosen: controlled variance-matched initializer (with explicit artifact identity version) before allocating more parameters. Multi-seed study confirms a substantial improvement but not superiority.

**Issue: trouble reports.** Compared a shell script capturing arbitrary machine logs with isolated, typed qualification and the existing report authority. Chosen: `doctor` orchestrator with privacy-bound test workspaces and a `doctor.json` attachment inside the existing report ZIP. The original `report` command remains for non-invasive snapshots.

**Next:** independent licensed, document-disjoint text corpus and stronger baseline/quality measurements. Do not append equation families or proliferate kernels until experiments justify them. Character models remain experimental.
