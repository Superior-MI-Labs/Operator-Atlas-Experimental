# Operator Atlas — Guide for researchers and coding agents

Read `README.md`, `docs/RESEARCH_DECISION.md`, `docs/ARCHITECTURE.md`, `docs/EXPERIMENT_CATALOG.md`, and `docs/REPRODUCIBILITY.md` before proposing architectural changes.

## Invariants

- **One numerical operator interface:** `atlas/operators.py`. Do not implement a second path for the browser or a shadow simulation that pretends to be the engine.
- **One data owner:** `atlas/memory.py` for evidence and equation provenance. Do not introduce parallel registries without a documented migration.
- **One response authority:** `atlas/responder.py` for response routing, with the same behavior through CLI and HTTP.
- **One diagnostic authority:** `atlas/doctor.py` plus existing test helpers; isolate runtime state and never collect private source contents.
- **The neural model is experimental:** `atlas/model.py` and `atlas/train.py`. Source retrieval is NOT parametric learning; low-rank matrices are NOT arbitrary weight-field invention.
- **The UI is an observer:** `web/` uses the local backend. Animation does not prove precise per-node timing.

## Evidence-first development

1. State a falsifiable hypothesis and the least complex control that could disprove it.
2. Characterize existing behavior in tests before modifying the owner module.
3. Use held-out or independent data. Do not select hyperparameters based on a sealed test score.
4. Measure correctness, task quality, parameter bytes, actual runtime latency and relevant failure rates separately.
5. Preserve failures and negative results, not only peak positive metrics.
6. Update source, tests, `docs/EXPERIMENT_CATALOG.md`, `docs/RESEARCH_DECISION.md`, and the whitepaper when research claims change.
7. Run `bash launch.sh doctor`, inspect its skips, and verify `SHA256SUMS.json` in the release before publication.

## Useful paths

`atlas/` source; `tests/` characterization/regression; `examples/fixtures/` fictional inputs; `data/corpus.txt` authored toy language training; `results/` committed research artifacts; `models/` small experimental checkpoints; `docs/` scientific and API documentation; `tools/package.py` deterministic release filter; `tools/publish_github.sh` conservative Git publishing.

## Refuse overclaims

Do not present constructed symbolic-regression examples as independent discovery of physical laws. Do not call citation retrieval neural learning. Do not describe the toy model as a useful chatbot or assert lower neural inference cost without measured matched-quality throughput. Do not extrapolate qualified input domains in user-facing predictions. Declare unavailable behavior rather than inventing it.

## Repository security

Keep `.venv`, `logs`, `runs`, `candidates`, `data/evidence.sqlite3`, imported documents, and diagnostic ZIPs out of public commits. The local server is not production-hardened. Public CI must not upload private host diagnostics.
