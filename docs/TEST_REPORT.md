## v0.8 verification

50 Python regression tests, including seven new composition/interaction/numerical-bound tests.
The browser smoke suite now exercises learning two CSV equations and composing
them at desktop and mobile sizes; both passed against the actual local API.
The `doctor` report adds `equation_study.json` and composition measurements.
Recorded five-seed grammar ablation is in `results/equation_family_study_v08.json`.
These are bounded synthetic tests, not a validated language model.

# Operator Atlas v0.3 — Verification report

**Scope:** Local container validation, not a claim of user-machine or public-server qualification.

## Regression checks

- 24 Python unit and characterization tests: passed. Existing dense/factorized/sparse/int8 operator and checkpoint tests retained; new tests cover local document decoding, original-input hashes, source deduplication, old SQLite schema migration, explicit v0.2 database import, extractive source answers, sanitized events, immutable experiment files, operator task-loss probing, candidate serialization and corrected launcher selection.
- Chromium browser smoke: passed at desktop 1440×900 and mobile 390×844. Interactions: 13 real graph nodes, small neural generation, imported Markdown source, passage retrieval, research log, genuine model probe. No uncaught page JavaScript exceptions observed.
- API calls executed through a local Python bridge because direct Chromium localhost navigation was blocked by this test environment. The bridge sent actual HTTP calls to the running backend. This is not a full end-to-end browser-origin/security test on Linux Mint.
- `./launch.sh` uses `.venv/bin/python` when executable, otherwise system `python3`. Fixture test confirmed the expected executable routing. v0.2 launcher's bug is fixed in the distributed script.

## Measured v0.3 result (NumPy, dense-control checkpoint, 400-character validation/test partitions)

| Metric | Reference dense | Validation-selected compressed |
|---|---:|---:|
| In-domain validation cross entropy | 0.564711 | 0.566842 |
| Sealed reporting test cross entropy | 0.587021 | 0.586099 |
| Shuffled-character negative-control cross entropy | 8.109100 | 8.099657 |
| All parameter payload saving | — | 55.252% |

One local single-thread CPU timing sample (Python/NumPy recurrent step, six alternating-order passes) measured **0.0070 ms per character** for dense versus **0.0144 ms per character** for int8. This is a hardware- and interpreter-dependent result, not a production runtime speed benchmark.

Every selected operator was **int8**, not factorized. The v0.2 factorized native model still loses to a size-matched dense RNN. This probe uses an authored repetitive corpus, so do not generalize its quality or speed to production language tasks.

## Unverified surfaces

- Large / scanned / encrypted PDF and high-volume corpus ingestion; optional pypdf may not be installed in target runtime.
- Hardware-specific throughput, CUDA kernels, cross-platform UI behavior beyond tested Chromium and Linux.
- Multi-user privacy/isolation, long-running service reliability, large SQLite indexing performance, training from uploaded books, scientific symbolic invention.

## Command record

```
python -m pytest -q tests --ignore=tests/browser_smoke.py
ATLAS_SMOKE_PORT=18765 python tests/browser_smoke.py
./launch.sh diagnose
./launch.sh probe --characters 400 --tolerance 0.05
```

These tests define the current confidence boundary. They do not constitute scientific validation of an efficient new general-purpose neural architecture.


## v0.4 validation

- Existing tests preserved; added stacked checkpoint state/graph/backward compatibility and doctor privacy/integrity tests.
- Pytest regressions: 30 passed on development environment.
- Original default graph browser smoke: desktop and mobile passed against local v0.4 server.
- Shipped two-layer factorized checkpoint: trained and checked PyTorch-to-NumPy parity; `graph(model)` exposes 17 nodes and canonical additional layer edges.
- Multi-seed scientific study: three seeds, fixed 400 steps, same authored corpus, fixed sizes; results in `results/stacked_ablation_v04.json`.
- Diagnostic artifacts: single ZIP includes model/environment summary, sanitized recent events, experiment summary and `doctor.json` results. Active test fixtures run on disposable copy; no user DB imported to that copy.
- Not verified: user laptop GPU performance, general natural-language understanding, full production load, untrusted public hosting.

## v0.9 qualification additions (2026-10-20)

- Added strict sentence-level lexical evidence, preserving `EvidenceStore` as the only evidence owner. Hand-authored fixture: 18/18 expected cited sentences and 10/10 refusals; conflict case preserves both sources. This is not real-world language understanding.
- Added 16-term positive-coordinate mathematical grammar and affine polynomial coordinates; prior persisted equation version compatibility tested.
- Synthetic five-seed log study: mean independent test RMSE 0.204580 (13 terms) versus 0.001012 (16 terms), with the logarithm explicitly in the new vocabulary.
- Added ZIP permission reproducibility regression. `unzip` on Linux must yield executable `launch.sh`; Python `zipfile.extractall` itself does not restore Unix mode bits and is not the supported installation command.
- Source and results version manifest governs release. A clean-extraction doctor is required to qualify this release; check its final output before calling it verified.

### Final local v0.9 candidate regression

57 Python tests passed. Browser smoke passed on 1440×900 and 390×844 viewports, against the actual local API, including imported CSV data, equation composition, evidence answer routing, compression policy and diagnostics. The confirmed independent clean-archive result is recorded in the distributed `doctor` artifact rather than inferred from this narrative.
