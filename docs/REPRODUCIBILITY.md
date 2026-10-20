# Reproducing Operator Atlas results

**Scope:** local research environment, not an online service. For a deep protocol and experimental caveats see [`EXPERIMENT_CATALOG.md`](EXPERIMENT_CATALOG.md).

## System prerequisites

Linux, Python 3.10+, a browser, NumPy and pytest. For training studies you need optional PyTorch from `requirements-train.txt`. The default NumPy execution path does not use NVIDIA CUDA. Hardware observations from another machine do not establish equivalent performance on your machine.

```bash
git clone https://github.com/Superior-MI-Labs/Operator-Atlas-Experimental.git
cd Operator-Atlas-Experimental
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash launch.sh doctor
```

The diagnostic command checks source integrity, trained checkpoint compatibility, operator semantics, fixture ingestion, response routing and selected research studies, creates a disposable test sandbox, then writes a sanitized report ZIP to `~/Downloads`. Optional PyTorch/Chromium stages may be skipped and are recorded as such. **A complete regression test suite validates the software; it does not validate scientific generality.**

## Replication sequence

1. **Inspect the static study record first:** `results/historical_sunspots_v1.json`, `results/equation_family_study_v08.json`, `results/coordinate_study_v09.json`, `results/multi_seed_ablation.json`.
2. **Repeat the offline numerical example:** `bash launch.sh demonstrate` runs fictional source ingestion plus synthetic calibration and domain refusals. It does not modify personal evidence state.
3. **Repeat the historical stress comparison:** `bash launch.sh study-real`. Six chronological cutoffs: 1880, 1900, 1920, 1940, 1960, 1980. The subsequent 15 years at each cutoff are held out from fitting and model selection. Atlas is deliberately evaluated outside its normal observed-range qualification, *only as a stress test*, not as a deployable predictor.
4. **Repeat a simple real-world-facing source task:** Query before importing, import `examples/fixtures/evidence_qa.md`, query again. Expected before: explicit abstention. Expected after: source excerpt with the 28-minute fictional interval.
5. **Inspect the neural model rather than assuming language competence:** `bash launch.sh respond --mode experimental 'The mathematical atlas'`. This is a character-level RNN, and outputs often contain gibberish.
6. **Run the visualization:** `bash launch.sh`, visit `http://127.0.0.1:8765`, inspect the Graph, Evidence, Equation Discovery, Answer Studio and Research Bench paths. Graph animation depicts computation *flow*, not measured per-node GPU time.

## File integrity

The ZIP includes `SHA256SUMS.json`; it describes explicitly selected release files and intentionally excludes local `data/evidence.sqlite3`, `logs/`, `.venv/`, transient training candidates, and raw uploaded documents. `tools/publish_github.sh` checks all listed hashes before any GitHub transfer.

For an installed ZIP release:

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
m=json.loads(Path('SHA256SUMS.json').read_text())
bad=[n for n,h in m['files'].items() if not Path(n).is_file() or hashlib.sha256(Path(n).read_bytes()).hexdigest()!=h]
print('OK' if not bad else 'FAILED',len(m['files']),'files',bad[:10])
if bad: raise SystemExit(1)
PY
```

## Privacy and diagnostics

Diagnostic reports intentionally exclude imported source content and the evidence DB, but may include release paths, environment versions and GPU model. Review every ZIP before making it public. Do not commit your local `logs/`, `runs/`, `.venv/`, imported files or database.

For a reproducibility issue, provide the release version, command line, exact failure, synthetic or permitted minimal input, random seed, relevant JSON result, `doctor` summary, and hardware/runtime environment. Avoid uploading private files by default.

## What counts as successful replication?

- Software reproduction: numerical tolerances and regression tests pass on an identified platform.
- Experimental replication: a separately run experiment matches the expected result within documented variance, including negative comparisons.
- General scientific support: **not** established by either of the preceding outcomes. It requires independent, harder datasets and credible task-appropriate controls.
