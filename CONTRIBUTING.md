# Contributing research, code, and falsifications

Operator Atlas is an experimental Superior MI Labs project. We especially value **independent datasets**, **negative reproductions**, **simpler competing models**, **precise bug reports**, and **measurements that contradict published claims**.

Before preparing substantial code work, open an issue describing the hypothesis, target owner module, proposed controls and test plan. Prefer a small characterization test and an update to one existing authority rather than a second execution pipeline.

Run `bash launch.sh doctor` locally, or `python -m pytest -q tests` in a prepared environment. Note any skipped optional checks. Document data provenance, licensing, seed, exact command, platform, machine-independent correctness tests and measurable outcomes. Avoid uploading a raw personal document, local evidence DB, diagnostics with identity information, secrets or unpublished datasets.

Do not treat a source-cited sentence as factually verified, nor a good synthetic curve fit as evidence of forecasting skill. Carefully describe dataset construction, holdouts, failure rates and classical baselines.

**Licensing:** This public project has no declared open-source code license yet. Before accepting third-party code or distributing derivative builds, the maintainer must establish appropriate reuse terms. See `docs/LICENSING.md`.
