# Operator Atlas — Superior MI Labs

**Experimental mathematical-learning research system · v1.0.0**

Operator Atlas explores whether useful numerical relationships and some neural transformations can be represented by compact mathematics rather than relying exclusively on large stored parameter matrices. It is a locally runnable Python/NumPy research environment with a detachable, zoomable computation graph.

> **Research status:** Experimental, not peer-reviewed. Not a general-purpose chatbot, not unrestricted equation discovery, and not a demonstrated LLM compression breakthrough. The bundled tiny character neural model often produces gibberish.

## What it does

- Fits equations from numerical observations using a bounded mathematical vocabulary, with validation and recorded input domains.
- Executes dense, factorized, int8, and sparse-corrected operators through one numerical interface.
- Preserves imported source documents and retrieves source-backed excerpts; importing does **not** retrain neural weights.
- Composes previously fitted equations with source identities and stage-by-stage domain checks.
- Provides an interactive node graph, CLI, localhost API, and privacy-conscious `./launch.sh doctor` diagnostics.

## Representative findings

| Test | Result |
|---|---:|
| Synthetic logarithmic data, 13-term grammar | RMSE 0.20458 |
| Synthetic logarithmic data, expanded 16-term grammar | RMSE 0.001012 |
| Near-byte-matched dense character RNN, 3 seeds | Mean validation loss 0.58505 |
| Factorized character RNN, 3 seeds | Mean validation loss 1.01851 |
| Historical solar activity: simple AR(2) control | RMSE 30.91 |
| Historical solar activity: Atlas 16-term equations | RMSE 41.40 |

The synthetic equation result is deliberately favorable. The real-data study uses 309 annual sunspot observations from 1700–2008 and six forward-in-time, 15-year evaluation windows. These windows are correlated; the experiment is **not a qualified forecasting service**.

**Scientific interpretation:** Atlas currently has defensible value as an inspectable mathematical research workbench. Its current mathematical operators have **not** shown better general-language prediction than comparably sized dense neural controls.

## Installation (once full source is synchronized)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./launch.sh doctor
./launch.sh
```

Launch the local graph at http://127.0.0.1:8765. Do not expose the unauthenticated local research server to the internet.

**Repository publication status:** The full verified v1.0.0 source package, trained experimental checkpoints, CSV fixtures, research graphics, whitepaper, and test evidence are being synchronized to this repository. This README alone is not an executable release.

## Verdict and future work

Continue bounded symbolic-regression research, domain-aware computation, scientific units, uncertainty propagation, source provenance, and real-data comparisons. Do **not** invest in scaling a replacement LLM based only on the mathematical elegance of factorized weights. Preserve and publish failures alongside successes.

[Superior MI Labs GitHub](https://github.com/Superior-MI-Labs) · [Interactive mathematical fields exhibit](https://huggingface.co/spaces/Superior-Mind-Labs/mathematical-weight-fields) · [Superior MI Labs Facebook](https://www.facebook.com/profile.php?id=61594542915752)

**License:** Not yet selected by the project owner. Public availability is not automatically an open-source reuse license.