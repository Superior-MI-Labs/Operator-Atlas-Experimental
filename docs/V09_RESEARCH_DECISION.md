# Operator Atlas v0.9 — Research decision record

Date: 2026-10-20. Sponsor: Superior MI Labs. Status: exploratory, not peer reviewed.

## Question

Is it worthwhile to continue building a mathematical-operator architecture that discovers compact representations of observations and neural transformations, potentially replacing some stored weight data?

## Decision

**CONTINUE** developing a general-purpose *research toolkit for bounded equation discovery, compositional operators and source-backed information.* **DO NOT CLAIM** that it is a competitive replacement for contemporary language models, or that the current recurrent decoder understands language.

This is a conditional research decision, not an endorsement of any architecture as optimal. A genuinely competitive native weight-field language model remains **unproven**. Continue such training only under controlled matched-budget studies; keep negative outcomes public.

## Why

1. **Working and reproducible mathematics:** v0.8 fitted equations from CSV and composed source-linked operators, with restricted domains. The v0.9 16-term positive-domain basis learned a deliberately generated logarithmic relationship with mean unseen RMSE 0.001012 vs 0.204580 for the older 13-term family (five seeds). The new affine polynomial coordinate prevents instability on large shifted coordinate values.
2. **Practical near-term application:** source-backed documents allow quotation of facts that weren't available before import. The v0.9 controlled fixture study returned the expected source-bearing sentence for 18/18 hand-authored positive questions and abstained on 10/10 unsupported questions. The older broad chunk search returned misleading matches on all ten negatives. These are in-domain phrase-overlap checks, **not general QA accuracy or linguistic understanding**.
3. **Important negative finding:** factorized character RNNs did not beat near-byte-matched dense RNNs on the tiny authored corpus. Int8 conversion reduced payload but worsened per-character CPU speed in workstation diagnostics. Mathematical fields offer no demonstrated advantage for natural-language generation yet.
4. **Equations cannot eliminate irreducible facts:** storage/retrieval remains necessary for arbitrary events, names, dates, and unknown evidence.

## Alternative strategies compared

**A — One universal generated equation:** Not preferred. Arbitrary data are incompressible, and a coordinate generator can be slower than a compact dense operator. Single global functions create fragility under distribution shift.

**B — Native operator atlas with mathematical families, provenance and explicit fallback (chosen):** Stable typed operators; fit where structure exists, keep factual evidence when it does not, use conventional matrices as fallback. This is falsifiable and has a direct scientific measurement path.

**C — Off-the-shelf LLM or embedding retrieval to produce convincing prose:** Useful for future optional integration, but would obscure what the native operator architecture actually learned and add dependencies. The research demo must keep the character generator and grounded retrieval plainly separate.

## Where the architecture plausibly matters

- Calibration curves, sensors and industrial measurements with physically meaningful coordinates.
- Scientific workflows with small composable empirical transformations and known source documents.
- Data quality research: detect when a regular relationship becomes unstable or the input leaves the observed range.
- Compact edge inference **only if** a native mathematical operator beats quantized dense operators on the target hardware and quality budget.
- Educational demonstrations of factorization, equation fitting, information preservation and inference tradeoffs.

## Why the model still produces gibberish

Its tiny character-level RNN saw a very small repetitive authored corpus. Temperature reshuffles learned character probabilities; it does not supply syntax, factual understanding or reasoning. v0.9 does not change that neural checkpoint and does **not** claim fluent generation. The more readable grounded mode quotes matched source sentences, with provenance; it is not generated neural-language fluency.

## Failure and risk inventory

- Question paraphrases absent from the literal text may be refused. Keyword matches can still be unrelated; a source can itself be false.
- Every lookup currently scans stored source content; this is **not** an indexed large-book search service. For scale, benchmark an indexed retrieval backend under the same `EvidenceStore` authority.
- The added log/square-root/reciprocal terms only apply to all-positive x data; no learned grammar is unrestricted.
- Regressions fit their training region; out-of-domain predictions must remain refused. Validation RMSE is **not** a guaranteed prediction interval.
- Chained equations lack unit/type compatibility and calibrated uncertainty propagation. They are examples, not safe physical automation.
- Training is CPU-first, GPU acceleration not demonstrated. Remote exposure is unsupported without authentication and isolation.
- Positive QA fixture questions were authored to overlap the reference sentences. No claim of blind generalization is valid.

## Publication decision

Ready to release as **an open research software preprint/demo with negative findings clearly stated**. Not ready for a paper claiming a novel or superior general-purpose language model, autonomous scientific law discovery, cognitive reasoning, or production-safe scientific estimation.

## Next hypothesis to falsify

Does a context-aware, layer-specific mathematical basis with *truly stored* residual information beat a parameter/byte/time-matched dense or quantized model on an independent, licensed dataset?

Pre-register the training split, target hardware, comparable total bytes, measured task accuracy, p50/p95 latency, peak RAM/VRAM, seed count and failure threshold. If it does not win, do not scale the weight-generator as an LLM replacement simply because the equations are elegant.
