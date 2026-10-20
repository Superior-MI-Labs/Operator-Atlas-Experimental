# Evidence atlas · experiment catalog

**Release:** 1.0.0 · **Classification:** exploratory reproducible research, not peer reviewed.

This catalog distinguishes (a) deterministic software correctness, (b) favorable tests where the data were deliberately chosen to align with the learner, (c) limited authored language comparisons, and (d) an out-of-domain historical stress test. An entry's success is **not** transferable to a different category.

## Reading the evidence

| File | Procedure and sources | Main recorded finding | Limitation |
|---|---|---|---|
| `results/simulation.json` | Synthetic structured, exceptional and random matrices. Operator alternatives evaluated under tolerance constraints. | Compact representations can approximate selected structured operators. | Inputs engineered for diagnostic variety, not representative real networks. |
| `results/multi_seed_ablation.json` | Three seeded character-RNN trainings; same authored corpus and steps; near-matched stored parameter bytes. | Dense: mean validation CE **0.58505**, 21,672 bytes; factorized rank16: **1.01851**, 21,888 bytes. | Tiny repetitive authored corpus; hidden width differs; training FLOPs not equal. |
| `results/stacked_ablation_v04.json` | Three seeds, depth two, improved variance matching; approximately matched budgets. | Dense test CE **0.30033**, factorized **0.36647**. | Separate experiment from the previous row; cannot compare losses directly across protocols. |
| `results/task_level_probe_v0.3.json` | Actual-model validation-only candidate selection, independent test measurement. | Up to ~55% checkpoint operator bytes saved through int8 with little CE change in this small case. | Not full-process peak-memory savings; CPU quantized path was slower. |
| `results/equation_family_study_v08.json` | Five seeded synthetic interaction/polynomial/random controls, 8 vs 13 basis terms, held-out and blocked splits. | Interaction RMSE ~**0.08794 → 0.00168** (lower better). | Generating structure intentionally expressible with expanded grammar. |
| `results/coordinate_study_v09.json` | Five seeded positive-domain logarithmic patterns, 13 vs 16 basis terms, shifted-coordinate test. | Logarithmic holdout RMSE **0.20458 → 0.001012**; shifted polynomial numerics stabilized. | Designed to be representable within 16-term grammar. |
| `results/evidence_quality_study_v09.json` | Fictional source sentences: 18 expected positives, 10 unsupported attributes; chunk-overlap negative control. | **18/18** expected sentence matches; **10/10** abstentions. | Hand-authored fixture, direct lexical cues; not paraphrase or veracity benchmark. |
| `results/public_composition_study_v08.json` | Two synthetic CSV equations, recorded source hashes, staged output-input chaining. | Intermediate and final calculations reproduced with domain checks and provenance. | No physical-unit safety; errors not rigorously propagated through stages. |
| `results/historical_sunspots_v1.json` | Public-domain annual sunspot observations, six chronological training cutoffs; 15 subsequent future years each. | Atlas16 RMSE **41.397293**, AR(2) **30.913529**, mean **39.062339**. | Off-domain forecast stress; temporally correlated windows, weak physical inductive bias. |
| `results/language_probe_v07.json` | 3 seeded character generators at T=0.25,0.75,1.30, with heldout-corpus CE. | Trigram diversity increases with temperature; native prose remains garbled. | Distinct n-grams do not measure coherence or semantic truth. |

The machine-readable files are the primary numerical record. Historical study method is described in [`HISTORICAL_STUDY.md`](HISTORICAL_STUDY.md), the authored-corpus controls in [`RESEARCH.md`](RESEARCH.md), and the overall interpretation in [`WHITEPAPER_V1.md`](WHITEPAPER_V1.md).

## What the measurements mean

**Cross-entropy (CE):** lower next-character negative log likelihood on the specified holdout is better. Corpus/domain differences make absolute CE values noncomparable across studies. Token-level accuracy and fluency are not established.

**RMSE:** root mean squared error over the specified unseen target values; lower is better. A good RMSE on a signal crafted using the learner's functions is *not* an out-of-distribution generalization claim.

**Payload bytes:** bytes to represent a selected mathematical operator's parameters, not peak GPU memory, memory allocator reservations, model metadata or full application footprint.

**Latency:** observed duration of a specified local workload. Small NumPy kernels and CPU int8 dequantization can invert expected theoretical speedups; use hardware-specific measurements with repeats, warmups, and credible controls.

**Evidence match:** an exact or close source-sentence retrieval. The answer is attributable, but a source can still be false or contextually irrelevant.

## Avoided scientific overclaims

- No experimentally supported assertion of discovering general physical laws or world knowledge.
- No observed native neural language fluency; the released trained model is tiny and specialized.
- No independent human-subject evaluation, blind natural-language QA benchmark, or quality-adjusted LLM compression advantage.
- No unbiased proof of forecasting skill from the six overlapping solar periods.
- No validated prediction intervals or physical-unit compatibility for equation chains.
- No automatic retraining from imported documents; evidence storage is a separate learning route.

## Further study specifications

Pre-register datasets, selection rules, budgets, and sealed test partitions before looking at final metrics. For a future mixed-operator experiment report at least: data source and license; total training tokens/rows; all seeds; train FLOPs and wall time; stored bytes and runtime peak memory; p50/p95 latency; comparisons to dense, quantized, and task-appropriate traditional models; failures and abstentions; code+hardware checksum; and uncertainty bounds where valid.

**Useful negative results** are still valid R&D outcomes. If Atlas declines to fit random data or a traditional AR model outperforms symbolic fitting, preserve the result.
