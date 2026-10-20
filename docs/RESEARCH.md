# v0.8 scientific record — richer bounded equation structures

This is an independent test of **grammar expressiveness**, not a general AI
benchmark. Five deterministic seeds, 110 fit observations and 80 separate
held-out coordinates per dataset, with three dataset families: a wave
interaction deliberately constructed from an expanded basis, an ordinary
polynomial control, and independent random targets.

| Task | 8-term test RMSE | 13-term test RMSE | Finding |
|---|---:|---:|---|
| Interaction | 0.087939 | 0.001678 | Additional term matches construction |
| Polynomial | 0.001602 | 0.001602 | No gain |
| Unstructured | 0.956902 | 0.963586 | No improvement; slight regression |

We retain the raw per-seed predictions/metrics in
`results/equation_family_study_v08.json`. The v0.8 doctor attaches this exact
study as `equation_study.json`, plus the separate five-source fixture run as
`fixture_demo.json`. Both contain only synthetic data. The fixture composition
uses two independently learned source-backed equations and refuses a new x
coordinate when a stage exceeds its observed domain.

**Scientific caveat:** The interaction example was constructed using the
learned grammar. This is a capability/implementation characterization, not a
new theorem about real data, a physical law, or evidence that neural model
text becomes more fluent. Grammar selection uses a seeded train/validation
split. The blocked-coordinate holdout is diagnostic only. Composition
uncertainty and unit compatibility remain unvalidated.

**User workstation (2026-10-20 diagnostic):** v0.7 ran 10/12 checks, with 0
failures and 2 optional skips (torch, automated Chromium); NumPy 2.5.4 on
Python 3.12, detected RTX 3080 Laptop 16GB. The neural character generator
still exhibits gibberish. Compact int8 operators remain slower than dense in
that small CPU runtime test. These negative results are not overridden.

---

# Operator Atlas — Empirical Research Record

**Hypothesis:** Under controlled information and computation budgets, some neural operations can be represented more efficiently by learned mathematical structures than by independent dense weights. This is conditional; random information and distribution shift are potential failure cases.

## What v0.2 actually establishes

1. Mathematically equivalent operators execute from dense, int8, low-rank or genuinely sparse-corrected representations under a shared NumPy contract, with reproducibility tests.
2. A small recurrent neural model was actually trained on a small authored corpus; the exact exported weight arrays produce PyTorch/NumPy inference parity to within approximately 1e-5 in the shipped checkpoint.
3. A bounded symbolic regression grammar correctly fits included synthetic patterns. It is a grammar-selection and coefficient-fitting demonstration, **not autonomous discovery of unknown mathematical laws**.
4. A validation-constrained selector can favor compact representations where they meet numerical error budgets and retain dense storage for strict-error random matrices.
5. A local numerical microbenchmark measures compact versus dense operations, but its findings **do not predict LLM token speed**.
6. Ingested information is retained in a content-hashed source store and retrieved by lexical matches; it is **not** incorporated into the trained weights.

## Multi-seed near-equal-storage neural comparison (seeds 7, 11, 19; 700 steps)

These architectures use different hidden sizes to closely match parameter bytes; they share corpus, sequence length, optimizer family and 700 steps. The dense control was smaller in payload and consistently achieved lower validation loss. This is a **negative result for superiority claims**, not evidence that every factorized network is worse. All runs are in `models/ablations` with machine-generated details in `results/multi_seed_ablation.json`.

The exact machine-generated metrics are in `results/factorized_training.json` and `results/dense_training.json`. Expected shipped results:

| Model | Parameter payload | Held-out character cross entropy |
|---|---:|---:|
| Rank-16 factorized (hidden 48; 3-seed mean ± SD) | 21,888 bytes | 1.0185 ± 0.1371 |
| Dense control (hidden 38; 3-seed mean ± SD) | 21,672 bytes | 0.5851 ± 0.0539 |

An additional narrower dense hidden-33 control (18,792 bytes) from seed 7 yielded validation loss 0.7433. The shipped interactive starter is factorized hidden-48 seed 7 (loss 0.8881). All metrics are held-out character loss on the authored demonstration corpus, not open-domain reasoning scores.

The split is positional and corpus sentences repeat synthetic templates. Validation likely shares most grammar structures with training. We cannot claim broad language generalization, real-world understanding, or improved efficiency from these results.

## Explicit experiments and falsification gates

- **Operator correctness:** randomized parity between direct factor application and dense equivalent; reject mismatched dimensions and invalid sparse indices.
- **Representation selection:** choose on validation, record held-out shifted-input errors separately, measure actual payload including sparse rows/columns.
- **Equation discovery:** compare to mean-only predictor and test on random/noisy data; fitted grammatical equations may extrapolate incorrectly.
- **Train/infer agreement:** compare torch and NumPy logits for same input; fail checkpoint export when max difference >1e-4.
- **Hardware:** measure warm median latency for the actual NumPy application paths, not just theoretical flops or simulated GPU bandwidth.
- **Language model:** compare held-out next-character loss of trained factorized vs smaller dense model using controlled tokenization and corpus.

## Key negative results

The dense neural control demonstrates **better** loss with a smaller payload. The symbolic discovery grammar only discovers expressions made from supplied functions. No multi-layer language model with automatic operator selection has been trained. The program cannot infer new arbitrary facts from compression alone. Fitting useful behavior on a narrow input distribution says little about shifted distributions.

## Prior work to cite in any public scientific manuscript

- Ha, Dai and Le (2016), HyperNetworks — https://arxiv.org/abs/1609.09106
- Ashkenazi et al. (2022/2023), NeRN — https://arxiv.org/abs/2212.13554
- Yuan et al. (2023), ASVD — https://arxiv.org/abs/2312.05821
- Wang et al. (2025), BitNet — https://jmlr.org/papers/v26/24-2050.html
- Udrescu and Tegmark (2020), AI Feynman — https://pmc.ncbi.nlm.nih.gov/articles/PMC7159912/

## Publication status

This is an **exploratory implementation and research log**, not a peer-reviewed paper. Novelty must be established through literature review and comparisons with optimized quantized, low-rank, and activation-aware baselines using real language benchmarks at equal compute/storage budgets. The name Operator Atlas is a working title, not an originality claim.

---

## v0.3 empirical addition — task-level compression instead of matrix error alone

**Design change:** Use the shipped dense-control RNN's real next-character cross entropy as the compression qualification metric. Hold out a final partition that is never used to choose operators. Try dense, per-tensor int8, and fixed low-rank factors for each operator. Lower payload is preferred only while cumulative model validation loss stays within 0.05 nats of the unmodified checkpoint.

**Observed example** (`./launch.sh probe --characters 400 --tolerance 0.05`, local NumPy):

| Metric | Dense reference | Validation-selected candidate |
|---|---:|---:|
| Validation next-character cross entropy | 0.56471 | 0.56684 |
| Reporting-only test cross entropy | 0.58702 | 0.58610 |
| Parameter payload reduction | — | 55.25% |
| Selected operator family | Dense | int8 for input, recurrent, head |
| Shuffled-character negative control cross entropy | 8.10910 | 8.09966 |

A local Python/NumPy microbenchmark also observed slower per-character runtime after quantization (approximately 0.0144 ms vs 0.0070 ms for the reference). This is a six-pass CPU timing sample with Python overhead, not a statistically powered production benchmark.

This **does not validate the central weight-field superiority hypothesis**: the selected improvement came from simple int8 quantization, and the selected representation could have worse real inference latency. The tiny, highly repetitive corpus is particularly poor evidence of broad language behavior. A shuffled-text negative control is not an English out-of-distribution test.

The v0.2 three-seed training comparison remains the strongest evidence on *native* learning: factorized 1.0185 vs byte-matched dense 0.5851 validation cross-entropy (lower is better). This indicates the next scientific bottleneck is training quality, not more graph features.

**Next falsifiable research:** multi-layer causal language models on larger licensed corpora with clean document-disjoint splits, equal-parameter and equal-byte controls, quantized and factorized baselines, measured CPU/GPU latency, varied rank per layer and (only if justified) selective residual training. Proposed methods should be rejected when they fail to beat the strongest compact baseline at matched quality and hardware cost.


## v0.4 — Stacked recurrence and initialization controlled experiment

The newly supported two-layer recurrent model was trained for 400 steps on the existing authored synthetic corpus. Its mathematical operators are **trained directly** and executed without materializing the dense weight matrices. Matched-budget controls use hidden=38 dense vs hidden=48, rank=16 factorized, with approximately 33,224 and 34,176 FP32 payload bytes respectively (factorized is ~2.9% *larger*).

A three-seed study (seeds 3,7,11) using variance-matched factor initialization produced these *reporting-only* held-out next-character cross-entropy losses (lower is better):

| Representation | Mean CE | Std dev | Payload |
|---|---:|---:|---:|
| Two-layer dense control | 0.30033 | 0.05040 | 33,224 B |
| Two-layer factorized | 0.36647 | 0.04517 | 34,176 B |

A preliminary two-seed study using legacy small-factor initial scales yielded factorized mean held-out CE 2.17858 vs dense 0.31157. The large improvement following variance-matched initialization is evidence that **optimizer and initialization design matter** before we attribute poor results to representation capacity. It is not an architecture-level superiority result; the dense control remains better at this budget. The legacy-vs-new comparison is exploratory and was not a prespecified confirmatory study.

These tests reuse the v0.3 templated corpus. Samples in training, validation and reporting holdouts share vocabulary and repeated linguistic patterns, so results may overestimate generalization. No new corpus knowledge, real reasoning, token-level Transformer, cross-device performance or self-invented mathematical law has been established. A change to initializer gets its own checkpoint identity.

See `results/stacked_ablation_v04.json`, original checkpoints under `models/stacked-*`, `atlas/study.py` and `docs/CONTINUATION.md`. The v0.3 quantization probe is preserved as a separate scientific finding.


## v0.5 — Cost-aware candidate selection (2026-10-20)

A user-uploaded v0.4 Doctor report contained 7 PASS, 0 FAIL, 2 SKIP.
The optional training and browser checks were SKIP for missing PyTorch and
Playwright/Chromium. All 30 installed Python tests passed. The loaded checkpoint
was the original one-layer factorized model, not the two-layer candidate.

On that workstation, v0.4's 100-character compression probe measured:
- Dense payload: 18,792 B; 0.01674 ms/character.
- Selected int8 payload: 8,409 B; 0.02629 ms/character.
- Memory payload reduction: 55.25%; **latency regression:** 57.05%.

This prompted an explicit selection-policy experiment, **not** a claim of new
intelligence capability. The v0.5 probe first chooses validation-qualified
operators using the existing greedy stored-byte objective. It then benchmarks
complete model execution on a validation-only token sequence (10 alternating-
order passes) and, in `latency_guarded` mode, rejects that *whole candidate*
if median runtime exceeds the baseline by the user-selected tolerance.
Rejection restores the exact source operator set; no candidate is automatically
promoted. `storage` mode preserves the previous results. The sealed test remains
unobserved until the plan has been chosen. All payload sizes exclude allocator
and runtime overhead. Any benefit is host-specific and must be retested.

A locally executed v0.5 example with 120 characters per held-out slice found
that the storage-oriented int8 candidate ran about 1.7× slower than dense; the
5% latency-guarded policy reverted to dense (zero payload savings, no measured
inference regression). This negative result is expected to depend on NumPy,
CPU configuration, batch size and measurement noise. The right next scientific
experiment is to compare genuinely optimized direct-execution operators, not to
hide slow kernels with more routing logic.

## v0.6 — Actual imported-document functional learning

A new public synthetic experiment uses the **exact same ingestion, SQLite,
search, equation fitting, prediction and rejection code as the GUI and CLI**.
This is not a precomputed slide or fictional response engine.

The test corpus comprises four explicitly fictional/synthetic files shipped in
`examples/fixtures/`: a named source record, an unrelated negative control,
81 sampled wave/curve observations with deterministic noise, and 81 seeded
random observations. The numeric target is intentionally generated from the
learner's available basis grammar; this is a favorable sanity check.

Running `./launch.sh demonstrate` produced:

- A query for the fictional reference prism has **zero matches before import**
  and retrieves the correct source after import. The search is lexical and
  still cannot establish scientific truth.
- Restricted equation selection found `1`, `x²`, and `sin(2x)` terms. Its
  held-out validation RMSE was approximately **0.002385**, compared with
  **1.597425** for the training-mean baseline.
- An unobserved in-domain synthetic coordinate (x = 0.375) was predicted with
  absolute error approximately **0.000181** against the known generating law.
- A query far outside the observed range was refused. A random numerical CSV
  was also rejected by the fixed held-out improvement criterion.

All values are stored in `results/public_fixture_study_v06.json` and are
recalculated under the disposable `doctor` test. The equation search does not
synthesize new neural representations, and this demonstration does **not**
prove language model quality, realistic scientific generalization, permanent
memory savings, or overall response-speed improvement. It provides a concrete
use case for separating uncompressible source evidence from compressible
numerical relationships.

### Follow-up falsification studies

- Evaluate sensor signals whose generating law is *not* in the eight-term
  grammar; compare failure rate under distribution shift and noise.
- Test missing samples, contradictory measurement labels, drifting processes,
  mixed physical units and correlations that disappear over time.
- Compare a compact equation with nearest-neighbor interpolation and splines
  at equal byte budgets; account for raw-source retention and indexing.
- Establish calibration and uncertainty bounds before allowing any use in
  safety-relevant contexts.
- Train learned operator families only after qualification of simpler
  representations; preserve independent held-out model loss and latency gates.


## v0.7 — The character-model failure is a real result

The user reported malformed generated text for "The mathematical atlas".
Using the unchanged starter checkpoint, we reproduced its continuation.
This is expected given character-level autoregression over a small, repetitive
hand-authored corpus. Temperature affects sampling entropy, not knowledge.

Local 3-seed, 96-character-per-seed illustrative decoding diagnostics:

| Temperature | Unique characters | Distinct trigram fraction |
|---|---:|---:|
| 0.25 | 22 | 0.38462 |
| 0.75 | 26 | 0.74126 |
| 1.30 | 30 | 0.84965 |

For a shipped held-out suffix from the same authored corpus, checkpoint
cross-entropy was 0.822275 nats/character (perplexity about 2.276). This is
*not* an independent broad-language evaluation. Different character variety
is not necessarily better prose.

The source-grounded response fixture demonstrates an actual, narrower gain:
no imported evidence => abstain; imported fictional notebook => retrieve a
passage containing cabinet KAPPA-47, with content SHA and source ID. Qualified
CSV equation evaluation has a separate computation route, validation metric,
and out-of-range refusal. These mechanics do not imply the neural weights
learned the document or discovered unrestricted equations.

**Research judgment:** Our most immediate error was presenting a small
character-completion network in an interface that invited expectations of
conversation. The system should expose the limitation and separate scientific
mechanisms rather than hide it. The next substantive neural research must
improve data and architecture, not tune temperature for better-looking text.


## v0.9 — Explicit evaluation and continued-research decision

Five synthetic logarithmic runs: legacy 13-term family mean unseen RMSE **0.204580**; new 16-term positive-domain family mean **0.001012**. Shifted 12,000–12,200 x-value quadratic probe max absolute error **0** (constructed polynomial, fitting in normalized coordinates). Results: `results/coordinate_study_v09.json`. The true log term was included in the candidate grammar; this is not discovery of an unknown law.

Source-response trial used 3 fictional packaged documents and 18 hand-authored questions designed to overlap relevant sentences. **18/18** expected source-bearing responses; **10/10** unsupported-attribute questions abstained; both conflicting accounts were displayed. The older chunk search returned matches for **all ten** unsupported questions, despite insufficient evidence to answer them. Results: `results/evidence_quality_study_v09.json`. This illustrates conservative lexical filtering, not general language comprehension.

The original neural character model, dense/factorized ablations and quantized CPU latency remain unchanged, with no claim of improved language fluency. Current decision: **pursue the mathematical source/equation laboratory; do not portray the learned operator as a competitive LLM**. See `docs/V09_RESEARCH_DECISION.md`.
