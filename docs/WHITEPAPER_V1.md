# Operator Atlas: Mathematical Representation, Source Evidence, and the Limits of Generalization

**Superior MI Labs — Experimental Software and Empirical Research Note, Version 1.0.0**  
**Date:** 20 October 2026  
**Status:** public research report, **not peer reviewed**; no claim of a newly superior AI model.

## Abstract

Operator Atlas investigates whether small mathematical representations can be used to organize, fit, compose, and execute selected computational transformations, while source-backed storage retains information that cannot safely be inferred. It includes dense, quantized, factorized, and sparse-corrected linear operators; an experimental character-level recurrent neural generator; a bounded symbolic-regression grammar; immutable source identities and document retrieval; and an inspectable local computation graph. We report favorable **constructed numerical** fitting, unfavorable **near-byte-matched neural** comparisons, a real-world historical time-series stress test, and conservative **hand-authored evidence** retrieval results. No one of these experiments establishes general language intelligence, unrestricted mathematical law discovery, a production-grade forecasting system, or an LLM performance improvement.

## 1. Research questions

**RQ1.** Under what conditions does structured mathematical representation reduce numerical storage without harming relevant operations?  
**RQ2.** Can bounded equation search generalize from observed coordinates to unseen in-range values?  
**RQ3.** Do expanded equation vocabularies generalize to external physical data not designed around those vocabularies?  
**RQ4.** Does a native factorized recurrent language model outperform a roughly matched-budget dense counterpart?  
**RQ5.** Can source evidence be retrieved without claiming that external documents have been learned into neural weights?

## 2. Architecture and mathematical method

The program has one canonical operator interface with distinct dense, int8, low-rank, and sparse-corrected implementations, shared across research, inference, and inspection. For a low-rank matrix W ≈ AB, Atlas executes y=A(Bx) where possible without constructing W. Its learned numerical equation subsystem fits coefficients for a declared finite grammar of at most 16 basis terms, including polynomial, trigonometric, multiplicative interactions and some positive-domain transforms.

A candidate takes the form:

    f(x) = Σ_k θ_k φ_k(x)

where basis functions φ_k are selected from a **human-supplied grammar**, not invented from arbitrary mathematical primitives. Candidate selection uses a fit-based criterion with a held-out validation diagnostic; this is not a guarantee of predictive calibration. Explicit stored-coordinate-domain rules prevent application predictions beyond observed range. An offline extrapolation study deliberately violates that usual prediction boundary to stress-test generalization and is not part of the supported prediction API.

Source-backed information uses SQLite records with content hashes. Retrieved text is an excerpt, not an independently verified answer. The toy neural checkpoint is separate: text generation by sampling character predictions does not consult retrieved documents unless a separate, explicitly disclosed pathway is used.

## 3. Experimental methods

### 3.1 Structured numerical tests

Controlled synthetic studies use five predetermined seeds and ground-truth generating functions selected from the candidate equation grammar. Training and held-out coordinates are disjoint. On the synthetic positive-domain logarithmic example, the 13-term learner averaged 0.204580 unseen-coordinate RMSE, while the 16-term learner averaged 0.001012. This is a test of accessible basis expressiveness, **not autonomous invention of physical laws**.

### 3.2 Matched neural comparison

On the small authored character corpus, a three-seed near-byte-matched comparison recorded mean validation loss of 0.58505 for the narrow dense RNN (21,672 stored FP32 bytes), versus 1.01851 for the factorized RNN (21,888 bytes). The dense system performed better. A separate two-layer, variance-matched study reduced—but did not reverse—the factorized disadvantage (0.30033 dense vs 0.36647 factorized; different experiment, 3 seeds). Training data are small, repetitive and domain-restricted; natural-language quality remains poor.

### 3.3 Source retrieval

The fictional sentence-overlap evaluation returned the expected source sentence for 18/18 hand-authored positive questions and abstained on 10/10 unsupported questions. Negative questions were designed to expose false positives of an older chunk-level lexical retrieval. This is **not** an independent paraphrase, fact-verification or general comprehension benchmark. Retrieved sentences may themselves be inaccurate.

### 3.4 Non-synthetic astronomical historical stress test

We included 309 annual sunspot observations from 1700 through 2008, obtained from the public-domain statsmodels sunspots dataset (original NGDC source). Six chronological training cutoffs (1880, 1900, 1920, 1940, 1960, 1980) are followed by separate 15-year held-out futures. The model fits only prior data. All formula basis selection takes place before each test horizon. We compare a historical mean, last known training value, and a separately fitted autoregressive process of order two (AR2) recursively forecasting each horizon. AR2 receives no held-out target values. We evaluate equation predictions **outside their normal qualified input domain only for stress analysis**.

| Method | Average future-15-year RMSE (six periods) |
|---|---:|
| AR(2) recursively forecast baseline | 30.913529 |
| Historical mean | 39.062339 |
| Atlas 13-term symbolic learner | 40.966374 |
| Atlas 16-term symbolic learner | 41.397293 |
| Last in-training observation (held fixed) | 59.157628 |

The 16-term learner is worse than the historical mean and AR2 controls. Data involve a temporally structured nonlinear physical process; a static year→activity expression drawn from the available candidate vocabulary is a poor inductive bias. These are six related historical splits, **not independent evidence of statistical significance**. This is not a competitive solar forecasting benchmark, nor a claim against modern physical or statistical solar models.

![Six chronological historical holdouts; lower error is better.](../assets/sunspot_backtest.png){width=85%}

## 4. Main findings

1. Structured observations with matching basis functions can be compressed into useful mathematical rules, but arbitrary knowledge cannot generally be reconstructed from compact equations.
2. Expanding the basis family improves performance on deliberately matching patterns; it does not help unstructured random observations and may overfit.
3. On historically out-of-domain real data, the tested basis family loses to a simple AR2 model. Distribution shift and missing dynamical state matter.
4. On our small authored language corpus, byte-matched dense RNNs outperform factorized ones. The project cannot claim efficient LLM replacement.
5. Quantization sometimes saves stored bytes while increasing actual CPU latency. Physical hardware efficiency must be measured, not inferred from a matrix dimension count.
6. Source storage and narrow extractive answers are useful independently of neural language quality, but they are not parametric learning.

## 5. Failure modes and publication boundaries

Potential failure modes: a misleading but lexically similar source passage, incompatible physical units in chained equations, uncalibrated uncertainty propagation, extrapolation outside observed coordinates, catastrophic forgetting if future weight updates are introduced, compute cost exceeding memory savings, data licensing uncertainty, unreviewed remote API exposure, and semantic failure of the tiny character generator.

Operator Atlas is currently suitable as a **local educational and empirical research tool**, or a bounded offline numerical fitting experiment under direct human supervision. It is not qualified for safety-critical or medical decisions, autonomous scientific claims, public unauthenticated server deployment, general conversational assistance, or production process control.

## 6. Go / no-go decision

**Continue:** reproducible mathematical operator research, scientific fitting with declared domains, source-backed experimental provenance, and comparison across different representations and CPU/GPU implementations.

**Do not automatically scale:** the native weight-field language model. Further scale requires evidence of a meaningful gain over strong **matched compute, bytes, and quality** baselines on independently sourced data. Treat failure to demonstrate that advantage as reason to redirect effort, not reason to hide negative results.

## 7. Reproduction and disclosures

From the release directory run `./launch.sh doctor` for isolated system tests and a sanitized ZIP, `./launch.sh study-real` for the non-synthetic backtest, or `./launch.sh demonstrate` for fictional document ingestion. The scientific results are stored in `results/*.json`. Users can inspect exact fixture hashes, evaluation seeds, scripts and source. Neither user documents nor private conversation transcripts are packed into the public release.

This report is a research-software note and must not be presented as peer-reviewed validation of a novel general-purpose cognitive architecture.

## 8. Related research and dataset provenance

- Ashkenazi et al., *NeRN—Learning Neural Representations for Neural Networks*, 2022, https://arxiv.org/abs/2212.13554
- Wang et al., *SVD-LLM V2: Optimizing Singular Value Truncation for Large Language Model Compression*, NAACL 2025, https://aclanthology.org/2025.naacl-long.217/
- Udrescu et al., *AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity*, NeurIPS 2020, https://proceedings.neurips.cc/paper/2020/hash/33a854e247155d590883b93bca53848a-Abstract.html
- statsmodels sunspots dataset (public domain; original National Geophysical Data Center, 1700–2008): https://github.com/statsmodels/statsmodels/blob/main/statsmodels/datasets/sunspots/data.py

**Code license:** See `docs/LICENSING.md`. An open-source license has not yet been selected by the project owner; public source visibility does not itself grant reuse rights.