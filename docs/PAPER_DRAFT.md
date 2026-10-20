# Operator Atlas: Learning Compact Mathematical Representations for a Small Recurrent Language Model

**Superior MI Labs · Working research manuscript · Not peer reviewed · v0.2**

## Abstract

We implement a common interface for dense, quantized, factorized, and sparsely corrected linear transformations, and attach it to a small character-level recurrent language model. We compare a factorized model trained from scratch against near-payload-matched dense controls on a small authored corpus. Although the representation supports direct `A(Bx)` execution and numerically agrees between PyTorch training and NumPy inference, the dense controls outperform the factorized network on held-out character prediction across three seeds. We also implement a bounded mathematical-basis search, validation-constrained operator selection, source-backed information retrieval, and an inspectable local computation graph. These are implementation foundations, not evidence of a new generally superior neural architecture.

## Research question

Can neural transformations learn a representation family that reduces physically stored information and actual execution cost **without degrading generalization at a matched resource budget**? Does the answer differ by layer, activation distribution, task and hardware?

## Method and baselines

For a standard operator `y=Wx`, we implement `y=Ax'` with `x'=Bx` where `A` has shape `(o,r)` and `B` has shape `(r,i)`. We also implement dense float32, per-tensor int8 quantization, and factorized operators with true indexed sparse corrections. A one-layer Elman RNN predicts the next character of a small synthetic-domain training corpus of 112,571 characters. It is trained by next-character cross entropy with AdamW. A training/validation split of 88/12 is fixed within the corpus. We run factorized hidden-48/rank-16 and dense hidden-38 controls with equal steps (700), batch (20), sequence length (48), and seeds 7, 11, 19. The models differ in effective training FLOPs and hidden size; this is a *near byte budget* rather than a fully matched-compute experiment.

## Results

Factorized (21,888 byte parameter payload): mean held-out loss 1.01851 ± 0.13710 (sample SD). Dense (21,672 byte payload): mean held-out loss 0.58505 ± 0.05393. Lower is better. This is a **negative result** for the factorized implementation. All per-seed results and artifact weights are in `results/multi_seed_ablation.json` and `models/ablations/`. PyTorch-to-NumPy final-logit export agreement for the shipped factorized checkpoint was within 5.73e-6 maximum absolute error.

Additional synthetic operator benchmarks show favorable low-rank compression only for deliberately structured matrices and poor approximations on irregular data. Numeric equation search can recover equations within its supplied grammar but does not discover laws outside that vocabulary. Source-backed document memory retrieves stored passages, not facts embedded into the recurrent weights.

## Interpretation and constraints

No architecture superiority, large-scale language quality, causal reasoning, open-ended equation invention, native continual neural learning, hardware energy savings, or response-speed improvement for LLMs has been demonstrated. The authored corpus has repeated templates, so held-out loss is not representative of broad natural language. We must next compare model families on independently sampled real text corpora under matched compute and storage, with stronger small-Transformer and quantized baselines, GPU/CPU profiling, distribution-shift stress tests, and trained selective corrections.

## Related work

- Ha et al., HyperNetworks (2016): https://arxiv.org/abs/1609.09106
- Ashkenazi et al., NeRN (2023): https://arxiv.org/abs/2212.13554
- Yuan et al., ASVD (2023): https://arxiv.org/abs/2312.05821
- Wang et al., BitNet (2025): https://jmlr.org/papers/v26/24-2050.html
- Udrescu and Tegmark, AI Feynman (2020): https://pmc.ncbi.nlm.nih.gov/articles/PMC7159912/

## Reproducibility

Install requirements and run `PYTHONPATH=. pytest -q`, `./launch.sh simulate`, or train the checkpoints from `data/corpus.txt` using `./launch.sh train --kind factorized --hidden 48 --rank 16 --steps 700 --seed 7 --out models/reproduction` and the equivalent dense hidden-38 command. Numerical parity and artifact metadata are verified by the exporter. `results/` holds recorded reference outputs; rerun measurements on local hardware before making performance claims.


## v0.4 Research update: initialization and depth

A two-layer extension of the recurrent mathematical operator engine was benchmarked against a similarly sized dense control. All runs used the same templated authored corpus and 400 optimizer steps. FP32 parameter payloads were 34,176 bytes (factorized rank16, hidden48) and 33,224 bytes (dense hidden38). Across seeds 3,7,11 the reporting-only holdout next-character cross-entropies were mean 0.36647 (factorized, SD 0.04517) and 0.30033 (dense, SD 0.05040). This does not support a factorized quality or size superiority claim.

A prior initial factorized scale had produced much worse loss. A variance-matched initialization policy improved optimization strongly; this suggests earlier negative results were partly a training-condition artifact. Because it was identified during iterative exploratory research, the initializer finding requires prospective confirmation on independent corpora and larger models before publication as a general effect.

The new `doctor` qualification command is an engineering contribution, not a new scientific result. We will not combine synthetic matrix errors with held-out language loss as if they were the same quantity.


## v0.5 experimental update — policy versus physical computation

A workstation diagnostic found that an int8 representation reduced stored
weights by 55.25% yet increased per-character latency by approximately 57%.
A validation-timed latency guard was therefore introduced. It retains the
existing storage minimizer for comparison while allowing the user to reject a
slower candidate and execute the original dense checkpoint instead. This is a
policy improvement and a negative scientific result for the current NumPy
quantized operator, **not a new mathematical architecture finding**.


## Addendum: Operator Atlas v0.9 methods and limitations

The representation search added log, sqrt and reciprocal terms only for all-positive x inputs, and uses an affine polynomial coordinate on large-offset data. The public five-seed positive-domain log fixture improved held-out test RMSE from 0.204580 (13-term search) to 0.001012 (16 terms). In the distinct source-response fixture, the program selected all 18 expected fictional source sentences and abstained on ten deliberately unsupported queries; questions were crafted for lexical overlap. This is **not** model-generated natural language fluency. The trained recurrent character model and its earlier negative baseline results are unchanged. Statistical uncertainty, novel-language performance and large external datasets still require independent evaluation. Refer to versioned machine-readable results; do not treat synthetic results as evidence of broad AI ability.
