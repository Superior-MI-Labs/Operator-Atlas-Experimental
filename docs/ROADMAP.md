# Operator Atlas — Evidence-Gated Scaling Strategy

## Completed in v0.2 (research prototype)

Canonical NumPy operators, genuine dense and factorized character-RNN training, checkpoint export and parity, arithmetic-basis search, validation-based representation chooser, source-backed evidence memory, CPU microbenchmark, local API, zoom/pan graph inspector, integration tests.

## Next research gate: actual improved representation, not feature count

1. **Comparable architecture search:** Several seeds and genuinely matched *byte + training FLOP* budgets; compare to dense, int8 and small Transformer controls on genuine diverse held-out text.
2. **Native trainable corrections:** Implement parameter-efficient sparse correction with differentiable/structured sparse placement and a measured kernel that really saves memory.
3. **Multi-layer neural operators:** Introduce multiple recurrent / attention layers *through the existing `Operator` contract*. Measure training stability and generalization, not only parameter counts.
4. **Learned equation family choice:** Compare fixed low-rank, codebook, sparse and structured bases with evidence-aware selection. Reject candidates that do not improve the Pareto frontier.
5. **Continual adaptation:** Add controlled training against new permissioned text with old-task replay and forgetting metrics; promote only validated checkpoint revisions.
6. **Real multilingual and open-domain workloads:** Tokenizer, diverse corpus, proper eval suites (perplexity, task accuracy, OOD, contamination checks), explicit data rights.

## Production gate (not implemented)

- Structured content ingestion beyond text, document copyright/consent handling, deletion rights and provenance propagation.
- Proper retrieval augmented generation with evidence-citation verification and hallucination/contradiction tests.
- GPU kernels, CPU SIMD optimizations, hardware detection, real peak RSS/VRAM profiling and energy metering where available.
- Distributed artifact registry, signed checkpoints, authentication, tenant isolation, rate limiting and audit logs.
- Robust version pinning, rolling checkpoints, recoverability from interrupted training, sandboxed proposed equations.
- Visually editable typed graph compilation once a canonical executing graph and command transaction model have proven value. GUI **cannot mutate compute graphs** in v0.2.

## Principle

A feature enters the runtime only if a smaller existing mechanism cannot solve its goal and an independent test establishes correctness. A mathematical representation is selected only when it meets a quality bound. The system must never claim to have compressed arbitrary knowledge into an equation.

## v0.3 completion and next scientific gate

v0.3 delivered local file ingestion (with content provenance), extractive source retrieval, sanitized event logging, immutable experiment records, a shareable diagnostic ZIP, v0.2 evidence migration, and task-loss-grounded operator probing. The real model search selected int8 rather than mathematical factorization, and its measured CPU execution was slower in one local comparison. Do **not** present that as a compression or intelligence breakthrough.

Next high-value work is *not* another GUI feature. It is a genuine multi-layer language baseline with document-disjoint training and learned mathematical operators evaluated against strong quantized/dense controls. All future training updates must preserve a checkpoint identity, evaluated quality and explicit promotion criteria.
