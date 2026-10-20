# Operator Atlas v0.8 — Mathematical Expression and Composition Contract

## Research objective
Test whether richer bounded mathematical vocabularies can learn useful relationships
from source-linked numerical observations and safely compose previously qualified
functions, **without** introducing a second information owner or execution engine.

## Design decisions and competing approaches
1. One unrestricted symbolic AI generator: open-ended search, uninterpretable costs,
   no reliable finite budget. Rejected for the present research stage.
2. Enlarge the canonical bounded grammar and permit **explicit** compositions of
   immutable, already qualified functions. Selected because every output can be
   replayed and checked against original observation records.

A compositional neural network is NOT yet implemented. These are qualified
scalar source-derived equations, not learned hidden neural weight generators.

## Contracts

- `atlas.discovery.discover(points, maximum_terms=3, basis_count=None)`
  searches 13 **declared** basis terms, including products such as `x*sin(2x)`.
  `basis_count=8` retains the old mathematical vocabulary for controlled ablation.
- A deterministic 80/20 split is used for equation selection, with training BIC
  ranking candidate terms and validation RMSE as a comparison measurement.
- A separate rightmost 15% coordinate block is withheld while refitting the
  selected basis, reported as `blocked_holdout_rmse`. That diagnostic does NOT
  influence basis selection or qualify extrapolation.
- CSV-derived equations are versioned as `compositional-bic-v2`. Legacy
  `basis-bic-v1` equations persist and continue evaluating because the first
  eight basis indices are unchanged. An existing equation's source and formula
  never change due to a future training request.
- `EvidenceStore.compose_equations(ids, x)` processes 2–6 distinct IDs in order.
  Intermediate values and each source's SHA-256 identity are returned. Each
  equation refuses inputs outside its observed x-range. Any failure stops the
  chain immediately; nothing is persisted. Type/range failures are explicit.
- Raw coordinates with absolute magnitude >1000 or observations >1e12 are explicitly refused because this prototype does not yet fit a stable learned coordinate transform. Rescale before fitting; automatic affine normalization remains future research.
- Composition does NOT verify physical-unit compatibility, quantify propagated
  uncertainty, or establish scientific causation. Validation RMSE for individual
  stages must not be summed to claim a bound on the chain.
- The same operation is accessible through `POST /api/compose_equations`,
  `./launch.sh compose-equations ID1 ID2 --at 0.375`, the Grounded Answer Studio
  (`chain equations ID1 ID2 at 0.375`), and the GUI Equation Discovery tab.
- `graph.py` owns all visual topology metadata. The GUI invokes existing service
  actions; the browser does not implement independent numerical inference.

## Workflows

1. Import `examples/fixtures/calibration.csv`, then learn columns
   `coordinate,reading` in the Equation Discovery tab.
2. Import `examples/fixtures/secondary_calibration.csv` and learn columns
   `measurement,calibrated`.
3. Compose those two equation IDs starting from `x=0.375`. The second equation
   takes the **first numerical output** as its input.
4. Inspect the two numerical stages, source hashes and error disclosures.
5. Try `x=2`: the first equation produces an in-range prediction, but its result
   is outside the second equation's measured domain, so the composition stops.
6. Run `./launch.sh doctor` and inspect the synthetic `fixture_demo.json`
   and `equation_study.json` records in the support bundle.

## Limitations and interpretation

The fixture data are wholly synthetic and deliberately chosen to favor declared
terms. They validate implementation contracts, not universal symbolic discovery,
real-world sensor calibration, improved LLM fluency, or global applicability.
The demo still has an experimental tiny character-RNN, not a general chatbot.

## Further experiments

- Native neural layers with different mathematical families and matched budgets.
- Learned coordinate changes with independent stability constraints.
- Explicit error propagation and measurement-unit schemas for chained models.
- Out-of-distribution and adversarial evaluation against independently sourced data.
- Large-scale storage and latency baselines against quantization and dense operators.
