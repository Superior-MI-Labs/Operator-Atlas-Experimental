# Operator Atlas fixture corpus (public, synthetic)

These are fictional and program-generated records for automated ingestion tests. They are not user uploads, not scientific observations, and are not suitable as empirical evidence about natural-language understanding.

- `research_notebook.md`: a unique fictional fact and a named calibration channel.
- `unrelated_notes.md`: deliberately irrelevant record, tests answer abstention.
- `calibration.csv`: sampled synthetic quadratic-plus-sinusoidal measurements, small deterministic perturbation.
- `random_control.csv`: pseudorandom values meant to defeat equation fitting.

The `./launch.sh demonstrate` command imports these in a temporary database and records the before/after results. `./launch.sh doctor` bundles its fixture-derived findings but **never real user file contents**.

- `secondary_calibration.csv`: synthetic second stage for numerical composition; quantities are dimensionless and deliberately calibrated for the example.
