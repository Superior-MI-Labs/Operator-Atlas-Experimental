# Public release qualification and claims

**Release identifier:** Operator Atlas 1.0.0 **Experimental Research Software**. Public distribution is an invitation to replicate, critique, and improve, not a declaration of a general-purpose AI model.

### Verified contracts

- Same NumPy inference and mathematical operator implementations power CLI, HTTP API, graph visualization and diagnostic suite.
- Doctor runs against a disposable copy, keeps private uploaded sources out of the published artifacts, and emits an inspectable report.
- Explicit distinction between neural character generation, source-backed text excerpts, and qualified mathematical predictions.
- Source hashes and model versions are recorded; model cannot read arbitrary uploaded files without explicit import.
- Dense/factorized/quantized comparison includes unfavourable outcomes; hardware measurements are local and nontransferable.
- Historical time-series stress tests are separate from supported in-domain prediction semantics.

### Not qualified

General-purpose chat, token-based high-quality language modeling, autonomous discovery of novel laws, high-scale retrieval, arbitrary uploaded file understanding, physical-unit-safe composition, calibrated predictive intervals, remote deployment, general-purpose GPU acceleration, and reliable LLM weight-memory replacement.

### Recommended public positioning

"Operator Atlas is a locally runnable experimental research workbench exploring mathematical representations, empirical equation fitting, source evidence, and their failure modes. The project includes negative comparisons and an independent historical stress test. The language component is a tiny exploratory character model, not a production AI assistant."

### License gate

The project owner must select an explicit redistribution license before representing the project as open source. Until then, public source is inspectable but reuse rights are not assured. Prior research references must be credited separately.

### Release publishing workflow

The GitHub repository may initially contain only its summary README. The authoritative complete release is the checksummed ZIP. To synchronize it to `Superior-MI-Labs/Operator-Atlas-Experimental`, run `bash tools/publish_github.sh --publish` on the downloaded, verified release. The command does not require GitHub Actions usage and never force-pushes. A connected agent cannot treat a documentation-only repository as proof the full source upload succeeded; verify the exact files on GitHub afterward.
