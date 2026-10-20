# Superior MI Labs — v0.4 workstation diagnostic review

**Evidence:** user-provided `Operator-Atlas-Diagnostic-20261020-043019.zip`,
produced by `./launch.sh doctor` on 2026-10-20. Avoid redistributing the
machine report without the operator's approval.

## Confirmed installation status

- Doctor: `PARTIAL`, 7 PASS / 0 FAIL / 2 SKIP; 30 Python tests passed.
- Checksum verification: 70 distribution files; 4 trained checkpoint fixtures.
- NumPy 2.5.4, Python 3.12.3, 16 logical CPUs.
- Optional PyTorch training skipped because torch was absent in that venv.
- Optional browser automation skipped because Playwright and/or Chromium
  were absent from the doctor's interpreter/runtime.
- No imported documents, saved experiments, or evidence source rows.
- Default checkpoint was a **single-layer** factorized character model;
  the two-layer candidate was not the selected model at startup.

## Physical-performance caution

v0.4 doctor reported 18,792 B original operator payload versus 8,409 B
post-training int8 payload, a 55.25% reduction. Decoding took 0.01674
ms/character dense versus 0.02629 ms/character in the selected NumPy route,
approximately a **57% slowdown**. This is a microbenchmark, not wall-clock
conversation latency or a GPU benchmark. v0.5 adds a validation-time
latency gate that can refuse slower candidates. It does not make the int8
kernel faster.

## Actionable host commands

```bash
./launch.sh doctor
./launch.sh probe --characters 400 --tolerance 0.05 --objective storage
./launch.sh probe --characters 400 --tolerance 0.05 --objective latency_guarded --max-slowdown 0.05
./launch.sh serve --model models/stacked-factorized
```

PyTorch or browser automation are optional to operate the program. Their
absence is not a runtime failure. Tests never read the user's documents or
train on them. The `doctor` report is suitable for private technical review,
not public posting without checking the metadata.
