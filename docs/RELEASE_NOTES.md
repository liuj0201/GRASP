# Public source release 0.1.0

Prepared on 2026-09-24 for the GRASP manuscript. The release uses the existing
`da_cf_gop` Python package name to preserve imports.

## Packaging changes

- Copied an explicit source allowlist into a separate public checkout; the
  original research workspace and results were left unchanged.
- Added installation/reproduction documentation, a code-to-paper map, MIT license,
  software citation metadata, third-party notices and a synthetic example.
- Changed only filesystem paths in `configs/frozen_v1.json` to the public checkout
  layout. Its configuration hash consequently differs from the local historical
  run. Scientific parameters and the paper's `candidate_pruning_20260909.json`
  remain unchanged.
- Restricted the local language runner's default tasks to `summary,sentence`,
  which are the two studies reported in the current manuscript. The unrelated
  phrase extraction study and cloud experiments are not included.
- Made the language runner's Windows process flag conditional. The bundled
  optional runtime downloader still targets Windows x64; other systems must
  supply a compatible local llama.cpp server as documented.
- Added a clear missing-runtime message and corrected an outdated sentence-task
  self-check: grammar remains an auxiliary check, without changing the evaluator.
- Declared `joblib` and `threadpoolctl` as direct pinned dependencies and added
  package metadata and repository links.
- Historical modules remain where required by shared imports, tests and upstream
  artifact construction. Their presence does not identify them as the paper's
  final method. The existing homework web application uses an earlier scorer;
  see the method map before using it.

## Validation scope

The staged release passed 316 unit tests on Python 3.10.20, the synthetic graph
demo, the sentence-task self-check, and 390 deterministic summary/sentence
baseline outputs. A Starlette/httpx deprecation warning was the only pytest warning.
Full TORGO extraction, acoustic inference, detector fitting, and local LLM
generation are not rerun during publication. No new paper results are claimed.

No manuscript publication venue, DOI or acceptance status is asserted by the
software citation. Update the citation metadata when the paper is published.
