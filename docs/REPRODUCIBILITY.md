# Reproducing the research workflow

Run commands from the repository root with Python 3.10 and an editable installation. This guide traces the current code's dependency chain. It does not claim a fresh end-to-end corpus rerun on every supported platform.

## External inputs

1. Obtain TORGO audio, prompts, PHN transcriptions, and speaker Notes through the corpus provider. Preserve its speaker/session structure. `data.py` expects speaker directories under the corpus's speaker-set directories, with `Session*/wav_headMic`, `prompts`, and `phn_headMic` or `phn_arrayMic`. The source filenames and prompt contents determine event identities and reference phones.
2. Obtain the released GOP-SF / CTC-SF acoustic checkpoint and matching processor from the [upstream CTC-based-GOP repository](https://github.com/frank613/CTC-based-GOP): XLSR-53 fine-tuned on LibriSpeech train-clean-100, with the 40-symbol blank/ARPABET inventory. The historical checkpoint folder was named `is24/models/checkpoint-8000`, and the matching processor folder `is24/models/processor_config_gop`. A generic wav2vec 2.0 checkpoint is not an equivalent replacement. The repository does not supply these weights or a newly verified model-download workflow.
3. Retain the bundled, frozen `resources/dual_graph/cmudict.dict` and its license. OOV phonemization uses `phonemizer` with the pinned `espeakng-loader` library/data, not an arbitrary system dictionary.

For the separate language study, obtain Qwen2.5-1.5B-Instruct in Q4_K_M GGUF form and a compatible llama.cpp CPU server. See the language section below.

## Local path configuration

Make a local copy of `configs/frozen_v1.json`, for example `configs/user.local.json`. Set `paths.data_root`, `paths.checkpoint`, and `paths.processor` to your local external inputs. Keep `paths.artifacts` as `artifacts` for the commands below: several later research modules intentionally use this fixed source-relative location. Relative paths in the initial configuration are resolved against the repository root when the configuration remains under `configs/`.

The public defaults are `data/torgo`, `models/ctc_sf/checkpoint-8000`, and `models/ctc_sf/processor_config_gop`. Public path fields replace the original machine layout; scientific parameters are preserved. Consequently the public configuration hash differs from the original local run. Recreate provenance and caches under the public configuration rather than expecting the original configuration hash.

The frozen configuration also contains earlier severity and legacy-comparator fields. Those inputs are not needed for the current manifest/logit/sparse pipeline. Do not run the old severity, `run-phone-loso`, or whole-study `verify` commands to reproduce the current manuscript.

Generated manifests, model fits, caches, and predictions contain corpus-level or participant-level information and remain local under ignored `artifacts/`. Their hashes and absolute paths are machine/run-specific; copying a historical cache to a new path is not the same as regenerating its provenance.

## Stage 1: prepare corpus and acoustic outputs

```sh
python -m da_cf_gop.cli --config configs/user.local.json build-manifests
python -m da_cf_gop.cli --config configs/user.local.json extract-logits --cohort sensitivity7
python -m da_cf_gop.dual_audit --cohort all
```

The larger cohort covers the patient recordings needed by both experiments, together with the healthy speakers used by the inherited preprocessing workflow. The audit checks PHN symbol/closure quality; its timing information is not supplied to the acoustic scorer.

Outputs include `artifacts/manifests/`, `artifacts/cache/logits/`, and `artifacts/evaluation/dual_graph/data_audit/`.

## Stage 2: regenerate shared full-graph evidence

```sh
python -m da_cf_gop.dual_experiment --cohort primary5 --workers 4
python -m da_cf_gop.dual_experiment --cohort sensitivity7 --workers 4
python -m da_cf_gop.final_experiment prepare
```

The dual-graph runner regenerates dictionary-based labels, fixed raw baselines, cyclic patient folds, and full-graph evidence. It also executes historical graph comparisons that the sparse experiment does not require as headline results; this is a computationally substantial dependency of the retained research implementation.

`final_experiment prepare` verifies the raw graph settings and extracts shared raw evidence from those outputs into `artifacts/evaluation/final_v2/fixed_evidence_without_gold.jsonl.gz`. Its `prepare` implementation uses this fixed destination even if `--output` is supplied. Historical fitted probabilities and thresholds are discarded at this extraction step. Do not substitute the original paper's summary metrics for the required raw evidence.

## Stage 3: sparse scorer, prediction, and evaluation

Use a new destination for every fresh experiment. The following commands process both cohorts:

```sh
python -m da_cf_gop.sparse_experiment prepare --output artifacts/evaluation/sparse_reproduction
python -m da_cf_gop.sparse_experiment infer --output artifacts/evaluation/sparse_reproduction --workers 4
python -m da_cf_gop.sparse_experiment predict --output artifacts/evaluation/sparse_reproduction
python -m da_cf_gop.sparse_evaluation --output artifacts/evaluation/sparse_reproduction
```

The default registered configuration is `configs/candidate_pruning_20260909.json`. It specifies full graph, acoustic top-1, acoustic top-2, and acoustic top-1 plus training substitutions. The last policy, `learned_top1_da`, is the proposed scorer. Candidate policies are not selected using newly evaluated test scores.

Preparation makes training maps that exclude the patient whose training features are being constructed. Prediction fits the detector and standardization using training patients, selects/calibrates on the development patient, and freezes all outer predictions. Evaluation requires the all-cohort completion marker before joining held-out labels.

The recorded corpus protocol expects 6,820 primary and 8,353 expanded assessment positions after quality, lexical, alignment, and malformed-prompt exclusions. The evaluator checks these population sizes. A mismatch requires investigating corpus/version/preprocessing differences; do not remove the check just to obtain a score. `F04_Session2_0009` is excluded by the registered configuration in all roles.

The main outputs are `primary5/metrics.json`, `sensitivity7/metrics.json`, frozen prediction files, and candidate-coverage/miss reports inside the chosen destination. Metrics are fractions; multiply AP and F1 by 100 for the manuscript table units.

## Graph resource pilot

After sparse preparation and prediction, run:

```sh
python scripts/benchmark_sparse_graph.py --include-learned --run-output artifacts/evaluation/sparse_reproduction --output artifacts/evaluation/sparse_reproduction/pilot_learned.json
```

The default sample contains 24 recordings with reference lengths of 1–40 phones. The benchmark warms the JIT decoder and compares each method with a full graph using the same state compaction. It measures selection, construction, and sum/max inference. The reported forward-history array is not process peak memory, and the timing does not include the acoustic encoder, cache loading, or detector. Hardware and numerical-library differences affect timing; the manuscript's CPU measurements are not portable runtime guarantees.

## Additional table controls

To fit and evaluate the prompt-clean full-graph/prior and isolated-score controls from the same shared evidence:

```sh
python -m da_cf_gop.final_experiment predict --cohort primary5 --config configs/final_v3_prompt_clean.json --output artifacts/evaluation/final_v3_prompt_clean
python -m da_cf_gop.final_experiment predict --cohort sensitivity7 --config configs/final_v3_prompt_clean.json --output artifacts/evaluation/final_v3_prompt_clean
python -m da_cf_gop.final_experiment evaluate --cohort primary5 --output artifacts/evaluation/final_v3_prompt_clean
python -m da_cf_gop.final_experiment evaluate --cohort sensitivity7 --output artifacts/evaluation/final_v3_prompt_clean
```

`timebox_ensemble.py` reproduces the ensemble predictions, reusing `timebox_compact.py` feature/model helpers and both `timebox_ensemble_20260905.json` and `timebox_compact_20260905.json`:

```sh
python -m da_cf_gop.timebox_ensemble --output artifacts/evaluation/ensemble_reproduction
```

The historical combined optimization-report script expects a separate multi-branch `protocol.json` and additional completed optimization branches under its fixed artifact directory. It is not a standalone evaluator for a new ensemble directory. The current release documents this limitation instead of claiming that the sparse commands alone regenerate every manuscript table. Ensemble method labels are listed in `METHOD_MAP.md`.

Earlier MixGOP, UQ-GOP, CA-GOP, severity, residual, ranking, and cloud/AudioLLM experiments are outside the current sparse experiment. Some source modules may remain for shared functions and historical tests. Their presence does not make them required replication steps.

## Local language feasibility study

The paper uses the summary and sentence tasks from `experiments/llm_three_tasks_20260921/`. Their cases are generated deterministically in code: each task has 30 development inputs and 100 test inputs. Sentence target validation uses the bundled CMUdict. Neither task needs TORGO, a cloud API key, or the acoustic model.

The recorded Qwen GGUF source is repository `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, revision `91cad51170dc346986eccefdc2dd33a9da36ead9`, filename `qwen2.5-1.5b-instruct-q4_k_m.gguf`, SHA-256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`. The historical CPU server used llama.cpp build `b10964`, commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`.

On Windows x64, the provided setup script downloads the pinned server archive and approximately 1.1 GB model, verifies their SHA-256 hashes, and creates a local runtime manifest:

```sh
python resources/feedback_value_runtime/setup_runtime.py
```

After this setup, the runner can start the local server automatically. The setup script downloads Windows binaries; on other systems, install/build a compatible server yourself and use the explicit launch below.

Start a compatible local llama.cpp server yourself, adjusting executable/model paths for your installation:

```sh
llama-server -m /path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8776 --device none --n-gpu-layers 0 --threads 2 --threads-batch 2 --parallel 1 --ctx-size 4096 --temp 0 --seed 20260921 --no-cont-batching --cache-ram 0 --alias Qwen2.5-1.5B-Instruct-Q4_K_M
```

The runner reuses a healthy local server on port 8776. If no server is running, its launcher expects a locally configured `resources/feedback_value_runtime/runtime_manifest.json`. Starting the server explicitly avoids that manifest dependency. Verify that the local server actually serves the intended GGUF and settings; the runner's health check alone does not establish model identity.

```sh
python experiments/llm_three_tasks_20260921/run.py --tasks summary,sentence --split test --label test_final
python experiments/llm_three_tasks_20260921/report.py
```

Use a fresh label for a changed prompt or independent run. The runner saves the raw single-attempt output and scores it without automatic retries or template fallback. Its fixed generation request uses temperature 0, seed 20260921, and a maximum of 128 output tokens. The report summarizes saved calls and makes no new model requests.

Summary checks verify phone identity, counts, assessed denominators, and every flagged word occurrence. Sentence checks require all selected target words, 5–10 words, one sentence, and no direct copy of the input sentence. Passing these checks does not establish naturalness or clinical value. Twenty summary test cases are paired changes to other test cases, so 100 cases are not 100 independent observations.

## Verification boundaries

Unit tests exercise synthetic inputs and small graph cases; they do not replace access to the exact corpus and acoustic model. The reconstructed command sequence follows source dependencies. Full corpus reproducibility additionally depends on the external checkpoint, matching TORGO files, pinned phonemizer resources, numerical libraries, and the historical raw-evidence extraction checks. Preserve failed runs and provenance rather than replacing results with manuscript numbers.
