# Paper-to-code map

The current manuscript is *GRASP: Graph-Based Reference-Anchored Pronunciation Assessment and Feedback for Dysarthric Speech*. The complete manuscript includes the synthetic summary/sentence study; the extended scorer manuscript shares the sparse pronunciation-assessment experiments. The current method is `learned_top1_da`, not the older method internally named `da_cf_gop` in `final_experiment.py`.

## Main pipeline

| Paper component | Implementation | Configuration / behavior |
| --- | --- | --- |
| Audio and fixed acoustic backend | `backend.py`, `stages.py` | 16 kHz; external CTC-SF checkpoint and processor; blank ID 0 plus 39 ARPABET phones |
| Reference pronunciations and acceptable variants | `dual_lexicon.py`, `phonology.py` | `dual_acceptable_v1.json`; frozen CMUdict, bundled eSpeak fallback; unsupported lexical variants masked |
| Training substitution maps | `candidate_training.py` | Speaker-normalized observed substitutions; at most four neighbors; a training example's own patient is excluded |
| Acoustic and combined selection | `candidate_selection.py` | Per-frame top-1 nonblank phone at posterior >= 0.01, union over the utterance; add training neighbors to substitutions only |
| Joint graph/CTC scoring | `dual_decode.py`, `dual_prior.py` | Preserve OK and epsilon DEL; up to two insertions per gap; prune SUB/INS without renormalizing retained arc weights |
| Four-score detector and calibration | `sparse_experiment.py`, reusable fitting functions in `final_experiment.py` | Negative sum/max GOP, log SUB/OK, log DEL/OK; signed-log transform; training-only transforms; development AP model selection and calibration |
| Label construction and quality exclusions | `data.py`, `dual_audit.py`, `dual_lexicon.py`, `dual_experiment.py` | Human PHN identities, all-optimal reference alignment, stable correctness labels, lexical mask; no PHN timing in scoring |
| Corpus metrics and candidate coverage | `sparse_evaluation.py`, `metrics.py` | Shared detection token keys; all-cohort prediction-completion barrier; speaker-macro AP/F1 and paired speaker bootstrap |
| Graph processing pilot | `scripts/benchmark_sparse_graph.py` | 24 recordings, one process/two threads, JIT warm-up, full-compacted control; encoder and detector excluded |

Paths in this table are under `src/da_cf_gop/` unless otherwise specified.

## Method names in the tables

| Manuscript label | Internal method | Source experiment |
| --- | --- | --- |
| DA-CF-GOP / Acoustic + training | `learned_top1_da` | `sparse_experiment.py`, `candidate_pruning_20260909.json` |
| Acoustic top-1 | `acoustic_top1_da` | Same sparse experiment |
| Acoustic top-2 | `acoustic_top2_da` | Same sparse experiment |
| Full graph + detector | `full_graph_da` (sparse control), corresponding to `graph_da` | Sparse experiment / prompt-clean final experiment |
| Full graph | `graph_generic` | `final_experiment.py`, `final_v3_prompt_clean.json` |
| Full graph + detector + prior reweighting | `graph_cf_da` | Same prompt-clean final experiment |
| Conventional GOP | `conventional_gop_raw` | Raw score in `dual_experiment.py`; prompt-clean development calibration/evaluation in `final_experiment.py` |
| GOP-SF | `ctc_sf_sd_norm_raw` | Same workflow; score implementation in `baselines.py` |
| PP-AF R / U | `ppaf_rps_raw` / `ppaf_ups_raw` | Same workflow; `parikh2025.py`, `parikh_experiment.py` |
| PP-AF R + detector | `ppaf_da` | Prompt-clean final experiment |
| Score ensemble | `ensemble_full` | `timebox_ensemble.py` and `timebox_compact.py` feature/model helpers |
| Score ensemble without prior features | `ensemble_graph_iso` | Same ensemble runner |

The original `final_v2` directory remains the location of shared *fixed raw evidence*. Its name does not mean that the sparse experiment uses the uncorrected historical population: the malformed event is removed by the sparse configuration before fitting and evaluation. Likewise, historical prior correction is an ablation; it is absent from `learned_top1_da`.

## Supporting language study

| Component | File under `experiments/llm_three_tasks_20260921/` |
| --- | --- |
| Synthetic observations, independent count checks, and summary templates | `summary_task.py` |
| Synthetic target words, sentence prompt, constraint checker, and rule baselines | `sentence_task.py` |
| Single-attempt local inference and saved requests/responses | `run.py` |
| Summary of saved outputs, without model calls | `report.py` |

Run only `--tasks summary,sentence` for this paper. The historical phrase-extraction task, autonomous LLM grouping experiments, cloud judges, and real-recording AudioLLM comparisons are separate experiments. The 2026-09-24 detector adapter in the original development workspace is not the source of the manuscript's 98/100 and 85/100 language results.

The older `homework_runtime.py` / `homework_web.py` prototype, if present in the release, uses the earlier prompt-clean scorer. It must not be presented as the current sparse paper runtime merely because it displays the GRASP name. The sparse research pipeline and the language feasibility tasks are reproduced separately here.
