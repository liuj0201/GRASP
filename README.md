# GRASP

Research code for **GRASP: Graph-Based Reference-Anchored Pronunciation Assessment and Feedback for Dysarthric Speech**, by Jiayu Liu, Jiawen Qi, and Qinyu Chen (Leiden University).

Repository: [liuj0201/GRASP](https://github.com/liuj0201/GRASP).

GRASP assesses assigned English text and speech with a sparse pronunciation graph. Its DA-CF-GOP scorer combines joint CTC graph evidence with a detector trained on dysarthric speech. Substitution candidates combine acoustic top-1 phones with up to four substitutes observed in training patients. An optional local language study evaluates verbalization of code-computed statistics and generation of practice sentences.

This release identifies the paper method as **`candidate_pruning_20260909` / `learned_top1_da`**. The Python package retains the name `da_cf_gop` and some historical module names because the current experiments reuse their implementations. In particular, the older `final_v2` and `final_v3_prompt_clean` methods are not the paper's sparse DA-CF-GOP scorer.

## Contents

- `src/da_cf_gop/`: acoustic input handling, pronunciation graphs, candidate selection, CTC inference, detector fitting, and evaluation.
- `configs/`: recorded experimental settings and lexical policy.
- `scripts/`: resource benchmarking and selected reporting utilities.
- `experiments/llm_three_tasks_20260921/`: synthetic summary and sentence tasks used by the supporting language study.
- `resources/dual_graph/`: the frozen CMU pronunciation dictionary and its original license.
- `tests/`: implementation checks, including small exhaustive graph comparisons and separation of prediction from evaluation.
- [Reproduction instructions](docs/REPRODUCIBILITY.md) and [paper-to-code map](docs/METHOD_MAP.md).

Raw TORGO recordings, PHN transcriptions, participant-level predictions, fitted patient models, acoustic checkpoints, and LLM binaries are not bundled. Reproducing the corpus experiments requires obtaining the external data and acoustic model and generating the local intermediate artifacts described below.

## Installation and checks

Use Python 3.10 and an editable installation from this directory:

```sh
python -m pip install -e ".[test,paper,web]"
python -m pytest
python examples/synthetic_graph.py
```

The project pins the numerical, phonemization, and acoustic-model dependencies in `pyproject.toml`. The historical experiment environment used PyTorch 2.6.0, Transformers 5.8.1, NumPy 1.26.4, and scikit-learn 1.4.1.post1. GPU installation may require a PyTorch build compatible with your system. Editable installation preserves the source-relative locations expected by the research runners; this is not a self-contained model wheel.

For the corpus workflow, start with [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The package's legacy `run-phone-loso` and `verify` commands concern the earlier frozen study, not the current sparse paper experiment.

## Reported scope

The manuscript reports speaker-macro phone-error detection AP of 27.29% on the five-speaker primary cohort and 36.38% on the overlapping seven-speaker sensitivity cohort for the sparse scorer. The primary PP-AF restricted-substitution comparator is 24.22%. These are manuscript results, not a claim that a new public installation has rerun the corpus study.

The study is retrospective and uses small, overlapping patient cohorts. Historical test outcomes informed earlier development. All candidate policies are reported; the sensitivity cohort is not independent confirmation. PHN transcriptions supply training/evaluation labels, while acoustic scoring uses audio and assigned text without PHN boundaries. Resource measurements cover graph processing on 24 recordings and exclude the acoustic encoder, cache loading, and detector.

The supporting Qwen study reports 98/100 summary cases and 85/100 sentence cases passing fixed factual/formal checks; templates pass all cases. These checks do not establish sentence naturalness, clinical benefit, or superiority over templates. The AudioLLM comparisons and cloud-judge experiments from the development workspace are outside this paper release.

## Citation and resource licenses

Use `CITATION.cff` to cite this software and record the repository version used for an experiment. The manuscript has no publication DOI recorded in this release; add its final bibliographic details when available.

The project code is released under the [MIT License](LICENSE). The bundled CMUdict resource retains its separate license in `resources/dual_graph/CMUDICT_LICENSE`; see [third-party notices](THIRD_PARTY_NOTICES.md). TORGO, the upstream acoustic model, Qwen, and llama.cpp have their own distribution and usage terms; this repository does not redistribute those data or model binaries.
