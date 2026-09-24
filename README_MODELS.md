# Model downloads, trained artifacts, and local web setup

This guide distinguishes external pretrained weights from parameters trained by
GRASP. Sources and small metadata files were checked on 2026-09-24. The acoustic
weight download endpoint was checked with an HTTP HEAD request; this documentation
update did not download the full weights or rerun corpus training.

**A fresh clone can open the web page, but cannot assess a recording with only
the source files.** It needs both the external acoustic model and our fitted
detector artifacts. There is currently no public download URL for our fitted
detectors. They must be regenerated using the released experiment code.

## 1. Acoustic model: upstream weights

The acoustic model is from [frank613/CTC-based-GOP](https://github.com/frank613/CTC-based-GOP),
not trained by the GRASP authors. The upstream authors fine-tuned wav2vec 2.0
XLSR-53 on LibriSpeech train-clean-100 for phoneme CTC recognition. GRASP's paper
pipeline keeps these weights frozen. See the [upstream model instructions](https://github.com/frank613/CTC-based-GOP/blob/ffc4823330efe2c836691931a456ea9e3360871e/is24/README_is24.md).

Use commit `ffc4823330efe2c836691931a456ea9e3360871e`, containing:

- [Acoustic checkpoint](https://github.com/frank613/CTC-based-GOP/tree/ffc4823330efe2c836691931a456ea9e3360871e/is24/models/checkpoint-8000)
- [Matching processor/tokenizer](https://github.com/frank613/CTC-based-GOP/tree/ffc4823330efe2c836691931a456ea9e3360871e/is24/models/processor_config_gop)
- [Actual weight file](https://media.githubusercontent.com/media/frank613/CTC-based-GOP/ffc4823330efe2c836691931a456ea9e3360871e/is24/models/checkpoint-8000/pytorch_model.bin), 1,262,066,282 bytes (about 1.26 GB).

The raw GitHub URL for `pytorch_model.bin` returns a 135-byte **Git LFS pointer**,
not usable model weights. Use the actual weight link above or a Git LFS-enabled
checkout. The upstream LFS record specifies SHA-256:

```text
035b95ada8ec20d83378981a050166391cffdba9810e417097b74f492112541b
```

Download the inference files using this Python snippet, run from the GRASP
repository root (for example, save it outside the source package and run it with
Python). It uses only the standard library and preserves both vocabularies:

```python
from pathlib import Path
from urllib.request import urlretrieve
import hashlib

revision = "ffc4823330efe2c836691931a456ea9e3360871e"
prefix = f"frank613/CTC-based-GOP/{revision}/is24/models/"
files = [
    "checkpoint-8000/config.json",
    "checkpoint-8000/preprocessor_config.json",
    "checkpoint-8000/vocab.json",
    "checkpoint-8000/pytorch_model.bin",
    "processor_config_gop/preprocessor_config.json",
    "processor_config_gop/special_tokens_map.json",
    "processor_config_gop/tokenizer_config.json",
    "processor_config_gop/vocab.json",
]
for relative in files:
    destination = Path("models/ctc_sf") / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    host = ("https://media.githubusercontent.com/media/"
            if relative.endswith("pytorch_model.bin")
            else "https://raw.githubusercontent.com/")
    if not destination.exists():
        print("Downloading", relative, flush=True)
        partial = destination.with_name(destination.name + ".partial")
        urlretrieve(host + prefix + relative, partial)
        partial.replace(destination)

weights = Path("models/ctc_sf/checkpoint-8000/pytorch_model.bin")
digest = hashlib.sha256()
with weights.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
assert weights.stat().st_size == 1262066282, "Wrong weight size or LFS pointer"
assert digest.hexdigest() == "035b95ada8ec20d83378981a050166391cffdba9810e417097b74f492112541b", "Weight checksum mismatch"
print("Acoustic weights verified")
```

These destinations match `configs/frozen_v1.json`. The vocabulary is the exact
40-output inventory (39 ARPABET phones plus blank ID 0). A generic XLSR model,
another CTC vocabulary, or just `pytorch_model.bin` without its matching
configuration/processor is not an equivalent replacement.

The snippet omits upstream optimizer/training-state files, which inference does
not need. Our provenance code hashes whole directories: an inference-only folder
will have a different directory hash from a complete historical checkout. Build
new caches with the selected files; do not mix in historical cache descriptors.
Obtain and use upstream assets under the upstream terms; GRASP's MIT license does
not relicense third-party model weights.

## 2. Models and parameters trained by this project

The missing `.joblib` and fitted JSON files are **our experiment outputs**, not
weights supplied by the CTC-based-GOP or Qwen authors. The source release does
not currently include a downloadable bundle of these fitted outputs.

| Artifact | What is learned or estimated | Source of supervision |
| --- | --- | --- |
| Error detector and feature scaler | StandardScaler plus logistic regression or histogram gradient boosting, selected from the recorded candidate set | Training patients; model selection uses development AP |
| Probability calibration and decision threshold | Nondecreasing Platt calibration and development F1 threshold | Development patient, without fitting on held-out test labels |
| Sparse substitution-neighbor tables | Up to four observed substitutes per target phone, using speaker-normalized counts | Training patients; a training example's own patient is excluded |
| Older web model's confusion prior and event classifier | Training-derived prior and a separate substitution-versus-deletion classifier | Training patients in the older prompt-clean experiment |

These are relatively small downstream statistical models and tables. GRASP does
not fine-tune the large acoustic model or Qwen in the current paper workflow.
The JSON files in `configs/` describe the protocol; they do not contain the
learned detector coefficients, fitted scaler, or fitted calibration.

### Current paper scorer

After completing Stages 1 and 2 of [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md),
the sparse training sequence is:

```sh
python -m da_cf_gop.sparse_experiment prepare --output artifacts/evaluation/sparse_reproduction
python -m da_cf_gop.sparse_experiment infer --output artifacts/evaluation/sparse_reproduction --workers 4
python -m da_cf_gop.sparse_experiment predict --output artifacts/evaluation/sparse_reproduction
```

The current paper method is `learned_top1_da`. This produces, among other files:

```text
artifacts/evaluation/sparse_reproduction/
  training_maps/<training-context>.json
  <cohort>/models/<held-out-speaker>.learned_top1_da.joblib
  <cohort>/fits/<held-out-speaker>.json
```

The current sparse `.joblib` contains the fitted binary detector, calibration,
and feature metadata. Training maps are stored separately. There is one set of
research models per fold, not a single released model validated on arbitrary new
users. Recreating them requires access to TORGO and the shared evidence described
in the reproduction guide. Run the separate evaluation command only after all
outer predictions have been frozen.

### Older web prototype

`homework_runtime.py` currently loads **`final_v3_prompt_clean`**, not the sparse
paper scorer. It defaults to research policy `F01` and needs these local files:

```text
artifacts/evaluation/final_v3_prompt_clean/sensitivity7/
  config.json
  F01.fit.json
  models/F01.da_cf_gop.joblib
```

After the same prerequisite Stages 1 and 2, generate this older model family with:

```sh
python -m da_cf_gop.final_experiment predict --cohort sensitivity7 --config configs/final_v3_prompt_clean.json --output artifacts/evaluation/final_v3_prompt_clean
```

Despite the command name `predict`, this step also **fits** the fold-specific
detectors and writes their artifacts. It uses existing acoustic/graph evidence;
it does not retrain wav2vec 2.0. The web model includes an event classifier and a
different feature/prior layout, so renaming a sparse `.joblib` file to the web
model's filename will not work. Its `config.json` and `F01.fit.json` must come
from the same run as the model. No original TORGO corpus recordings or PHN labels
are needed by web inference once the correct assets exist; inference uses the
new uploaded recording and assigned text. The corpus is needed to reproduce
the training workflow.

## 3. Optional Qwen for the web prototype: Transformers

The default **template** feedback mode requires no language model. For the
web prototype's optional Qwen mode, download the original Transformers model
from [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
This is a Qwen model, with no GRASP fine-tuning. The following download pins the
upstream revision checked for this guide:

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306', local_dir='models/Qwen2.5-1.5B-Instruct')"
```

Set the local model path before launching the web app. On Windows PowerShell:

```powershell
$env:DA_CF_GOP_FEEDBACK_MODEL = (Resolve-Path 'models/Qwen2.5-1.5B-Instruct').Path
```

On Linux/macOS:

```sh
export DA_CF_GOP_FEEDBACK_MODEL="$PWD/models/Qwen2.5-1.5B-Instruct"
```

The web loader expects a directory containing Transformers weights, model
configuration, and tokenizer files. It does not load a GGUF file or connect to
llama.cpp. Its feedback model selects permitted message IDs; this is different
from the paper's separate summary/sentence generation study.

## 4. Qwen for the paper language study: GGUF

The separate `experiments/llm_three_tasks_20260921/` summary/sentence study uses
[Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9),
revision `91cad51170dc346986eccefdc2dd33a9da36ead9`, file
`qwen2.5-1.5b-instruct-q4_k_m.gguf`.

On Windows x64, the existing setup script downloads the pinned model and
llama.cpp CPU runtime and verifies their checksums:

```sh
python resources/feedback_value_runtime/setup_runtime.py
python experiments/llm_three_tasks_20260921/run.py --tasks summary,sentence --split test --label test_final
```

For other platforms, see the explicit local-server launch in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md#local-language-feasibility-study).
These controlled synthetic tasks do not require TORGO or a trained GRASP
detector. Downloading this GGUF does **not** supply the web model in Section 3 or
the missing pronunciation detector in Section 2.

## 5. Start the older web prototype after preparing its assets

From the repository root, with Python 3.10:

```sh
python -m pip install -e ".[web]"
python -m da_cf_gop.homework_web
```

Open <http://127.0.0.1:8765>. The app reads acoustic paths directly from
`configs/frozen_v1.json`; a separate `configs/user.local.json` used by the
experiment CLI is not automatically used by the web app. Either keep the
documented `models/ctc_sf/` layout or set the web app's actual configuration paths.

The page and `/api/status` can return HTTP 200 before any model loads. A response
with `engine_loaded: false` is not a successful model-readiness check. In our
clean-clone probe, the first valid audio assessment returned HTTP 503 for the
missing `F01.fit.json`. After preparing all required assets, check a real
assessment request as well as the page. Download/installation instructions here
do not establish that a fresh full web deployment has passed that test.

The public web prototype has not yet been migrated to the current sparse paper
scorer. This README describes its actual prerequisites and does not claim that
downloading the external models alone completes that migration.
