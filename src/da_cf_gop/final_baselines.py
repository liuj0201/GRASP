"""Patient-only, automatically aligned transfers of the existing GOP systems.

The legacy healthy-trained assets are valid for held-out patients, but include
MC03/MC04. Consequently this runner calibrates on the development patient only
and makes no independent healthy-test claim. Test PHN/TextGrid files are never
used for feature extraction, segmentation, or scoring.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp
import torch

from .ctc_sf import ctc_viterbi_state_path
from .dual_experiment import (add_gold, apply_calibration, fit_calibration,
                              index_logits, log_probs_for, make_folds, phone_ids,
                              token_metrics)
from .parikh_experiment import read_jsonl, write_json, write_jsonl
from .phonology import ARPABET_39


ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT.parents[1] / "phoneme_project" / "code" / "code9"
OUT = ROOT / "artifacts" / "evaluation" / "final_baselines"
DUAL = ROOT / "artifacts" / "evaluation" / "dual_graph" / "dual_graph_v1"
sys.path.insert(0, str(LEGACY))


def patient_inputs():
    patients = {"F01", "F03", "F04", "M01", "M02", "M04", "M05"}
    acoustic = [r for r in read_jsonl(DUAL / "sensitivity7" / "acoustic_inputs.jsonl.gz")
                if r["speaker"] in patients]
    wavs = {r["reading_event_id"]: r["wav_path"] for r in read_jsonl(
        ROOT / "artifacts" / "manifests" / "sensitivity7.jsonl.gz")}
    return [{**r, "wav_path": wavs[r["event"]]} for r in acoustic]


def automatic_segments(log_probs, canonical):
    """One nonblank CTC state span per canonical anchor, in 20-ms frames."""
    path, _ = ctc_viterbi_state_path(log_probs, phone_ids(canonical))
    segments = []
    for i in range(len(canonical)):
        frames = np.flatnonzero(path == 2 * i + 1)
        if not len(frames):
            raise ValueError("CTC path lacks a required canonical state")
        segments.append((int(frames[0]), int(frames[-1]) + 1))
    return np.asarray(segments, dtype=np.int32)


def extract(device="cuda", limit=None):
    """A single XLS-R pass yields both dense logits and layer-19 centers."""
    from transformers import AutoFeatureExtractor, AutoModel
    from common.ctc_backend import load_audio

    rows = patient_inputs()
    if limit is not None:
        rows = rows[:limit]
    cache = OUT / "features"
    cache.mkdir(parents=True, exist_ok=True)
    sources = index_logits(ROOT / "artifacts" / "cache" / "logits", {r["event"] for r in rows})
    directory = LEGACY / "models" / "facebook__wav2vec2-xls-r-300m"
    processor = AutoFeatureExtractor.from_pretrained(directory, local_files_only=True)
    model = AutoModel.from_pretrained(directory, local_files_only=True).eval().to(device)
    payload = torch.load(LEGACY / "models" / "uq_gop_torgo_adapted" / "phone_head.pt",
                         map_location="cpu", weights_only=False)
    head = torch.nn.Linear(512, 39).to(device)
    head.load_state_dict(payload["phone_head_state_dict"])
    head.eval()
    started = time.monotonic()
    for number, row in enumerate(rows, 1):
        dest = cache / (row["event"] + ".npz")
        if dest.exists():
            continue
        audio = load_audio(Path(row["wav_path"]))
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            output = model(**{k: v.to(device) for k, v in inputs.items()}, output_hidden_states=True)
            # HF's extract_features is the layer-normalized 512-D conv output.
            dense = head(output.extract_features)[0].float().cpu().numpy()
            features = output.hidden_states[19][0].float().cpu().numpy()
        lp = log_probs_for(sources[row["event"]])
        if len(lp) != len(dense):
            raise ValueError(f"CTC/XLS-R timeline mismatch for {row['event']}: {len(lp)} vs {len(dense)}")
        spans = automatic_segments(lp, row["canonical_phones"])
        centers = (spans[:, 0] + spans[:, 1]) // 2
        np.savez_compressed(dest, event=row["event"], canonical=np.asarray(row["canonical_phones"]),
                            dense_logits=dense, segment_starts=spans[:, 0], segment_ends=spans[:, 1],
                            center_vectors=features[centers].astype(np.float32))
        if number == 1 or number % 50 == 0 or number == len(rows):
            print(f"XLS-R automatic features {number}/{len(rows)}; {time.monotonic()-started:.1f}s", flush=True)
        del output
    write_json(OUT / "feature_descriptor.json", {
        "n_requested": len(rows), "elapsed_s": time.monotonic()-started,
        "input": "audio and prompt only", "alignment": "canonical CTC Viterbi nonblank state spans; 20ms",
        "mixgop": "XLS-R300M hidden-state 19 midpoint center; 1024 dimensions; no PCA",
        "uq": "same XLS-R normalized 512-D conv features; restored legacy healthy-trained linear head",
        "healthy_training_speakers": payload["training"]["training_speakers"],
        "healthy_test_claim": False, "test_manual_boundaries": False,
    })


def plain_score(row, i, method, score):
    return {"event": row["event"], "speaker": row["speaker"], "phone_index": i,
            "target": row["canonical_phones"][i], "method": method, "gop": float(score),
            "candidate_event": "unspecified", "candidate_phone": None,
            "candidate_confidence": 0., "candidate_margin": 0.}


def fit_development_calibration(rows, config):
    # Native GMM log densities have magnitudes around 1e8. Standardizing the
    # development scores avoids an ill-conditioned Platt fit; AP uses raw scores.
    x = np.asarray([-r["gop"] for r in rows], dtype=float)
    center, scale = float(x.mean()), float(x.std())
    if scale == 0:
        scale = 1.
    normalized = [{**r, "gop": (r["gop"] + center) / scale} for r in rows]
    fit = fit_calibration(normalized, config)
    fit["intercept"] -= fit["slope"] * center / scale
    fit["slope"] /= scale
    fit.update({"development_score_mean": center, "development_score_std": scale,
                "standardization": "development patient only; raw score preserved for ranking"})
    return fit


def dense_scores():
    from recurrence.uq_gop.core import score_segment_all
    from recurrence.cagop.cagop.config import CaGOPConfig
    from recurrence.cagop.cagop.duration_model import CaGOPDurationModel
    from recurrence.cagop.cagop.tolerance import DurationToleranceModel
    from recurrence.cagop.cagop.scoring import score_segment

    torch.set_num_threads(2)
    dense_payload = torch.load(LEGACY / "models" / "uq_gop_torgo_adapted" / "phone_head.pt",
                              map_location="cpu", weights_only=False)
    duration_root = LEGACY / "results" / "raw_scores" / "cagop"
    payload = torch.load(duration_root / "duration_model" / "best_torgo_healthy_speaker_dev.pt",
                         map_location="cpu", weights_only=False)
    config = CaGOPConfig(**payload["config"])
    model = CaGOPDurationModel(config).eval()
    model.load_state_dict(payload["model_state"])
    tolerance = DurationToleranceModel.load_json(duration_root / "duration_tolerance_training_healthy_only.json")
    ids = {p: i for i, p in enumerate(ARPABET_39)}
    results = []
    for number, row in enumerate(patient_inputs(), 1):
        with np.load(OUT / "features" / (row["event"] + ".npz"), allow_pickle=False) as data:
            logits = data["dense_logits"].astype(float)
            starts, ends = data["segment_starts"], data["segment_ends"]
            if data["canonical"].tolist() != row["canonical_phones"]:
                raise ValueError("feature cache reference mismatch")
        phone_zero = [ids[p] for p in row["canonical_phones"]]
        durations = (ends - starts) * 20. / 30.
        speed = float(durations.mean())
        with torch.inference_mode():
            prediction = model(torch.tensor([[i + 1 for i in phone_zero]]), torch.tensor([speed]))[0].numpy()
        posteriors = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        for i, (phone, start, end) in enumerate(zip(phone_zero, starts, ends)):
            uq = score_segment_all(logits[start:end], phone, prior=dense_payload["prior"],
                                   temperature=dense_payload["temperature"])
            for method, score in uq.items():
                results.append(plain_score(row, i, "UQ_" + method,
                                           -score if method.startswith("entropy_") else score))
            balance = tolerance.lookup(phone + 1, speed)
            ca = score_segment(posteriors[start:end], phone, float(durations[i]),
                               float(prediction[i]), balance.value_frames, beta=config.beta)
            for name in ("gop", "center_gop", "cagop_full", "cagop_duration_only", "cagop_transition_only"):
                results.append(plain_score(row, i, "CaGOP_" + name, getattr(ca, name)))
        if number % 100 == 0:
            print(f"Dense GOP scoring {number}/1627", flush=True)
    write_jsonl(OUT / "dense_predictions_without_gold.jsonl.gz", results)
    write_json(OUT / "dense_descriptor.json", {
        "uq_training": dense_payload["training"], "cagop_split": payload["split"],
        "patient_training": False, "healthy_metrics": "not evaluated: legacy healthy-trained assets",
        "reference_and_alignment": "new canonical; automatic CTC Viterbi; no test PHN or TextGrid",
        "calibration": "development patient only", "headline": ["UQ_maxlogit_prior", "CaGOP_cagop_full"],
        "claim": "scoring/architecture transfer to TORGO phone MDD, not original system numbers",
    })


def mix_scores():
    """Fit/score one full-covariance GMM at a time to bound RAM usage."""
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=2)
    with (LEGACY / "results" / "raw_scores" / "mixgop" / "xlsr19.jsonl").open(encoding="utf-8") as f:
        metadata = json.loads(next(f))["metadata"]
    phones = metadata["fit_report"]["trained_phones"]
    healthy = [r for r in read_jsonl(LEGACY / "manifests" / "torgo_mixgop_release.jsonl")
               if r["speaker_group"] == "healthy"]
    train = defaultdict(list)
    for row in healthy:
        with np.load(LEGACY / "cache" / "mixgop" / "xlsr19" / (row["utterance_id"] + ".npz")) as data:
            for phone, vec in zip(data["phones"], data["center_vectors"]):
                if str(phone) in phones:
                    train[str(phone)].append(vec)
    test = defaultdict(list)
    for row in patient_inputs():
        with np.load(OUT / "features" / (row["event"] + ".npz"), allow_pickle=False) as data:
            for i, (phone, vec) in enumerate(zip(data["canonical"], data["center_vectors"])):
                if str(phone) in phones:
                    test[str(phone)].append((row, i, vec))
    pieces = OUT / "mixgop_phone_scores"
    pieces.mkdir(exist_ok=True, parents=True)
    fit_rows = []
    for seed, phone in enumerate(phones):
        piece = pieces / (phone + ".jsonl.gz")
        if piece.exists():
            continue
        started = time.monotonic()
        x = np.asarray(train[phone], dtype=float)
        if len(x) > 512:
            selected = np.arange(len(x))
            np.random.default_rng(seed).shuffle(selected)
            x = x[selected[:512]]
        gm = GaussianMixture(n_components=32, covariance_type="full", reg_covar=1e-5,
                             init_params="kmeans", n_init=1, max_iter=100, tol=1e-3,
                             random_state=seed).fit(x)
        scores = gm.score_samples(np.asarray([v[2] for v in test[phone]], dtype=float))
        write_jsonl(piece, [plain_score(row, i, "MixGoP_XLSR19_C32", score)
                           for (row, i, _), score in zip(test[phone], scores)])
        info = {"phone": phone, "n_training": len(x), "seed": seed, "converged": bool(gm.converged_),
                "n_iter": gm.n_iter_, "n_test": len(scores), "elapsed_s": time.monotonic()-started}
        write_json(pieces / (phone + ".fit.json"), info)
        fit_rows.append(info)
        print(f"MixGoP {phone} {seed+1}/32; {info['elapsed_s']:.1f}s", flush=True)
        del gm
        gc.collect()
    write_jsonl(OUT / "mixgop_predictions_without_gold.jsonl.gz", [r for phone in phones for r in read_jsonl(pieces / (phone + ".jsonl.gz"))])
    write_json(OUT / "mixgop_descriptor.json", {
        "trained_phones": phones, "reference_speakers": sorted({r["speaker_id"] for r in healthy}),
        "n_healthy_reference_utterances": len(healthy), "test_boundaries": "automatic CTC Viterbi",
        "training_boundaries": "released healthy-reference TextGrid center-vector caches",
        "feature_dim": 1024, "components": 32, "covariance": "full", "PCA": False,
        "inventory": "frozen historical 32-phone inventory; separate identical-token subset for all methods",
        "fit": [json.loads((pieces / (p + ".fit.json")).read_text()) for p in phones],
    })


def evaluate():
    config = json.loads((ROOT / "configs" / "dual_graph_v1.json").read_text())
    raw = list(read_jsonl(OUT / "dense_predictions_without_gold.jsonl.gz"))
    mixpath = OUT / "mixgop_predictions_without_gold.jsonl.gz"
    if mixpath.exists():
        raw += list(read_jsonl(mixpath))
    for cohort in ("primary5", "sensitivity7"):
        records = list(read_jsonl(DUAL / cohort / "evaluation_labels.jsonl.gz"))
        evaluated = add_gold(raw, records) if cohort == "sensitivity7" else add_gold(
            [r for r in raw if r["speaker"] in config["cohorts"][cohort]], records)
        methods = sorted({r["method"] for r in evaluated})
        results, predictions, fits = [], [], []
        for method in methods:
            own = [r for r in evaluated if r["method"] == method]
            per = []
            for fold in make_folds(config, cohort):
                dev = [r for r in own if r["speaker"] == fold["development_patient"]]
                test = [{**r, "fold": fold["fold"]} for r in own if r["speaker"] == fold["test_patient"]]
                calibrator = fit_development_calibration(dev, config)
                out = apply_calibration(test, calibrator, config)
                per.append({"speaker": fold["test_patient"], **token_metrics(out)})
                predictions.extend(out)
                fits.append({"fold": fold, "method": method, "calibrator": calibrator})
            macro = {key: float(np.mean([r[key] for r in per])) for key in ("auprc", "auroc", "f1", "brier", "nll")}
            results.append({"method": method, "macro": macro, "per_speaker": per})
        dest = OUT / cohort
        dest.mkdir(exist_ok=True)
        write_json(dest / "metrics.json", {"cohort": cohort, "results": results})
        write_json(dest / "fits.json", fits)
        write_jsonl(dest / "phone_predictions.jsonl.gz", predictions)
        if mixpath.exists():
            mix = [r for r in predictions if r["method"] == "MixGoP_XLSR19_C32"]
            reference = {(r["event"], r["phone_index"], r["target"]) for r in mix}
            joint = predictions + [r for r in read_jsonl(DUAL / cohort / "phone_predictions.jsonl.gz")
                                   if r["speaker"] in config["cohorts"][cohort]]
            finalpath = ROOT / "artifacts" / "evaluation" / "final_v2" / cohort / "phone_predictions.jsonl.gz"
            if finalpath.exists():
                newest = list(read_jsonl(finalpath))
                current_methods = {r["method"] for r in newest}
                joint = [r for r in joint if r["method"] not in current_methods] + newest
            groups = defaultdict(list)
            for row in joint:
                if (row["event"], row["phone_index"], row["target"]) in reference:
                    groups[row["method"]].append(row)
            subset = []
            for name, values in sorted(groups.items()):
                keys = [(r["event"], r["phone_index"], r["target"]) for r in values]
                if len(keys) != len(reference) or set(keys) != reference:
                    raise ValueError("MixGoP subset is not identical across all methods")
                per = [{"speaker": s, **token_metrics([r for r in values if r["speaker"] == s])}
                       for s in config["cohorts"][cohort]]
                subset.append({"method": name, "n": len(reference), "per_speaker": per,
                               "macro": {k: float(np.mean([r[k] for r in per])) for k in ("auprc", "auroc")}})
            write_json(dest / "mixgop32_shared_metrics.json", {"results": subset, "same_token_keys": True,
                       "n": len(reference), "n_errors": sum(r["gold_error"] for r in mix),
                       "final_v2_source": str(finalpath) if finalpath.exists() else None,
                       "note": "Supplementary frozen 32-phone inventory; 39-phone headline remains separate"})
        print(cohort, [(r["method"], round(r["macro"]["auprc"], 5)) for r in results
                       if r["method"] in ("UQ_maxlogit_prior", "CaGOP_cagop_full", "MixGoP_XLSR19_C32")], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("extract", "dense", "mix", "evaluate"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(2)
    {"extract": lambda: extract(args.device, args.limit), "dense": dense_scores,
     "mix": mix_scores, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
