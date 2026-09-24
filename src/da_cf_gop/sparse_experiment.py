"""Preregistered candidate pruning with patient-disjoint evidence adaptation.

Preparation creates graphs from reference/acoustic inputs and explicit training
maps. Prediction fits the existing four-cue Graph+DA model. Test-label joining
is a separate command in sparse_evaluation; no policy is selected there.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from .candidate_selection import select_candidates
from .candidate_training import estimate_neighbors
from .dual_decode import decode
from .dual_experiment import add_gold, phone_ids, json_safe
from .dual_prior import generic_prior
from .final_experiment import fit_predictor, model_score, calibration, output_rows, log_ratio, key
from .parikh_experiment import read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "artifacts/evaluation/dual_graph/dual_graph_v1"
BASE = ROOT / "artifacts/evaluation/final_v2/fixed_evidence_without_gold.jsonl.gz"
DEFAULT_CONFIG = ROOT / "configs/candidate_pruning_20260909.json"
DEFAULT_OUTPUT = ROOT / "artifacts/evaluation/candidate_pruning_20260909"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def acoustic_log_probs(event):
    filename = hashlib.sha256(event.encode()).hexdigest()[:24] + ".npz"
    with np.load(ROOT / "artifacts/cache/logits" / filename, allow_pickle=False) as data:
        assert str(data["utterance_id"].item()) == event
        logits = data["logits"].astype(np.float64)
    return logits - logsumexp(logits, axis=1, keepdims=True)


def inputs(config):
    excluded = set(config["excluded_events"])
    base = [r for r in read_jsonl(BASE) if r["event"] not in excluded]
    events = {r["event"] for r in base}
    acoustic = {r["event"]: r for r in read_jsonl(OLD / "sensitivity7/acoustic_inputs.jsonl.gz")
                if r["event"] in events}
    assert set(acoustic) == events
    return base, acoustic


def prepare(output, config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert not (output / "predictions_complete.json").exists(), "A frozen run cannot be re-prepared."
    assert not any((output / "graph_cache").glob("*.json")), "Cached graphs exist; use a new output directory."
    existing = output / "config.json"
    if existing.exists():
        assert json.loads(existing.read_text(encoding="utf-8")) == config
    write_json(existing, config)
    base, acoustic = inputs(config)
    records = [r for r in read_jsonl(OLD / "sensitivity7/evaluation_labels.jsonl.gz")
               if r["event"] in acoustic]
    contexts, mappings, folds_by_cohort = {}, {}, {}
    for cohort in config["cohorts"]:
        folds = json.loads((OLD / cohort / "folds.json").read_text())
        folds_by_cohort[cohort] = folds
        for fold in folds:
            train = fold["training_patients"]
            dev, test = fold["development_patient"], fold["test_patient"]
            assert not set(train) & {dev, test} and dev != test
            speaker_context = {}
            for speaker in config["cohorts"][cohort]:
                contributors = tuple(sorted(s for s in train if s != speaker))
                assert speaker not in contributors
                context = "+".join(contributors)
                speaker_context[speaker] = context
                if context not in contexts:
                    training_records = [r for r in records if r["speaker"] in contributors]
                    estimated = estimate_neighbors(training_records, contributors,
                        max_neighbors=config["training_neighbor_budget"], min_count=1)
                    contexts[context] = estimated
                    write_json(output / "training_maps" / f"{context}.json", estimated)
            mappings[fold["fold"]] = {"fold": fold, "speaker_context": speaker_context,
                                      "strategies": {}}

    jobs, event_selectors = {}, {}
    started = time.perf_counter()
    for number, (event, row) in enumerate(sorted(acoustic.items()), 1):
        lp = acoustic_log_probs(event)
        acceptable = [phone_ids(a) for a in row["acceptable_phones"]]
        local = {}
        for strategy in config["strategies"]:
            name = strategy["name"]
            if name == "full_graph_da":
                continue
            relevant_contexts = {"acoustic"}
            if strategy.get("training_neighbors"):
                relevant_contexts = {m["speaker_context"][row["speaker"]] for m in mappings.values()
                                     if row["speaker"] in m["speaker_context"]}
            for context in sorted(relevant_contexts):
                selection = select_candidates(lp, acceptable, policy=strategy["policy"],
                    top_k=strategy.get("top_k", 2), posterior_threshold=strategy.get("threshold", .01),
                    learned_neighbors=contexts[context]["neighbors"] if context != "acoustic" else None)
                job = {"event": event,
                       "substitution_candidates": selection.substitution_phone_ids,
                       "insertion_candidates": selection.insertion_phone_ids}
                identifier = digest(job)
                jobs[identifier] = {**job, "identifier": identifier, "acoustic": row}
                local[name, context] = {"graph": identifier, "context": context,
                    "active_phone_ids": selection.active_phone_ids,
                    "retained_acoustic_mass": selection.retained_acoustic_mass,
                    "selection_sources": selection.selection_sources}
        event_selectors[event] = local
        if number % 200 == 0:
            print(f"Selected {number}/{len(acoustic)} recordings; {len(jobs)} unique graphs", flush=True)

    for fold_id, mapping in mappings.items():
        for strategy in config["strategies"]:
            name = strategy["name"]
            if name == "full_graph_da":
                continue
            mapping["strategies"][name] = {
                event: event_selectors[event][name, mapping["speaker_context"][row["speaker"]]
                    if strategy.get("training_neighbors") else "acoustic"]
                for event, row in acoustic.items() if row["speaker"] in mapping["speaker_context"]}
        write_json(output / "fold_indices" / f"{fold_id}.json", mapping)
    write_jsonl(output / "graph_jobs.jsonl.gz", jobs.values())
    write_json(output / "preparation.json", {
        "events": len(acoustic), "reference_tokens": len(base), "unique_graphs": len(jobs),
        "training_contexts": len(contexts), "seconds": time.perf_counter() - started,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("sparse_experiment.py", "candidate_selection.py", "candidate_training.py", "dual_decode.py")},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_semantics": "Only identical event and selected graph branches share inference. Selection provenance stays per fold.",
        "test_labels_used_for_selection": False,
    })
    print(f"Prepared {len(jobs)} unique graphs for {len(acoustic)} recordings", flush=True)


def decode_job(job):
    row = job["acoustic"]
    lp = acoustic_log_probs(row["event"])
    started = time.perf_counter()
    decoded = decode(lp, [phone_ids(a) for a in row["acceptable_phones"]], generic_prior(),
        canonical_ids=phone_ids(row["canonical_phones"]),
        substitution_candidates=job["substitution_candidates"],
        insertion_candidates=job["insertion_candidates"])
    seconds = time.perf_counter() - started
    tokens = [{"event": row["event"], "speaker": row["speaker"], "phone_index": i,
        "target": target, "acceptable": row["acceptable_phones"][i],
        "gop": float(decoded["token_log_odds"][i]),
        "viterbi_gop": float(decoded["max_marginal_token_log_odds"][i]),
        "event_posteriors": decoded["token_posteriors"][i].tolist(),
        "allowed_substitution_phone_ids": job["substitution_candidates"][i]}
        for i, target in enumerate(row["canonical_phones"])]
    return {"identifier": job["identifier"], "event": row["event"], "tokens": tokens,
        "resources": {name: decoded[name] for name in ("log_partition", "graph_state_count",
            "ctc_state_count", "graph_arc_count", "graph_array_bytes", "sum_forward_bytes", "max_history_trace_bytes")},
        "frames": len(lp), "decode_seconds_parallel_observation": seconds}


def worker_init():
    threadpool_limits(limits=1)


def infer(output, workers):
    assert not (output / "predictions_complete.json").exists(), "Predictions have already been frozen."
    preparation = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    decoder_hash = hashlib.sha256((Path(__file__).parent / "dual_decode.py").read_bytes()).hexdigest()
    assert preparation["source_sha256"]["dual_decode.py"] == decoder_hash, "Decoder changed since preparation."
    jobs = list(read_jsonl(output / "graph_jobs.jsonl.gz"))
    pending = [job for job in jobs if not (output / "graph_cache" / (job["identifier"] + ".json")).exists()]
    # Start long recordings first, so the end of the run is not one large task.
    pending.sort(key=lambda job: -len(job["acoustic"]["canonical_phones"]))
    started = time.perf_counter()
    print(f"Inferring {len(pending)}/{len(jobs)} uncached graphs on {workers} workers", flush=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        futures = {pool.submit(decode_job, job): job["identifier"] for job in pending}
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            path = output / "graph_cache" / (result["identifier"] + ".json")
            write_json(path.with_suffix(".partial"), json_safe(result))
            path.with_suffix(".partial").replace(path)
            if count % 100 == 0 or count == len(pending):
                elapsed = time.perf_counter() - started
                print(f"Decoded {count}/{len(pending)} graphs ({elapsed:.1f}s)", flush=True)
    write_json(output / "inference_complete.json", {"graphs": len(jobs), "workers": workers,
        "new_graphs": len(pending), "wall_seconds": time.perf_counter() - started})


def features(rows):
    values = []
    for row in rows:
        ok, sub, dele = row["event_posteriors"]
        values.append([-row["gop"], -row["viterbi_gop"], log_ratio(sub, ok), log_ratio(dele, ok)])
    x = np.asarray(values)
    assert np.isfinite(x).all(), "Nonfinite graph evidence needs investigation, not held-out clipping."
    return np.sign(x) * np.log1p(np.abs(x))


def predict(output):
    assert not (output / "predictions_complete.json").exists(), "Do not refit a frozen run."
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    base, _ = inputs(config)
    for row in base:
        row["allowed_substitution_phone_ids"] = sorted(set(range(1, 40)) - set(phone_ids(row["acceptable"])))
    assert (output / "inference_complete.json").exists()
    source_model = json.loads((ROOT / "configs/final_v3_prompt_clean.json").read_text())
    assert config["classifier_candidates"] == source_model["classifier_candidates"]
    freeze_files = {}
    with threadpool_limits(limits=2):
        for cohort, speakers in config["cohorts"].items():
            records = [r for r in read_jsonl(OLD / cohort / "evaluation_labels.jsonl.gz")
                       if r["event"] not in config["excluded_events"] and r["speaker"] in speakers]
            directory = output / cohort
            all_predictions = []
            folds = json.loads((OLD / cohort / "folds.json").read_text())
            for fold in folds:
                fold_id = fold["fold"]
                index = json.loads((output / "fold_indices" / (fold_id + ".json")).read_text())
                fitting = {"fold": fold, "methods": {}}
                train_speakers, dev_speaker, test_speaker = fold["training_patients"], fold["development_patient"], fold["test_patient"]
                for strategy in config["strategies"]:
                    name = strategy["name"]
                    if name == "full_graph_da":
                        evidence = [r for r in base if r["speaker"] in speakers]
                    else:
                        evidence = []
                        for event, selected in sorted(index["strategies"][name].items()):
                            cached = json.loads((output / "graph_cache" / (selected["graph"] + ".json")).read_text())
                            assert cached["event"] == event
                            evidence.extend(cached["tokens"])
                    evidence.sort(key=key)
                    train = add_gold([r for r in evidence if r["speaker"] in train_speakers],
                                     [r for r in records if r["speaker"] in train_speakers])
                    dev = add_gold([r for r in evidence if r["speaker"] == dev_speaker],
                                   [r for r in records if r["speaker"] == dev_speaker])
                    test = [r for r in evidence if r["speaker"] == test_speaker]
                    xt, xd, xx = features(train), features(dev), features(test)
                    trials, fitted_options = [], []
                    for candidate in config["classifier_candidates"]:
                        fitted, dev_score = fit_predictor(train, dev, xt, xd, candidate)
                        ap = float(average_precision_score([r["gold_error"] for r in dev], dev_score))
                        trials.append({"parameters": candidate, "development_ap": ap})
                        fitted_options.append((fitted, dev_score))
                    best = max(range(len(trials)), key=lambda i: (trials[i]["development_ap"], -i))
                    fitted, dev_score = fitted_options[best]
                    cal = calibration(dev_score, [r["gold_error"] for r in dev])
                    predicted = output_rows(test, name, fold_id, model_score(fitted, test, xx), cal)
                    for prediction, raw in zip(predicted, test):
                        prediction.update({field: raw[field] for field in ("acceptable", "allowed_substitution_phone_ids")})
                    all_predictions.extend(predicted)
                    model_path = directory / "models" / f"{test_speaker}.{name}.joblib"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump({"binary": fitted, "calibration": cal, "graph_features": "four original Graph+DA cues"}, model_path)
                    fitting["methods"][name] = {"search": trials, "selected": trials[best], "calibration": cal,
                        "train_tokens": len(train), "development_tokens": len(dev), "test_predictions": len(test)}
                    print(f"Fitted {fold_id}/{name} (train={len(train)}, dev={len(dev)}, test={len(test)})", flush=True)
                write_json(directory / "fits" / f"{test_speaker}.json", fitting)
            path = directory / "predictions_without_gold.jsonl.gz"
            write_jsonl(path, all_predictions)
            freeze_files[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "predictions_complete.json", {"files_sha256": freeze_files,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prediction_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "all_policies_reported": [s["name"] for s in config["strategies"]],
        "test_labels_joined": False})
    print("All outer predictions frozen. Evaluation can now join test labels.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "infer", "predict"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.config)
    elif args.command == "infer":
        infer(args.output, args.workers)
    else:
        predict(args.output)


if __name__ == "__main__":
    main()
