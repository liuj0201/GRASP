"""Fixed probability ensemble of the eight preregistered compact classifiers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from .dual_experiment import add_gold
from .dual_prior import serializable_prior
from .final_experiment import OLD, OUT, ROOT, calibration, output_rows, retained_rows
from .parikh_experiment import read_jsonl, write_json, write_jsonl
from .timebox_compact import (
    FeatureBuilder, METHOD_COLUMNS as COMPACT_COLUMNS, PROTOCOL as COMPACT_PROTOCOL,
    compact_features, file_sha256, fit_model, score_model, utc_now,
)

PROTOCOL = ROOT / "configs/timebox_ensemble_20260905.json"
DESTINATION = ROOT / "artifacts/evaluation/optimization_20260905_1h/ensemble"
METHOD_COLUMNS = {name.replace("compact_", "ensemble_"): columns
                  for name, columns in COMPACT_COLUMNS.items()}


def ensemble_score(members, candidates, rows, x):
    probabilities = np.asarray([expit(score_model(model, rows, x)) for model in members])
    logistic = [i for i, candidate in enumerate(candidates) if candidate["family"] == "logistic"]
    histogram = [i for i, candidate in enumerate(candidates) if candidate["family"] == "histogram"]
    mixture = .5 * probabilities[logistic].mean(axis=0) + .5 * probabilities[histogram].mean(axis=0)
    return logit(np.clip(mixture, 1e-12, 1. - 1e-12))


def run(output=DESTINATION, protocol_path=PROTOCOL):
    output, protocol_path = Path(output), Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    compact_protocol = json.loads(COMPACT_PROTOCOL.read_text(encoding="utf-8"))
    base = json.loads((ROOT / "configs/final_v3_prompt_clean.json").read_text(encoding="utf-8"))
    assert protocol["excluded_events"] == base["excluded_events"]
    candidates = protocol["candidate_members"]
    assert candidates == compact_protocol["candidates"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_identity.json").exists():
        raise FileExistsError("Cannot overwrite an existing ensemble run.")
    write_json(output / "config.json", protocol)
    write_json(output / "run_identity.json", {"started_utc": utc_now(),
        "protocol_sha256": file_sha256(protocol_path), "source_sha256": file_sha256(__file__),
        "compact_source_sha256": file_sha256(ROOT / "src/da_cf_gop/timebox_compact.py"),
        "compact_protocol_sha256": file_sha256(COMPACT_PROTOCOL),
        "fixed_evidence_sha256": file_sha256(OUT / "fixed_evidence_without_gold.jsonl.gz"),
        "base_protocol_sha256": file_sha256(ROOT / "configs/final_v3_prompt_clean.json")})
    evidence = retained_rows(read_jsonl(OUT / "fixed_evidence_without_gold.jsonl.gz"), base)
    assert not any(any(k.startswith("gold") for k in row) for row in evidence)
    totals, started = {}, time.perf_counter()
    with threadpool_limits(limits=2):
        for cohort in ("primary5", "sensitivity7"):
            directory = output / cohort
            directory.mkdir()
            (directory / "models").mkdir()
            write_json(directory / "config.json", {**base, "optimization_protocol": protocol})
            folds = json.loads((OLD / cohort / "folds.json").read_text(encoding="utf-8"))
            write_json(directory / "folds.json", folds)
            records = retained_rows(read_jsonl(OLD / cohort / "evaluation_labels.jsonl.gz"), base)
            all_predictions = []
            for fold in folds:
                train_speakers = fold["training_patients"]
                dev_speaker, test_speaker = fold["development_patient"], fold["test_patient"]
                assert not set(train_speakers) & {dev_speaker, test_speaker}
                training_records = [r for r in records if r["speaker"] in train_speakers]
                builder = FeatureBuilder(training_records)
                train = add_gold([r for r in evidence if r["speaker"] in train_speakers], training_records)
                dev = add_gold([r for r in evidence if r["speaker"] == dev_speaker],
                    [r for r in records if r["speaker"] == dev_speaker])
                test = [r for r in evidence if r["speaker"] == test_speaker]
                xt, sources = builder.training(train, train_speakers)
                prior = builder.prior(train_speakers)
                xd, xx = compact_features(dev, prior), compact_features(test, prior)
                method_fits, predictions = {}, []
                print(f"{cohort}/{test_speaker}: fitting fixed8-member ensembles", flush=True)
                for method, columns in METHOD_COLUMNS.items():
                    members = [fit_model(train, xt[:, columns], candidate) for candidate in candidates]
                    dev_score = ensemble_score(members, candidates, dev, xd[:, columns])
                    test_score = ensemble_score(members, candidates, test, xx[:, columns])
                    cal = calibration(dev_score, [r["gold_error"] for r in dev])
                    model_path = directory / "models" / f"{test_speaker}.{method}.joblib"
                    joblib.dump({"members": members, "candidates": candidates,
                        "family_weights": {"logistic": .5, "histogram": .5},
                        "feature_columns": columns, "cf_prior": prior,
                        "calibration": cal, "fold": fold, "method": method}, model_path)
                    method_fits[method] = {"members": candidates, "n_members": len(members),
                        "component_weights": [1. / 12.] * 6 + [.25] * 2,
                        "selection": None, "calibration": cal,
                        "model_sha256": file_sha256(model_path),
                        "supervised_fit_speakers": train_speakers,
                        "transform_fit_speakers": train_speakers}
                    predictions.extend(output_rows(test, method, fold["fold"], test_score, cal))
                assert not any(any(k.startswith("gold") for k in r) for r in predictions)
                write_jsonl(directory / f"{test_speaker}.predictions_without_gold.jsonl.gz", predictions)
                write_json(directory / f"{test_speaker}.fit.json", {"fold": fold,
                    "methods": method_fits, "training_crossfit_speakers": sources,
                    "cf_prior": serializable_prior(prior), "n_train": len(train),
                    "n_development": len(dev), "n_test_predictions_per_method": len(test),
                    "selection_speakers": [], "calibration_speakers": [dev_speaker],
                    "outer_test_labels_joined": False})
                all_predictions.extend(predictions)
                print(f"{cohort}/{test_speaker}: frozen {len(predictions)} predictions", flush=True)
            write_jsonl(directory / "predictions_without_gold.jsonl.gz", all_predictions)
            totals[cohort] = len(all_predictions)
            write_json(directory / "prediction_complete.json", {"completed_utc": utc_now(),
                "n_predictions": len(all_predictions), "outer_evaluation_performed": False,
                "prediction_sha256": file_sha256(directory / "predictions_without_gold.jsonl.gz")})
    write_json(output / "prediction_complete.json", {"completed_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started, "counts": totals,
        "outer_evaluation_performed": False})
    print(f"Both fixed-ensemble cohorts frozen in {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    run(args.output, args.protocol)


if __name__ == "__main__":
    main()
