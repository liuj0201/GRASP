"""Patient-balanced ranking attempt; outer evaluation is deliberately separate."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from . import final_experiment as final
from .dual_experiment import add_gold
from .parikh_experiment import read_jsonl, write_json, write_jsonl
from .timebox_compact import FeatureBuilder, compact_features

CONFIG = final.ROOT / "configs/timebox_ranking_20260905.json"
OUTPUT = final.ROOT / "artifacts/evaluation/optimization_20260905_1h/ranking"
METHOD_COLUMNS = {"ranking_graph": [0, 1, 2, 3],
                  "ranking_graph_cf": [0, 1, 2, 3, 4, 5],
                  "ranking_graph_iso": [0, 1, 2, 3, 6, 7, 8],
                  "ranking_full": list(range(9))}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design(rows, numeric):
    # Preserve graph expressiveness while avoiding per-phone auxiliary slopes.
    graph = final.phone_design(rows, numeric[:, :4], interactions=True)
    return np.c_[graph, numeric[:, 4:]]


def sampled_pairs(rows, maximum=512):
    rng = np.random.default_rng(20260905)
    first, second, weights = [], [], []
    speakers = sorted({row["speaker"] for row in rows})
    for speaker in speakers:
        positive = np.asarray([i for i, row in enumerate(rows) if row["speaker"] == speaker and row["gold_error"]])
        negative = np.asarray([i for i, row in enumerate(rows) if row["speaker"] == speaker and not row["gold_error"]])
        assert len(positive) and len(negative)
        count = min(maximum, len(positive) * len(negative))
        indices = rng.choice(len(positive) * len(negative), count, replace=False)
        first.extend(positive[indices // len(negative)])
        second.extend(negative[indices % len(negative)])
        weights.extend([1. / len(speakers) / count] * count)
    return np.asarray(first), np.asarray(second), np.asarray(weights)


def objective(parameters, x, labels, point_weight, difference, pair_weight, ridge):
    coefficient, intercept = parameters[:-1], parameters[-1]
    response = x @ coefficient + intercept
    point_loss = point_weight @ (np.logaddexp(0., response) - labels * response)
    error = point_weight * (expit(response) - labels)
    pair_response = difference @ coefficient
    pair_loss = pair_weight @ np.logaddexp(0., -pair_response)
    pair_error = -pair_weight * expit(-pair_response)
    loss = point_loss + .5 * pair_loss + .5 * ridge * np.dot(coefficient, coefficient)
    gradient = np.r_[x.T @ error + .5 * difference.T @ pair_error + ridge * coefficient, error.sum()]
    return loss, gradient


def fit_model(rows, numeric, candidate):
    if candidate["transform"] == "normal_rank":
        scaler = QuantileTransformer(n_quantiles=min(256, len(rows)), output_distribution="normal",
                                     random_state=20260905, subsample=100000)
    else:
        scaler = StandardScaler()
    x = design(rows, scaler.fit_transform(numeric))
    labels = np.asarray([r["gold_error"] for r in rows], dtype=float)
    weight = final.speaker_weights(rows) / len(rows)
    first, second, pair_weight = sampled_pairs(rows)
    difference = x[first] - x[second]
    initial = np.zeros(x.shape[1] + 1)
    prevalence = float(weight @ labels)
    initial[-1] = np.log(prevalence / (1. - prevalence))
    optimum = minimize(objective, initial, args=(x, labels, weight, difference, pair_weight, candidate["ridge"]),
                       method="L-BFGS-B", jac=True, options={"maxiter": 180, "ftol": 1e-9, "gtol": 1e-5})
    assert optimum.success, optimum.message
    return {"scaler": scaler, "coefficient": optimum.x[:-1], "intercept": optimum.x[-1],
            "iterations": int(optimum.nit), "objective": float(optimum.fun),
            "n_pairs": len(first), "fit_speakers": sorted({r["speaker"] for r in rows})}


def score_model(model, rows, numeric):
    return design(rows, model["scaler"].transform(numeric)) @ model["coefficient"] + model["intercept"]


def run():
    protocol = json.loads(CONFIG.read_text())
    deadline = datetime.fromisoformat(protocol["deadline_utc"].replace("Z", "+00:00"))
    base = json.loads((final.ROOT / "configs/final_v3_prompt_clean.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert not (OUTPUT / "run_identity.json").exists()
    write_json(OUTPUT / "config.json", protocol)
    write_json(OUTPUT / "run_identity.json", {"started_utc": utc(), "source_sha256": digest(__file__),
        "feature_source_sha256": digest(Path(__file__).with_name("timebox_compact.py")),
        "protocol_sha256": digest(CONFIG), "outer_evaluation_performed": False})
    evidence = final.retained_rows(read_jsonl(final.OUT / "fixed_evidence_without_gold.jsonl.gz"), base)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        for cohort in ("primary5", "sensitivity7"):
            directory = OUTPUT / cohort
            directory.mkdir()
            (directory / "models").mkdir()
            write_json(directory / "config.json", {**base, "optimization_protocol": protocol})
            folds = json.loads((final.OLD / cohort / "folds.json").read_text())
            write_json(directory / "folds.json", folds)
            records = final.retained_rows(read_jsonl(final.OLD / cohort / "evaluation_labels.jsonl.gz"), base)
            predictions = []
            for fold in folds:
                speakers = fold["training_patients"]
                dev_speaker, test_speaker = fold["development_patient"], fold["test_patient"]
                fitting_records = [r for r in records if r["speaker"] in speakers]
                builder = FeatureBuilder(fitting_records)
                train = add_gold([r for r in evidence if r["speaker"] in speakers], fitting_records)
                dev = add_gold([r for r in evidence if r["speaker"] == dev_speaker],
                               [r for r in records if r["speaker"] == dev_speaker])
                test = [r for r in evidence if r["speaker"] == test_speaker]
                scores = {m: [[] for _ in protocol["candidates"]] for m in METHOD_COLUMNS}
                inner_splits = []
                print(f"{cohort}/{test_speaker}: inner ranking selection", flush=True)
                for heldout in speakers:
                    inner_speakers = [s for s in speakers if s != heldout]
                    fitting = [r for r in train if r["speaker"] in inner_speakers]
                    validation = [r for r in train if r["speaker"] == heldout]
                    xt, sources = builder.training(fitting, inner_speakers)
                    xv = compact_features(validation, builder.prior(inner_speakers))
                    assert all(heldout not in value for value in sources.values())
                    for method, columns in METHOD_COLUMNS.items():
                        for i, candidate in enumerate(protocol["candidates"]):
                            if datetime.now(timezone.utc) >= deadline:
                                raise TimeoutError("Ranking fit deadline; no outer evaluation.")
                            model = fit_model(fitting, xt[:, columns], candidate)
                            response = score_model(model, validation, xv[:, columns])
                            scores[method][i].append(float(average_precision_score(
                                [r["gold_error"] for r in validation], response)))
                    inner_splits.append({"heldout": heldout, "fit_speakers": inner_speakers,
                                         "training_prior_sources": sources, "validation_prior_sources": inner_speakers})
                xt, sources = builder.training(train, speakers)
                prior = builder.prior(speakers)
                xd, xx = compact_features(dev, prior), compact_features(test, prior)
                fitted = {}
                fold_predictions = []
                for method, columns in METHOD_COLUMNS.items():
                    means = [float(np.mean(values)) for values in scores[method]]
                    chosen = max(range(len(means)), key=lambda i: (means[i], -i))
                    candidate = protocol["candidates"][chosen]
                    model = fit_model(train, xt[:, columns], candidate)
                    dev_score = score_model(model, dev, xd[:, columns])
                    test_score = score_model(model, test, xx[:, columns])
                    cal = final.calibration(dev_score, [r["gold_error"] for r in dev])
                    artifact = directory / "models" / f"{test_speaker}.{method}.joblib"
                    joblib.dump({"model": model, "columns": columns, "cf_prior": prior,
                                 "calibration": cal, "fold": fold, "method": method}, artifact)
                    fitted[method] = {"selected": candidate, "inner_macro_ap": means[chosen],
                        "search": [{"candidate": c, "inner_speaker_ap": v, "inner_macro_ap": mean}
                                   for c, v, mean in zip(protocol["candidates"], scores[method], means)],
                        "calibration": cal, "model_sha256": digest(artifact)}
                    fold_predictions.extend(final.output_rows(test, method, fold["fold"], test_score, cal))
                write_json(directory / f"{test_speaker}.fit.json", {"fold": fold, "methods": fitted,
                    "inner_splits": inner_splits, "training_prior_sources": sources,
                    "selection_speakers": speakers, "calibration_speakers": [dev_speaker],
                    "outer_test_labels_joined": False})
                write_jsonl(directory / f"{test_speaker}.predictions_without_gold.jsonl.gz", fold_predictions)
                predictions.extend(fold_predictions)
                print(f"{cohort}/{test_speaker}: {len(fold_predictions)} raw predictions saved", flush=True)
            file = directory / "predictions_without_gold.jsonl.gz"
            write_jsonl(file, predictions)
            write_json(directory / "prediction_complete.json", {"completed_utc": utc(),
                "n_predictions": len(predictions), "prediction_sha256": digest(file), "outer_evaluation_performed": False})
    write_json(OUTPUT / "prediction_complete.json", {"completed_utc": utc(),
        "elapsed_seconds": time.perf_counter() - started, "outer_evaluation_performed": False})


if __name__ == "__main__":
    run()
