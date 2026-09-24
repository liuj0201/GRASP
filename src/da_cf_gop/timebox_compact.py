"""Registered compact CF fusion, with nested speaker-only model selection.

This module writes outer predictions without joining outer labels. Metrics are
deliberately not implemented here. The original final experiment is untouched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .dual_experiment import add_gold
from .dual_prior import estimate_error_prior, serializable_prior
from .final_experiment import (
    OLD, OUT, ROOT, RAW, calibration, log_ratio, output_rows, phone_design,
    retained_rows, speaker_weights,
)
from .parikh_experiment import read_jsonl, write_json, write_jsonl
from .phonology import PHONE_TO_CTC_ID

DESTINATION = ROOT / "artifacts/evaluation/optimization_20260905_1h/compact"
PROTOCOL = ROOT / "configs/timebox_compact_20260905.json"
METHOD_COLUMNS = {
    "compact_graph": [0, 1, 2, 3],
    "compact_graph_cf": [0, 1, 2, 3, 4, 5],
    "compact_graph_iso": [0, 1, 2, 3, 6, 7, 8],
    "compact_full": list(range(9)),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact_prior(records, speakers):
    """Occurrence and substitution identity have separate fixed smoothers."""
    speakers = tuple(sorted(speakers))
    selected = [r for r in records if r["speaker"] in speakers and r["quality_exclusion"] is None]
    arguments = dict(training_speakers=speakers, effective_tokens=1000.,
                     min_phone_tokens=20, min_phone_speakers=2)
    prior = estimate_error_prior(selected, alpha=50., **arguments)
    identity = estimate_error_prior(selected, alpha=10., **arguments)
    prior["sub_probs"] = identity["sub_probs"]
    sub_counts = np.zeros(40)
    sub_speakers = [set() for _ in range(40)]
    for record in selected:
        for token in record["labels"]["tokens"]:
            if token["stable_event"] and token["diagnostic"] and token["event_type"] == "substitution":
                p = PHONE_TO_CTC_ID[record["canonical_phones"][token["phone_index"]]]
                sub_counts[p] += 1
                sub_speakers[p].add(record["speaker"])
    prior["substitution_count"] = sub_counts
    prior["substitution_speakers"] = np.asarray([len(s) for s in sub_speakers])
    prior["identity_reliability"] = sub_counts / (sub_counts + 10.) * np.minimum(
        1., prior["substitution_speakers"] / 2.)
    prior["identity_alpha"] = 10.
    return prior


def compact_features(rows, priors):
    """Nine numeric cues; CF modifies error evidence, never acceptability.

    Identity support is a Bayes factor between a reliability-tempered learned
    substitution distribution and the uniform substitution distribution. The
    posterior weights come from the fixed generic candidate graph.
    """
    result = []
    for row in rows:
        prior = priors[row["speaker"]] if "op_probs" not in priors else priors
        p = PHONE_TO_CTC_ID[row["target"]]
        allowed = [PHONE_TO_CTC_ID[q] for q in row["acceptable"]]
        ok, sub, dele = row["event_posteriors"]
        graph = [-row["gop"], -row["viterbi_gop"], log_ratio(sub, ok), log_ratio(dele, ok)]
        operation = prior["op_probs"][p]
        occurrence = log_ratio(operation[1:].sum(), operation[0]) - np.log(.1 / .9)
        q = np.asarray(prior["sub_probs"][p]).copy()
        q[0] = 0.
        q[allowed] = 0.
        q /= q.sum()
        uniform = np.ones(40)
        uniform[0] = 0.
        uniform[allowed] = 0.
        uniform /= uniform.sum()
        reliability = prior["identity_reliability"][p]
        q = reliability * q + (1. - reliability) * uniform
        mass = np.asarray(row["substitution_posteriors"])
        denominator = float(np.dot(mass, uniform))
        identity = float(log_ratio(np.dot(mass, q), denominator)) if denominator > 0 else 0.
        isolated = [-row["baseline"][name] for name in RAW[1:]]
        result.append(graph + [occurrence, identity] + isolated)
    x = np.asarray(result, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def fit_model(rows, x, candidate):
    if candidate["transform"] == "normal_rank":
        scaler = QuantileTransformer(n_quantiles=min(256, len(rows)), output_distribution="normal",
                                     random_state=20260905, subsample=100000)
    else:
        scaler = StandardScaler()
    xt = phone_design(rows, scaler.fit_transform(x), interactions=False)
    if candidate["family"] == "logistic":
        classifier = LogisticRegression(C=candidate["C"], max_iter=1200, solver="lbfgs")
    else:
        classifier = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=candidate["leaves"],
            min_samples_leaf=40, l2_regularization=10., learning_rate=.05,
            early_stopping=False, random_state=20260905)
    classifier.fit(xt, [r["gold_error"] for r in rows], sample_weight=speaker_weights(rows))
    return {"scaler": scaler, "model": classifier, "interactions": False}


def score_model(model, rows, x):
    design = phone_design(rows, model["scaler"].transform(x), interactions=False)
    return model["model"].decision_function(design)


class FeatureBuilder:
    def __init__(self, records):
        self.records = records
        self.prior_cache = {}

    def prior(self, speakers):
        speakers = tuple(sorted(speakers))
        if speakers not in self.prior_cache:
            self.prior_cache[speakers] = compact_prior(self.records, speakers)
        return self.prior_cache[speakers]

    def training(self, rows, speakers):
        priors = {s: self.prior([t for t in speakers if t != s]) for s in speakers}
        return compact_features(rows, priors), {s: p["training_speakers"] for s, p in priors.items()}


def select_candidate(train, train_speakers, builder, candidates):
    """Each inner split refits every prior and scaler without its heldout speaker."""
    ap = {method: [[] for _ in candidates] for method in METHOD_COLUMNS}
    inner_provenance = []
    for heldout in train_speakers:
        fitting_speakers = [s for s in train_speakers if s != heldout]
        fitting = [r for r in train if r["speaker"] in fitting_speakers]
        validation = [r for r in train if r["speaker"] == heldout]
        xt, crossfit_sources = builder.training(fitting, fitting_speakers)
        xv = compact_features(validation, builder.prior(fitting_speakers))
        assert all(heldout not in speakers for speakers in crossfit_sources.values())
        for method, columns in METHOD_COLUMNS.items():
            for index, candidate in enumerate(candidates):
                model = fit_model(fitting, xt[:, columns], candidate)
                score = score_model(model, validation, xv[:, columns])
                ap[method][index].append(float(average_precision_score(
                    [r["gold_error"] for r in validation], score)))
        inner_provenance.append({"validation_speaker": heldout,
            "fit_speakers": fitting_speakers, "training_prior_sources": crossfit_sources,
            "validation_prior_sources": fitting_speakers,
            "transform_fit_speakers": fitting_speakers,
            "n_fit_tokens": len(fitting), "n_validation_tokens": len(validation)})
    results = {}
    for method in METHOD_COLUMNS:
        scores = [float(np.mean(values)) for values in ap[method]]
        selected = max(range(len(candidates)), key=lambda i: (scores[i], -i))
        results[method] = {"selected_index": selected, "selected": candidates[selected],
            "inner_macro_ap": scores[selected], "search": [
                {"candidate": c, "inner_speaker_ap": values, "inner_macro_ap": score}
                for c, values, score in zip(candidates, ap[method], scores)]}
    return results, inner_provenance


def run(output=DESTINATION, protocol_path=PROTOCOL):
    output, protocol_path = Path(output), Path(protocol_path)
    config = json.loads(protocol_path.read_text(encoding="utf-8"))
    base = json.loads((ROOT / "configs/final_v3_prompt_clean.json").read_text(encoding="utf-8"))
    assert config["excluded_events"] == base["excluded_events"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_identity.json").exists():
        raise FileExistsError("A registered run cannot be overwritten; use a new explicit output directory.")
    write_json(output / "config.json", config)
    write_json(output / "run_identity.json", {"started_utc": utc_now(),
        "protocol_sha256": file_sha256(protocol_path), "source_sha256": file_sha256(__file__),
        "fixed_evidence_sha256": file_sha256(OUT / "fixed_evidence_without_gold.jsonl.gz"),
        "base_protocol_sha256": file_sha256(ROOT / "configs/final_v3_prompt_clean.json")})
    evidence = retained_rows(read_jsonl(OUT / "fixed_evidence_without_gold.jsonl.gz"), base)
    assert not any(any(k.startswith("gold") for k in row) for row in evidence)
    started = time.perf_counter()
    totals = {}
    with threadpool_limits(limits=2):
        for cohort in ("primary5", "sensitivity7"):
            directory = output / cohort
            directory.mkdir()
            (directory / "models").mkdir()
            cohort_config = {**base, "optimization_protocol": config}
            write_json(directory / "config.json", cohort_config)
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
                print(f"{cohort}/{test_speaker}: selecting via {len(train_speakers)} inner speakers", flush=True)
                selected, inner = select_candidate(train, train_speakers, builder, config["candidates"])
                xt, sources = builder.training(train, train_speakers)
                prior = builder.prior(train_speakers)
                xd = compact_features(dev, prior)
                xx = compact_features(test, prior)
                predictions = []
                for method, columns in METHOD_COLUMNS.items():
                    model = fit_model(train, xt[:, columns], selected[method]["selected"])
                    dev_score = score_model(model, dev, xd[:, columns])
                    test_score = score_model(model, test, xx[:, columns])
                    cal = calibration(dev_score, [r["gold_error"] for r in dev])
                    model_path = directory / "models" / f"{test_speaker}.{method}.joblib"
                    joblib.dump({"binary": model, "feature_columns": columns,
                        "feature_protocol": config["feature_definitions"], "cf_prior": prior,
                        "calibration": cal, "fold": fold, "method": method}, model_path)
                    selected[method]["calibration"] = cal
                    selected[method]["model_sha256"] = file_sha256(model_path)
                    predictions.extend(output_rows(test, method, fold["fold"], test_score, cal))
                assert not any(any(k.startswith("gold") for k in r) for r in predictions)
                write_jsonl(directory / f"{test_speaker}.predictions_without_gold.jsonl.gz", predictions)
                write_json(directory / f"{test_speaker}.fit.json", {"fold": fold, "methods": selected,
                    "inner_splits": inner, "training_crossfit_speakers": sources,
                    "cf_prior": serializable_prior(prior), "n_train": len(train),
                    "n_development": len(dev), "n_test_predictions_per_method": len(test),
                    "selection_speakers": train_speakers, "calibration_speakers": [dev_speaker],
                    "outer_test_labels_joined": False})
                all_predictions.extend(predictions)
                print(f"{cohort}/{test_speaker}: frozen {len(predictions)} predictions; no outer metric", flush=True)
            write_jsonl(directory / "predictions_without_gold.jsonl.gz", all_predictions)
            totals[cohort] = len(all_predictions)
            write_json(directory / "prediction_complete.json", {"completed_utc": utc_now(),
                "n_predictions": len(all_predictions), "outer_evaluation_performed": False,
                "prediction_sha256": file_sha256(directory / "predictions_without_gold.jsonl.gz")})
    write_json(output / "prediction_complete.json", {"completed_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started, "counts": totals,
        "outer_evaluation_performed": False})
    print(f"Both cohorts frozen in {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    run(args.output, args.protocol)


if __name__ == "__main__":
    main()
