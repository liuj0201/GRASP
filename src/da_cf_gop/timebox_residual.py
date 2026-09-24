"""Registered training-speaker cross-validated residual score fusion.

This module writes raw outer predictions only. It has no evaluation entry point.
All four variants share a graph detector chosen on original training patients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import final_experiment as fe
from .dual_prior import generic_prior, serializable_prior
from .parikh_experiment import read_jsonl, write_json, write_jsonl


ROOT = fe.ROOT
CONFIG = ROOT / "configs/final_v3_prompt_clean.json"
OUTPUT = ROOT / "artifacts/evaluation/optimization_20260905_1h/residual"
METHODS = (
    "residual_graph_da", "residual_graph_cf_da",
    "residual_graph_isolated_da", "residual_full",
)
BACKBONES = (
    {"family": "histogram", "leaves": 4, "iterations": 120},
    {"family": "histogram", "leaves": 8, "iterations": 60},
)
STRENGTHS = (0.0, 0.15, 0.4)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inner_roles(speakers):
    """The excluded patient supplies selection labels, never fit information."""
    return [
        {"held_out": speaker, "training": [s for s in speakers if s != speaker]}
        for speaker in sorted(speakers)
    ]


def candidate_weights(method):
    cf = STRENGTHS if method in ("residual_graph_cf_da", "residual_full") else (0.0,)
    isolated = STRENGTHS if method in ("residual_graph_isolated_da", "residual_full") else (0.0,)
    return [{"cf": c, "isolated": i} for c in cf for i in isolated]


def mix_scores(graph, isolated, cf_delta, weights):
    graph = np.asarray(graph)
    return graph + weights["cf"] * np.asarray(cf_delta) + weights["isolated"] * (np.asarray(isolated) - graph)


def cf_log_odds_delta(rows, prior):
    """Local prior replacement, in the same natural-log units as model logits."""
    _, cf = fe.continuous_features(rows, prior, "graph_da")
    generic_error = np.asarray([r["event_posteriors"][1] + r["event_posteriors"][2] for r in rows])
    generic_ok = np.asarray([r["event_posteriors"][0] for r in rows])
    return fe.log_ratio(cf[:, 1:].sum(axis=1), cf[:, 0]) - fe.log_ratio(generic_error, generic_ok), cf


def fit_isolated(train):
    # CF is not an input to this expert. A generic prior only satisfies the
    # existing shared feature extractor's signature.
    x, _ = fe.continuous_features(train, generic_prior(), "isolated_da")
    scaler = StandardScaler().fit(x)
    design = fe.phone_design(train, scaler.transform(x), interactions=False)
    model = LogisticRegression(C=0.1, max_iter=1500, solver="lbfgs")
    model.fit(design, [r["gold_error"] for r in train], sample_weight=fe.speaker_weights(train))
    return {"scaler": scaler, "model": model, "interactions": False}


def score_isolated(model, rows):
    x, _ = fe.continuous_features(rows, generic_prior(), "isolated_da")
    return fe.model_score(model, rows, x)


def macro_ap(rows, scores):
    labels = np.asarray([r["gold_error"] for r in rows])
    speaker = np.asarray([r["speaker"] for r in rows])
    values = {s: float(average_precision_score(labels[speaker == s], np.asarray(scores)[speaker == s]))
              for s in sorted(set(speaker))}
    return float(np.mean(list(values.values()))), values


def get_labeled(evidence, records, speakers):
    speakers = set(speakers)
    return fe.add_gold([r for r in evidence if r["speaker"] in speakers],
                       [r for r in records if r["speaker"] in speakers])


def collect_oof(evidence, training_records, training_speakers, config):
    """Every patient's predictions use models and priors from other patients."""
    joined, graph_scores, isolated_scores, cf_deltas, provenance = [], [[], []], [], [], []
    for role in inner_roles(training_speakers):
        held_out, fit_speakers = role["held_out"], role["training"]
        train = get_labeled(evidence, training_records, fit_speakers)
        heldout = get_labeled(evidence, training_records, [held_out])
        fit_records = [r for r in training_records if r["speaker"] in fit_speakers]
        prior = fe.prior_for(fit_records, fit_speakers, config)
        xt, _ = fe.continuous_features(train, generic_prior(), "graph_da")
        xh, _ = fe.continuous_features(heldout, generic_prior(), "graph_da")
        for i, candidate in enumerate(BACKBONES):
            _, score = fe.fit_predictor(train, heldout, xt, xh, candidate)
            graph_scores[i].extend(score.tolist())
        isolated_model = fit_isolated(train)
        isolated_scores.extend(score_isolated(isolated_model, heldout).tolist())
        delta, _ = cf_log_odds_delta(heldout, prior)
        cf_deltas.extend(delta.tolist())
        joined.extend(heldout)
        provenance.append({**role, "model_fit_speakers": fit_speakers,
            "scaler_fit_speakers": fit_speakers, "cf_prior_speakers": prior["training_speakers"],
            "n_fit": len(train), "n_held_out": len(heldout)})
    return (joined, np.asarray(graph_scores), np.asarray(isolated_scores),
            np.asarray(cf_deltas), provenance)


def registry(config_path):
    return {
        "registered_utc": utc_now(), "method": "residual score fusion",
        "source_sha256": file_hash(__file__), "base_config_sha256": file_hash(config_path),
        "source": str(Path(__file__).relative_to(ROOT)),
        "information_flow": "Original training-speaker LOSO selects backbone and weights. Original dev only calibrates. Outer test supplies raw evidence only.",
        "backbone_candidates": list(BACKBONES),
        "backbone_selection": "Highest original-training LOSO speaker macro AP, ties first listed; shared by every ablation.",
        "isolated_expert": {"family": "logistic", "C": 0.1, "numeric_cues": 3,
                            "phone_main_effects": 39, "interactions": False,
                            "speaker_equal_weights": True},
        "score": "g + w_cf*(log(CF_error/CF_OK)-log(generic_error/generic_OK)) + w_iso*(isolated_logit-g)",
        "weight_candidates": {m: candidate_weights(m) for m in METHODS},
        "weight_selection": "Original-training LOSO speaker macro AP; exact ties prefer lower sum of auxiliary strengths, then earlier listed.",
        "matched_ablations": list(METHODS),
        "diagnosis": "Not optimized or produced in this detection-only branch; candidate_event unspecified.",
        "zero_weights": "Explicitly permitted and reported as degeneration; zero-weight full is not evidence for auxiliary modules.",
        "prior": "Unchanged final_v3_prompt_clean alpha/effective counts and input exclusion.",
        "independence_limit": "Retrospective development; historical outer outcomes previously inspected, this branch selects without accessing new outer metrics.",
        "evaluation": "No outer evaluation in this program. Write both cohort predictions and barrier first.",
    }


def predict_cohort(cohort, config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "prediction_complete.json").exists():
        raise FileExistsError(f"Predictions already complete: {directory}")
    if (directory / "metrics.json").exists():
        raise FileExistsError(f"Outer metrics already exist: {directory}")
    write_json(directory / "config.json", config)
    speakers = set(config["cohorts"][cohort])
    evidence = [r for r in fe.retained_rows(read_jsonl(fe.OUT / "fixed_evidence_without_gold.jsonl.gz"), config)
                if r["speaker"] in speakers]
    # Records are split before attachment or prior fitting. No test record is
    # passed into collect_oof, fitting, calibration, or output_rows.
    records = [r for r in fe.retained_rows(read_jsonl(fe.OLD / cohort / "evaluation_labels.jsonl.gz"), config)
               if r["speaker"] in speakers]
    folds = fe.read_jsonl_folds(fe.OLD / cohort)
    write_json(directory / "folds.json", folds)
    all_predictions = []
    started = time.perf_counter()
    for fold in folds:
        train_speakers = fold["training_patients"]
        dev_speaker, test_speaker = fold["development_patient"], fold["test_patient"]
        assert not set(train_speakers) & {dev_speaker, test_speaker}
        print(f"{cohort}/{test_speaker}: fitting training-only LOSO", flush=True)
        training_records = [r for r in records if r["speaker"] in train_speakers]
        training_evidence = [r for r in evidence if r["speaker"] in train_speakers]
        oof_rows, oof_graph, oof_iso, oof_cf, oof_roles = collect_oof(
            training_evidence, training_records, train_speakers, config)
        backbone_search = []
        for i, parameters in enumerate(BACKBONES):
            score, per_speaker = macro_ap(oof_rows, oof_graph[i])
            backbone_search.append({"parameters": parameters, "inner_macro_ap": score,
                                    "inner_per_speaker_ap": per_speaker})
        backbone_index = max(range(len(BACKBONES)), key=lambda i: (backbone_search[i]["inner_macro_ap"], -i))
        selected_backbone = BACKBONES[backbone_index]
        oof_g = oof_graph[backbone_index]
        train = get_labeled(training_evidence, training_records, train_speakers)
        dev = get_labeled(evidence, records, [dev_speaker])
        test = [r for r in evidence if r["speaker"] == test_speaker]
        assert not any(any(k.startswith("gold") for k in r) for r in test)
        xt, _ = fe.continuous_features(train, generic_prior(), "graph_da")
        xd, _ = fe.continuous_features(dev, generic_prior(), "graph_da")
        xx, _ = fe.continuous_features(test, generic_prior(), "graph_da")
        graph_model, dev_graph = fe.fit_predictor(train, dev, xt, xd, selected_backbone)
        test_graph = fe.model_score(graph_model, test, xx)
        iso_model = fit_isolated(train)
        dev_iso, test_iso = score_isolated(iso_model, dev), score_isolated(iso_model, test)
        prior = fe.prior_for(training_records, train_speakers, config)
        dev_cf, _ = cf_log_odds_delta(dev, prior)
        test_cf, _ = cf_log_odds_delta(test, prior)
        fit = {"fold": fold, "n_train": len(train), "n_development": len(dev),
               "n_test_predictions": len(test), "inner_roles": oof_roles,
               "backbone_search": backbone_search, "selected_backbone": selected_backbone,
               "cf_prior": serializable_prior(prior), "methods": {}}
        predictions = []
        models = directory / "models"
        models.mkdir(exist_ok=True)
        for method in METHODS:
            candidates = candidate_weights(method)
            search = []
            for weights in candidates:
                score, per_speaker = macro_ap(oof_rows, mix_scores(oof_g, oof_iso, oof_cf, weights))
                search.append({"weights": weights, "inner_macro_ap": score,
                               "inner_per_speaker_ap": per_speaker})
            index = max(range(len(search)), key=lambda i: (search[i]["inner_macro_ap"],
                        -sum(search[i]["weights"].values()), -i))
            weights = candidates[index]
            dev_score = mix_scores(dev_graph, dev_iso, dev_cf, weights)
            test_score = mix_scores(test_graph, test_iso, test_cf, weights)
            cal = fe.calibration(dev_score, [r["gold_error"] for r in dev])
            predictions.extend(fe.output_rows(test, method, fold["fold"], test_score, cal))
            fit["methods"][method] = {
                "search": search, "selected": search[index], "calibration": cal,
                "supervised_fit_speakers": train_speakers,
                "selection_speakers": train_speakers, "calibration_speakers": [dev_speaker],
                "cf_active": weights["cf"] > 0, "isolated_active": weights["isolated"] > 0,
                "full_degenerate": method == "residual_full" and not all(w > 0 for w in weights.values()),
                "model_path": f"models/{test_speaker}.{method}.joblib",
            }
            joblib.dump({"graph": graph_model, "isolated": iso_model, "weights": weights,
                         "cf_prior": prior, "calibration": cal, "method": method,
                         "training_speakers": train_speakers, "graph_parameters": selected_backbone},
                        models / f"{test_speaker}.{method}.joblib")
            print(f"  {method}: training-only selection frozen, weights={weights}", flush=True)
        write_json(directory / f"{test_speaker}.fit.json", fe.json_safe(fit))
        write_jsonl(directory / f"{test_speaker}.predictions_without_gold.jsonl.gz", predictions)
        write_jsonl(directory / f"{test_speaker}.training_oof.jsonl.gz", [
            {**{k: r[k] for k in ("event", "speaker", "phone_index", "target", "gold_error")},
             "graph_candidate_scores": oof_graph[:, i].tolist(), "isolated_score": float(oof_iso[i]),
             "cf_delta": float(oof_cf[i])}
            for i, r in enumerate(oof_rows)
        ])
        all_predictions.extend(predictions)
    path = directory / "predictions_without_gold.jsonl.gz"
    write_jsonl(path, all_predictions)
    write_json(directory / "prediction_complete.json", {
        "completed_utc": utc_now(), "n_predictions": len(all_predictions),
        "methods": list(METHODS), "elapsed_seconds": time.perf_counter() - started,
        "predictions_sha256": file_hash(path), "outer_evaluation_performed": False,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    registry_path = args.output / "registered_protocol.json"
    if registry_path.exists():
        raise FileExistsError(f"Registered run already exists: {registry_path}")
    write_json(registry_path, registry(args.config))
    config = json.loads(args.config.read_text())
    config["timebox_residual"] = {
        "registry_sha256": file_hash(registry_path), "primary_metric": "speaker macro AP",
        "methods": list(METHODS),
    }
    with threadpool_limits(limits=2):
        for cohort in ("primary5", "sensitivity7"):
            predict_cohort(cohort, config, args.output / cohort)
    write_json(args.output / "prediction_barrier.json", {
        "completed_utc": utc_now(), "outer_evaluation_performed": False,
        "registry_sha256": file_hash(registry_path),
        "cohorts": {c: json.loads((args.output / c / "prediction_complete.json").read_text())
                    for c in ("primary5", "sensitivity7")},
    })


if __name__ == "__main__":
    main()
