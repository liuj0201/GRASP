"""Speaker-first evaluation for DA-CF-GoP.

The public helpers in this module deliberately operate on JSON-compatible
records.  That keeps the evaluation artifact independently auditable and
prevents the evaluator from depending on a model implementation.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


DELETE = "<DEL>"
PHONE_VOCAB = frozenset(
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG "
    "OW OY P R S SH T TH UH UW V W Y Z ZH".split()
)
GOLD_EVENTS = frozenset({"correct", "substitution", "deletion"})
PRECISION_COVERAGES = (0.10, 0.25, 0.50, 0.75, 1.00)
REQUIRED_PREDICTION_KEYS = frozenset(
    {
        "method",
        "cohort",
        "fold",
        "speaker",
        "event",
        "phone_index",
        "target",
        "gold_event",
        "gold_realized",
        "stable",
        "gop",
        "p_error",
        "top_alt",
        "p_alt_given_error",
        "candidate_probabilities",
        "top2_margin",
    }
)


class MetricError(ValueError):
    """Raised when an evaluation record would make a metric ambiguous."""


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MetricError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise MetricError(f"{name} must be finite")
    return result


def _probability(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise MetricError(f"{name} must be in [0, 1]")
    return result


def _ranked_candidates(probabilities: Mapping[str, object]) -> list[str]:
    """Return a deterministic probability ranking.

    Lexicographic tie-breaking makes the ranking independent of JSON/dict
    insertion order, which is essential for deterministic top-k and MRR.
    """

    return [
        phone
        for phone, _ in sorted(
            ((str(phone), _probability(value, f"candidate {phone}"))
             for phone, value in probabilities.items()),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def validate_prediction_row(row: Mapping[str, object]) -> None:
    """Validate the frozen evaluation-row contract.

    Extra keys are permitted because an evaluation-only row may carry
    additional diagnostic values.  The required names themselves have no
    aliases: producers must conform to this contract rather than relying on
    evaluator-side guessing.
    """

    missing = REQUIRED_PREDICTION_KEYS.difference(row)
    if missing:
        raise MetricError(f"prediction row is missing keys: {sorted(missing)}")
    for key in ("method", "cohort", "fold", "speaker", "event", "target"):
        if not isinstance(row[key], str) or not str(row[key]).strip():
            raise MetricError(f"{key} must be a non-empty string")
    if type(row["phone_index"]) is not int or int(row["phone_index"]) < 0:
        raise MetricError("phone_index must be a non-negative integer")
    if type(row["stable"]) is not bool:
        raise MetricError("stable must be boolean")
    if row["gold_event"] not in GOLD_EVENTS:
        raise MetricError(f"gold_event must be one of {sorted(GOLD_EVENTS)}")

    target = str(row["target"])
    if target not in PHONE_VOCAB:
        raise MetricError("target is outside the frozen 39-phone vocabulary")
    realized = row["gold_realized"]
    if row["gold_event"] == "correct" and realized != target:
        raise MetricError("a correct token must have gold_realized == target")
    if row["gold_event"] == "substitution":
        if realized not in PHONE_VOCAB or realized == target:
            raise MetricError("a substitution needs a non-target realized phone")
    if row["gold_event"] == "deletion" and realized not in {None, DELETE}:
        raise MetricError("a deletion must use null or <DEL> as gold_realized")

    _finite_float(row["gop"], "gop")
    _probability(row["p_error"], "p_error")
    p_alt = _probability(row["p_alt_given_error"], "p_alt_given_error")
    margin = _probability(row["top2_margin"], "top2_margin")
    probabilities = row["candidate_probabilities"]
    if not isinstance(probabilities, Mapping) or len(probabilities) != 39:
        raise MetricError("candidate_probabilities must contain 38 phones and <DEL>")
    expected_candidates = (PHONE_VOCAB - {target}) | {DELETE}
    if set(probabilities) != expected_candidates:
        raise MetricError("candidate probabilities do not match the frozen error vocabulary")
    values = np.asarray(
        [_probability(value, f"candidate_probabilities[{phone}]")
         for phone, value in probabilities.items()],
        dtype=np.float64,
    )
    if not np.isclose(float(values.sum()), 1.0, atol=1e-6, rtol=0.0):
        raise MetricError("candidate_probabilities must sum to one")
    ranked = _ranked_candidates(probabilities)
    if row["top_alt"] != ranked[0]:
        raise MetricError("top_alt is inconsistent with candidate_probabilities")
    expected_alt = float(probabilities[ranked[0]])
    expected_margin = expected_alt - float(probabilities[ranked[1]])
    if not math.isclose(p_alt, expected_alt, abs_tol=1e-6, rel_tol=0.0):
        raise MetricError("p_alt_given_error is inconsistent with the candidate posterior")
    if not math.isclose(margin, expected_margin, abs_tol=1e-6, rel_tol=0.0):
        raise MetricError("top2_margin is inconsistent with the candidate posterior")
    if "candidate_log_scores" in row:
        scores = row["candidate_log_scores"]
        if not isinstance(scores, Mapping) or set(scores) != expected_candidates:
            raise MetricError("candidate_log_scores must use the frozen error vocabulary")
        finite_count = 0
        for phone, value in scores.items():
            if value is None:
                continue
            _finite_float(value, f"candidate_log_scores[{phone}]")
            finite_count += 1
        if finite_count == 0:
            raise MetricError("candidate_log_scores must contain finite retained scores")
    for key in (
        "predicted_error",
        "specific_output_authorized",
        "specific_prediction_authorized",
        "specific_abstain",
    ):
        if key in row and type(row[key]) is not bool:
            raise MetricError(f"{key} must be boolean")


def validate_prediction_rows(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise MetricError("prediction artifact is empty")
    seen: set[tuple[object, ...]] = set()
    for row in rows:
        validate_prediction_row(row)
        key = tuple(row[name] for name in (
            "method", "cohort", "fold", "speaker", "event", "phone_index"
        ))
        if key in seen:
            raise MetricError(f"duplicate prediction token: {key}")
        seen.add(key)


def binary_metrics(y_true: Sequence[int], p_error: Sequence[float]) -> dict[str, Any]:
    """Return calibrated binary-error metrics without emitting NaN JSON."""

    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(p_error, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise MetricError("y_true and p_error must be non-empty, equal-length vectors")
    if not np.isin(y, (0, 1)).all():
        raise MetricError("y_true must be binary")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise MetricError("p_error must contain finite probabilities")
    clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "n_error": int(y.sum()),
        "prevalence": float(y.mean()),
        "auprc": float(average_precision_score(y, p)) if two_classes else None,
        "auroc": float(roc_auc_score(y, p)) if two_classes else None,
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
    }


def substitution_rank_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Top-k and reciprocal rank on stable, realized substitutions only."""

    ranks: list[int] = []
    for row in rows:
        if not bool(row["stable"]) or row["gold_event"] != "substitution":
            continue
        probabilities = row["candidate_probabilities"]
        if not isinstance(probabilities, Mapping):
            raise MetricError("candidate_probabilities is not a mapping")
        realized = str(row["gold_realized"])
        ranking = _ranked_candidates(probabilities)
        if realized not in ranking:
            raise MetricError(f"gold realized phone is absent from candidates: {realized}")
        ranks.append(ranking.index(realized) + 1)
    if not ranks:
        return {"n": 0, "top1": None, "top3": None, "mrr": None}
    array = np.asarray(ranks, dtype=np.float64)
    return {
        "n": len(ranks),
        "top1": float(np.mean(array <= 1)),
        "top3": float(np.mean(array <= 3)),
        "mrr": float(np.mean(1.0 / array)),
    }


def deletion_metrics(
    rows: Sequence[Mapping[str, object]], *, error_threshold: float = 0.5
) -> dict[str, Any]:
    """Deletion precision/recall/F1 on stable canonical tokens."""

    if not 0.0 <= error_threshold <= 1.0:
        raise MetricError("error_threshold must be in [0, 1]")
    stable = [row for row in rows if bool(row["stable"])]
    if not stable:
        return {"n": 0, "support": 0, "predicted": 0, "precision": None,
                "recall": None, "f1": None}
    gold = np.asarray([row["gold_event"] == "deletion" for row in stable], dtype=bool)
    pred = np.asarray([
        float(row["p_error"]) >= error_threshold and row["top_alt"] == DELETE
        for row in stable
    ], dtype=bool)
    tp = int(np.sum(gold & pred))
    fp = int(np.sum(~gold & pred))
    fn = int(np.sum(gold & ~pred))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "n": len(stable),
        "support": int(gold.sum()),
        "predicted": int(pred.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def specific_output_metrics(
    rows: Sequence[Mapping[str, object]], *, error_threshold: float = 0.5
) -> dict[str, Any]:
    """Evaluate fold-gated concrete substitution/deletion statements.

    The optional ``specific_output_authorized`` field is evaluation-only.  It
    makes the no-abstention ablation observable without changing raw GoP or
    calibrated error probabilities.
    """

    present = ["specific_output_authorized" in row for row in rows]
    if not any(present):
        return {
            "available": False,
            "n_error_predictions": 0,
            "n_authorized": 0,
            "coverage": None,
            "precision": None,
        }
    if not all(present):
        raise MetricError("specific-output authorization is missing from part of a method")
    if any(type(row["specific_output_authorized"]) is not bool for row in rows):
        raise MetricError("specific_output_authorized must be boolean")
    eligible = [row for row in rows if float(row["p_error"]) >= error_threshold]
    authorized = [row for row in eligible if bool(row["specific_output_authorized"])]
    correct = 0
    for row in authorized:
        if row["top_alt"] == DELETE:
            correct += int(row["gold_event"] == "deletion")
        else:
            correct += int(
                row["gold_event"] == "substitution"
                and row["gold_realized"] == row["top_alt"]
            )
    return {
        "available": True,
        "n_error_predictions": len(eligible),
        "n_authorized": len(authorized),
        "coverage": len(authorized) / len(eligible) if eligible else 0.0,
        "precision": correct / len(authorized) if authorized else None,
    }


def risk_coverage_curve(
    y_true: Sequence[int], p_error: Sequence[float], *, threshold: float = 0.5
) -> dict[str, Any]:
    """Selective binary-classification risk ordered by posterior confidence.

    ``aurc`` is the mean prefix risk at all attainable non-zero coverages.  A
    lower value is better.  Ties are resolved by original order, and callers
    should therefore provide their frozen token order.
    """

    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(p_error, dtype=np.float64)
    if len(y) == 0 or y.shape != p.shape or y.ndim != 1:
        raise MetricError("risk-coverage inputs must be non-empty equal vectors")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all():
        raise MetricError("risk-coverage inputs are invalid")
    if not 0.0 <= threshold <= 1.0:
        raise MetricError("threshold must be in [0, 1]")
    confidence = np.abs(p - threshold)
    order = np.argsort(-confidence, kind="stable")
    mistakes = ((p >= threshold).astype(np.int8) != y).astype(np.float64)[order]
    count = np.arange(1, len(y) + 1, dtype=np.float64)
    risks = np.cumsum(mistakes) / count
    coverage = count / len(y)
    return {
        "n": int(len(y)),
        "aurc": float(np.mean(risks)),
        "coverage": coverage.tolist(),
        "risk": risks.tolist(),
    }


def precision_at_coverage(
    correctness: Sequence[bool], confidence: Sequence[float], coverage: float
) -> dict[str, Any]:
    """Return precision at a predeclared minimum selective coverage."""

    correct = np.asarray(correctness, dtype=bool)
    conf = np.asarray(confidence, dtype=np.float64)
    if len(correct) == 0 or correct.shape != conf.shape or correct.ndim != 1:
        raise MetricError("precision-at-coverage inputs must be equal non-empty vectors")
    if not np.isfinite(conf).all() or not 0.0 < coverage <= 1.0:
        raise MetricError("invalid confidence or coverage")
    n_selected = max(1, int(math.ceil(coverage * len(correct))))
    chosen = np.argsort(-conf, kind="stable")[:n_selected]
    return {
        "requested_coverage": float(coverage),
        "achieved_coverage": n_selected / len(correct),
        "n_selected": n_selected,
        "precision": float(np.mean(correct[chosen])),
    }


def classification_precision_at_coverages(
    y_true: Sequence[int],
    p_error: Sequence[float],
    *,
    threshold: float = 0.5,
    coverages: Sequence[float] = PRECISION_COVERAGES,
) -> dict[str, dict[str, Any]]:
    """Report selective classification precision at frozen coverage points.

    Here ``precision`` is the fraction of correct binary error decisions among
    the most confident retained tokens.  This accompanies, rather than
    replaces, the complete risk--coverage curve.
    """

    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(p_error, dtype=np.float64)
    if len(y) == 0 or y.shape != p.shape or y.ndim != 1:
        raise MetricError("precision-at-coverage inputs must be equal non-empty vectors")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all():
        raise MetricError("precision-at-coverage inputs are invalid")
    if not 0.0 <= threshold <= 1.0:
        raise MetricError("threshold must be in [0, 1]")
    requested = tuple(float(value) for value in coverages)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(not 0.0 < value <= 1.0 for value in requested)
    ):
        raise MetricError("coverage points must be unique values in (0, 1]")
    predicted = (p >= threshold).astype(np.int8)
    correctness = predicted == y
    confidence = np.abs(p - threshold)
    return {
        f"{value:.2f}": precision_at_coverage(correctness, confidence, value)
        for value in requested
    }


def _macro(per_speaker: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [float(item[key]) for item in per_speaker if item.get(key) is not None]
    return {"value": float(np.mean(values)) if values else None,
            "n_valid_speakers": len(values)}


def _supplemental_strata(
    rows: Sequence[Mapping[str, object]], key: str
) -> list[dict[str, Any]]:
    """Compute explicitly pooled, descriptive strata for one evaluation field."""

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        label = format(float(value), ".12g") if isinstance(value, float) else str(value)
        grouped[label].append(row)
    output: list[dict[str, Any]] = []
    for label in sorted(grouped):
        local = grouped[label]
        y = [int(row["gold_event"] != "correct") for row in local]
        p = [float(row["p_error"]) for row in local]
        output.append(
            {
                "stratum": label,
                "n_tokens": len(local),
                "n_speakers": len({str(row["speaker"]) for row in local}),
                "binary": binary_metrics(y, p),
                "substitution": substitution_rank_metrics(local),
                "deletion": deletion_metrics(local),
            }
        )
    return output


def evaluate_prediction_group(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Evaluate one method/cohort using stable-token, speaker-macro metrics."""

    if not rows:
        raise MetricError("cannot evaluate an empty group")
    methods = {str(row["method"]) for row in rows}
    cohorts = {str(row["cohort"]) for row in rows}
    if len(methods) != 1 or len(cohorts) != 1:
        raise MetricError("evaluation group must contain one method and one cohort")
    by_speaker: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if bool(row["stable"]):
            by_speaker[str(row["speaker"])].append(row)
    if not by_speaker:
        raise MetricError("evaluation group has no stable tokens")

    speaker_rows: list[dict[str, Any]] = []
    for speaker in sorted(by_speaker):
        tokens = by_speaker[speaker]
        y = [int(row["gold_event"] != "correct") for row in tokens]
        p = [float(row["p_error"]) for row in tokens]
        binary = binary_metrics(y, p)
        rank = substitution_rank_metrics(tokens)
        deletion = deletion_metrics(tokens)
        specific = specific_output_metrics(tokens)
        risk = risk_coverage_curve(y, p)
        precision_by_coverage = classification_precision_at_coverages(y, p)
        speaker_rows.append({
            "speaker": speaker,
            **binary,
            "substitution_top1": rank["top1"],
            "substitution_top3": rank["top3"],
            "substitution_mrr": rank["mrr"],
            "deletion_precision": deletion["precision"],
            "deletion_recall": deletion["recall"],
            "deletion_f1": deletion["f1"],
            "aurc": risk["aurc"],
            "risk_coverage_auc": risk["aurc"],
            "specific_output_coverage": specific["coverage"],
            "specific_output_precision": specific["precision"],
            **{
                f"precision_at_coverage_{key}": value["precision"]
                for key, value in precision_by_coverage.items()
            },
        })

    macro_keys = (
        "auprc", "auroc", "brier", "nll", "substitution_top1",
        "substitution_top3", "substitution_mrr", "deletion_precision",
        "deletion_recall", "deletion_f1", "aurc", "risk_coverage_auc",
        "specific_output_coverage", "specific_output_precision",
        *(f"precision_at_coverage_{value:.2f}" for value in PRECISION_COVERAGES),
    )
    pooled_rows = [row for rows_ in by_speaker.values() for row in rows_]
    pooled_y = [int(row["gold_event"] != "correct") for row in pooled_rows]
    pooled_p = [float(row["p_error"]) for row in pooled_rows]
    return {
        "method": next(iter(methods)),
        "cohort": next(iter(cohorts)),
        "n_speakers": len(speaker_rows),
        "n_stable_tokens": len(pooled_rows),
        "per_speaker": speaker_rows,
        "macro": {key: _macro(speaker_rows, key) for key in macro_keys},
        "pooled_binary": binary_metrics(pooled_y, pooled_p),
        "pooled_substitution": substitution_rank_metrics(pooled_rows),
        "pooled_deletion": deletion_metrics(pooled_rows),
        "pooled_specific_output": specific_output_metrics(pooled_rows),
        "pooled_risk_coverage": risk_coverage_curve(pooled_y, pooled_p),
        "pooled_precision_at_coverage": classification_precision_at_coverages(
            pooled_y, pooled_p
        ),
        "supplemental_strata": {
            "canonical_phone": _supplemental_strata(pooled_rows, "target"),
            "audio_microphone": _supplemental_strata(
                pooled_rows, "audio_microphone"
            ),
            "phn_label_microphone": _supplemental_strata(
                pooled_rows, "phn_microphone"
            ),
            "severity_rank": _supplemental_strata(pooled_rows, "severity_rank"),
            "severity_label": _supplemental_strata(pooled_rows, "severity_label"),
        },
    }


def evaluate_predictions(rows: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Build the machine-readable ``da-cf-gop.metrics.v1`` report."""

    validate_prediction_rows(rows)
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["cohort"]), str(row["method"]))].append(row)
    results = [evaluate_prediction_group(groups[key]) for key in sorted(groups)]
    return {
        "schema_version": "da-cf-gop.metrics.v1",
        "n_prediction_rows": len(rows),
        "results": results,
    }


def paired_speaker_bootstrap(
    values_a: Mapping[str, float],
    values_b: Mapping[str, float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20260829,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired speaker bootstrap for macro metric A minus B."""

    if set(values_a) != set(values_b) or not values_a:
        raise MetricError("paired methods must have the same non-empty speaker set")
    if n_bootstrap < 1 or not 0.0 < confidence < 1.0:
        raise MetricError("invalid bootstrap configuration")
    speakers = sorted(values_a)
    a = np.asarray([_finite_float(values_a[s], f"A[{s}]") for s in speakers])
    b = np.asarray([_finite_float(values_b[s], f"B[{s}]") for s in speakers])
    difference = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(speakers), size=(n_bootstrap, len(speakers)))
    replicates = np.mean(difference[indices], axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return {
        "difference_a_minus_b": float(np.mean(difference)),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "n_speakers": len(speakers),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "unit": "speaker",
    }


def exact_sign_flip_test(
    differences: Mapping[str, float] | Sequence[float],
) -> dict[str, Any]:
    """Exact two-sided paired sign-flip test of the mean difference."""

    if isinstance(differences, Mapping):
        labels = sorted(differences)
        values = np.asarray([
            _finite_float(differences[label], f"difference[{label}]") for label in labels
        ])
    else:
        values = np.asarray([_finite_float(value, "difference") for value in differences])
    if len(values) == 0:
        raise MetricError("sign-flip test needs at least one speaker")
    if len(values) > 20:
        raise MetricError("exact sign-flip enumeration is limited to 20 speakers")
    observed = float(np.mean(values))
    null = np.asarray([
        np.mean(values * np.asarray(signs, dtype=np.float64))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return {
        "statistic": "mean_paired_difference",
        "difference": observed,
        "p_two_sided": p_value,
        "n_speakers": int(len(values)),
        "n_permutations": int(len(null)),
    }


def severity_correlations(
    severity: Sequence[float], scores: Sequence[float]
) -> dict[str, float | int | None]:
    """Speaker-level severity correlations for the secondary experiment."""

    y = np.asarray(severity, dtype=np.float64)
    score = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or score.shape != y.shape or len(y) < 3:
        raise MetricError("severity correlations need at least three paired speakers")
    if not np.isfinite(y).all() or not np.isfinite(score).all():
        raise MetricError("severity values and scores must be finite")
    if len(np.unique(y)) < 2 or len(np.unique(score)) < 2:
        return {"n": len(y), "kendall_tau_b": None, "abs_kendall_tau_b": None,
                "spearman_rho": None, "pearson_r": None}
    tau = float(stats.kendalltau(y, score, variant="b").statistic)
    return {
        "n": int(len(y)),
        "kendall_tau_b": tau,
        "abs_kendall_tau_b": abs(tau),
        "spearman_rho": float(stats.spearmanr(y, score).statistic),
        "pearson_r": float(stats.pearsonr(y, score).statistic),
    }
