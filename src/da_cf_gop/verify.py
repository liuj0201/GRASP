"""Independent verification of DA-CF-GoP artifacts.

The CLI should call :func:`verify_and_print`; on success it emits exactly one
line, ``PASS``.  Every helper raises :class:`VerificationError` on the first
auditable contract violation and never silently repairs an artifact.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import load_config
from .llm_schema import build_llm_record, read_llm_records
from .metrics import (
    MetricError,
    evaluate_predictions,
    exact_sign_flip_test,
    paired_speaker_bootstrap,
    validate_prediction_rows,
)
from .provenance import (
    ProvenanceError,
    companion_descriptor_path,
    load_npz,
    read_json,
    sha256_file,
    sha256_json,
    validate_cache_descriptor,
    verify_path_record,
)


FROZEN_METHODS = (
    "conventional_forced_alignment_gop",
    "ctc_sf_sd_norm",
    "ctc_sf_sd_norm_sequence_recalibrated",
    "all_phone_cf_uniform_raw",
    "all_phone_cf_calibrated",
    "code9_legacy_hard_graph",
    "da_cf_fixed_no_phn",
    "da_cf_adapted",
    "da_cf_adapted_fixed_prior",
    "da_cf_adapted_uniform_prior",
    "da_cf_adapted_hard_pruning",
    "da_cf_adapted_no_topology_correction",
    "da_cf_adapted_no_sequence_recalibration",
    "da_cf_adapted_no_deletion_calibration",
    "da_cf_adapted_no_deletion",
    "da_cf_adapted_no_abstention",
)
ADAPTED_METHOD = "da_cf_adapted"
PRIMARY_COMPARATOR = "all_phone_cf_calibrated"
MANIFEST_ROW_KEYS = frozenset(
    {
        "agreed_insertions", "audio_microphone", "canonical_phones",
        "category_insertions", "duration_s", "levenshtein_insertions",
        "phn_microphone", "phn_model_phones", "phn_path", "phn_raw_labels",
        "phn_timit_phones", "prompt", "prompt_id", "prompt_path",
        "reading_event_id", "sample_rate", "schema_version", "session",
        "severity_label", "severity_rank", "sex", "speaker_group",
        "speaker_id", "stable_count", "stable_fraction", "stable_tokens",
        "stem", "uncertain_count", "wav_path",
    }
)
MANIFEST_SOURCE_HASH_KEYS = frozenset(
    {"wav_sha256", "prompt_sha256", "phn_sha256", "g2p_descriptor_sha256"}
)
STABLE_TOKEN_KEYS = frozenset(
    {
        "canonical_phone", "category_event_type", "category_observed_index",
        "category_realized_phone", "event_type", "levenshtein_event_type",
        "levenshtein_observed_index", "levenshtein_realized_phone",
        "phone_index", "realized_phone", "stable",
    }
)
EXCLUSION_ROW_KEYS = frozenset(
    {"detail", "reading_event_id", "reason", "session", "speaker_id", "stem"}
)
MANIFEST_AUDIT_KEYS = frozenset(
    {
        "agreed_insertions", "audio_policy", "canonical_tokens",
        "category_insertions", "cohort", "data_root",
        "dysarthric_speakers_requested", "exact_substitution_deletion_headline_allowed",
        "exclusion_reasons", "headline_stability_population",
        "healthy_speakers_requested", "levenshtein_insertions", "manifest_sha256",
        "minimum_stable_fraction", "n_exclusions", "n_patient_rows", "n_rows",
        "n_unique_canonical_sequences", "n_unique_prompts", "patient_canonical_tokens",
        "patient_stable_fraction", "patient_stable_tokens", "patient_uncertain_tokens",
        "phn_boundaries_serialized", "phn_microphone_counts", "phn_policy",
        "reading_event_key", "rows_by_speaker", "schema_version", "stable_fraction",
        "stable_tokens", "uncertain_tokens",
    }
)
MANIFEST_AUDIT_SOURCE_KEYS = frozenset({"input_source_hashes", "g2p_provenance"})
STAGE_SUMMARY_KEYS = frozenset(
    {
        "schema_version", "stage", "config_sha256", "training_speakers",
        "source_files", "models", "manifests", "upstream_artifacts",
        "exclusions", "outputs", "details",
    }
)


class VerificationError(AssertionError):
    """An artifact set is incomplete, inconsistent, unsafe, or stale."""


def _logical_json_sha256(value: object) -> str:
    """Match the compact, no-trailing-newline hash used in manifest audits."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_deterministic_gzip(path: str | os.PathLike[str]) -> None:
    item = Path(path)
    header = item.read_bytes()[:10]
    _require(len(header) == 10 and header[:2] == b"\x1f\x8b", f"not gzip: {item}")
    _require(int.from_bytes(header[4:8], "little") == 0,
             f"gzip mtime is not deterministic: {item}")


def verify_deterministic_npz(path: str | os.PathLike[str]) -> None:
    item = Path(path)
    try:
        with zipfile.ZipFile(item, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            _require(names == sorted(names) and len(names) == len(set(names)),
                     f"NPZ members are not uniquely sorted: {item}")
            _require(all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in infos),
                     f"NPZ ZIP timestamps are not deterministic: {item}")
    except zipfile.BadZipFile as error:
        raise VerificationError(f"invalid NPZ ZIP: {item}") from error
    try:
        load_npz(item)
    except ProvenanceError as error:
        raise VerificationError(str(error)) from error


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, object]] = []
    try:
        with opener(path, "rt", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                _require(bool(line.strip()), f"blank JSONL line {line_number}: {path}")
                value = json.loads(line)
                _require(isinstance(value, dict), f"non-object JSONL line {line_number}: {path}")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read JSONL artifact: {path}") from error
    _require(bool(rows), f"empty JSONL artifact: {path}")
    return rows


def verify_evaluation_artifact(path: str | os.PathLike[str]) -> list[dict[str, object]]:
    item = Path(path)
    _require(item.is_file(), f"missing prediction artifact: {item}")
    if item.suffix == ".gz":
        verify_deterministic_gzip(item)
    rows = _read_jsonl(item)
    try:
        validate_prediction_rows(rows)
    except MetricError as error:
        raise VerificationError(f"invalid prediction artifact {item}: {error}") from error
    return rows


def verify_matched_method_coverage(rows: Sequence[Mapping[str, object]]) -> None:
    """Require every controlled method to score exactly the same token keys."""

    by_cohort_method: dict[tuple[str, str], dict[tuple[object, ...], tuple[object, ...]]] = {}
    for row in rows:
        group = (str(row["cohort"]), str(row["method"]))
        key = (row["speaker"], row["event"], row["phone_index"], row["target"])
        gold = (row["gold_event"], row["gold_realized"], row["stable"])
        mapping = by_cohort_method.setdefault(group, {})
        _require(key not in mapping, f"duplicate token within method: {group} {key}")
        mapping[key] = gold
    cohorts = sorted({cohort for cohort, _ in by_cohort_method})
    for cohort in cohorts:
        methods = sorted(method for candidate, method in by_cohort_method if candidate == cohort)
        _require(bool(methods), f"cohort has no methods: {cohort}")
        reference = by_cohort_method[(cohort, methods[0])]
        for method in methods[1:]:
            observed = by_cohort_method[(cohort, method)]
            _require(set(observed) == set(reference),
                     f"method token coverage differs in {cohort}: {method}")
            _require(all(observed[key] == reference[key] for key in reference),
                     f"gold/stability fields differ between methods in {cohort}: {method}")


def _same_number(observed: object, expected: object, label: str) -> None:
    if expected is None:
        _require(observed is None, f"{label} should be null")
        return
    _require(observed is not None, f"{label} is unexpectedly null")
    try:
        left, right = float(observed), float(expected)
    except (TypeError, ValueError) as error:
        raise VerificationError(f"{label} is not numeric") from error
    _require(math.isfinite(left) and math.isclose(left, right, abs_tol=1e-12, rel_tol=1e-10),
             f"{label} cannot be recomputed")


def _verify_recomputed(observed: object, expected: object, label: str) -> None:
    """Recursively compare a serialized claim with an independent recomputation."""

    if isinstance(expected, Mapping):
        _require(isinstance(observed, Mapping), f"{label} is not an object")
        _require(set(observed) == set(expected), f"{label} key inventory mismatch")
        for key in expected:
            _verify_recomputed(observed[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        _require(isinstance(observed, list), f"{label} is not a list")
        _require(len(observed) == len(expected), f"{label} length mismatch")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _verify_recomputed(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        _require(observed == expected and type(observed) is type(expected),
                 f"{label} cannot be recomputed")
        return
    if isinstance(expected, (int, float)):
        if isinstance(expected, int):
            _require(type(observed) is int and observed == expected,
                     f"{label} cannot be recomputed")
        else:
            _same_number(observed, expected, label)
        return
    _require(observed == expected, f"{label} cannot be recomputed")


def _metric_group(report: Mapping[str, object], method: str) -> Mapping[str, object]:
    results = report.get("results")
    _require(isinstance(results, list), "metrics results must be a list")
    matches = [row for row in results if isinstance(row, Mapping) and row.get("method") == method]
    _require(len(matches) == 1, f"metrics lacks exactly one group for {method}")
    return matches[0]


def _difference(left: object, right: object) -> float | None:
    return None if left is None or right is None else float(left) - float(right)


def _recompute_paired_and_success(
    core_report: Mapping[str, object],
    config: Mapping[str, object],
    cohort: str,
    *,
    headline_allowed: bool,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    evaluation = config.get("evaluation")
    _require(isinstance(evaluation, Mapping), "frozen evaluation configuration is missing")
    comparator_name = str(evaluation.get("primary_comparator", PRIMARY_COMPARATOR))
    adapted = _metric_group(core_report, ADAPTED_METHOD)
    comparator = _metric_group(core_report, comparator_name)

    def speaker_auprc(group: Mapping[str, object]) -> dict[str, float]:
        rows = group.get("per_speaker")
        _require(isinstance(rows, list), "per-speaker metrics are missing")
        return {
            str(row["speaker"]): float(row["auprc"])
            for row in rows if isinstance(row, Mapping) and row.get("auprc") is not None
        }

    adapted_speakers = speaker_auprc(adapted)
    comparator_speakers = speaker_auprc(comparator)
    _require(set(adapted_speakers) == set(comparator_speakers) and bool(adapted_speakers),
             "headline paired speaker coverage differs")
    differences = {
        speaker: adapted_speakers[speaker] - comparator_speakers[speaker]
        for speaker in sorted(adapted_speakers)
    }
    bootstrap_replicates = int(evaluation.get("bootstrap_replicates", 10_000))
    seed = int(config.get("seed", 20260829))
    paired: dict[str, object] = {
        "schema_version": "da-cf-gop.paired-comparison.v1",
        "adapted_method": ADAPTED_METHOD,
        "comparator": comparator_name,
        "metric": "speaker_auprc",
        "per_speaker_difference": differences,
        "n_improved": sum(value > 0 for value in differences.values()),
        "bootstrap": paired_speaker_bootstrap(
            adapted_speakers, comparator_speakers,
            n_bootstrap=bootstrap_replicates, seed=seed,
        ),
        "exact_sign_flip": exact_sign_flip_test(differences),
    }

    all_paired: list[dict[str, object]] = []
    results = core_report["results"]
    assert isinstance(results, list)
    for result in sorted(results, key=lambda row: str(row["method"])):
        assert isinstance(result, Mapping)
        method = str(result["method"])
        if method == ADAPTED_METHOD:
            continue
        other = speaker_auprc(result)
        common = sorted(set(adapted_speakers) & set(other))
        _require(bool(common), f"no paired AUPRC speakers for {method}")
        left = {speaker: adapted_speakers[speaker] for speaker in common}
        right = {speaker: other[speaker] for speaker in common}
        delta = {speaker: left[speaker] - right[speaker] for speaker in common}
        all_paired.append({
            "adapted_method": ADAPTED_METHOD,
            "comparator": method,
            "metric": "speaker_auprc",
            "per_speaker_difference": delta,
            "n_improved": sum(value > 0 for value in delta.values()),
            "bootstrap": paired_speaker_bootstrap(
                left, right, n_bootstrap=bootstrap_replicates, seed=seed
            ),
            "exact_sign_flip": exact_sign_flip_test(delta),
        })

    adapted_macro = adapted["macro"]
    comparator_macro = comparator["macro"]
    assert isinstance(adapted_macro, Mapping) and isinstance(comparator_macro, Mapping)
    auprc_difference = _difference(
        adapted_macro["auprc"]["value"], comparator_macro["auprc"]["value"]
    )
    auroc_difference = _difference(
        adapted_macro["auroc"]["value"], comparator_macro["auroc"]["value"]
    )
    brier_difference = _difference(
        adapted_macro["brier"]["value"], comparator_macro["brier"]["value"]
    )
    top1_difference = _difference(
        adapted_macro["substitution_top1"]["value"],
        comparator_macro["substitution_top1"]["value"],
    )
    deletion_difference = _difference(
        adapted_macro["deletion_f1"]["value"], comparator_macro["deletion_f1"]["value"]
    )
    adapted_auprc = adapted_macro["auprc"]["value"]
    all_auprc = [
        row["macro"]["auprc"]["value"]
        for row in results
        if row["macro"]["auprc"]["value"] is not None
    ]
    checks = {
        "adapted_is_highest_controlled_macro_auprc": bool(
            adapted_auprc is not None
            and all_auprc
            and float(adapted_auprc) >= max(float(value) for value in all_auprc) - 1e-12
        ),
        "auprc_absolute_improvement": bool(
            auprc_difference is not None
            and auprc_difference >= (
                float(evaluation.get("required_absolute_improvement", 0.03))
                if cohort == "primary5" else 0.0
            )
        ),
        "required_speakers_improved": paired["n_improved"] >= int(
            evaluation.get(
                "required_primary_speakers_improved" if cohort == "primary5"
                else "required_sensitivity_speakers_improved",
                4 if cohort == "primary5" else 5,
            )
        ),
        "auroc_non_degradation": bool(
            auroc_difference is not None
            and auroc_difference >= -float(evaluation.get("maximum_auroc_drop", 0.01))
        ),
        "brier_non_degradation": bool(
            brier_difference is not None
            and brier_difference <= float(evaluation.get("maximum_brier_increase", 0.01))
        ),
        "substitution_top1_non_degradation": bool(
            top1_difference is not None
            and top1_difference >= -float(evaluation.get("maximum_identification_drop", 0.01))
        ),
        "deletion_f1_non_degradation": bool(
            deletion_difference is not None
            and deletion_difference >= -float(evaluation.get("maximum_identification_drop", 0.01))
        ),
    }
    required = checks if cohort == "primary5" else {
        "auprc_direction_positive": auprc_difference is not None and auprc_difference > 0,
        "required_speakers_improved": checks["required_speakers_improved"],
    }
    success: dict[str, object] = {
        "schema_version": "da-cf-gop.success-assessment.v1",
        "cohort": cohort,
        "adapted_macro_auprc": adapted_auprc,
        "comparator_macro_auprc": comparator_macro["auprc"]["value"],
        "auprc_difference": auprc_difference,
        "auroc_difference": auroc_difference,
        "brier_difference": brier_difference,
        "substitution_top1_difference": top1_difference,
        "deletion_f1_difference": deletion_difference,
        "checks": checks,
        "cohort_success": all(bool(value) for value in required.values()),
        "negative_result_frozen_if_false": True,
        "test_set_driven_retuning_permitted": False,
    }
    if headline_allowed:
        success["assessment_status"] = "evaluated"
    else:
        success["cohort_success"] = None
        success["assessment_status"] = "not_evaluable_alignment_stability_below_threshold"
        success["negative_result_frozen_if_false"] = False
    return paired, all_paired, success


def verify_metrics_report(
    report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object] | None = None,
    expected_cohort: str | None = None,
    manifest_audit: Mapping[str, object] | None = None,
) -> None:
    """Recompute the full metric tree and every registered success claim."""

    _require(report.get("schema_version") == "da-cf-gop.metrics.v1",
             "metrics schema version changed")
    _require(isinstance(report.get("results"), list), "metrics results must be a list")
    cohorts_in_report = {
        str(item.get("cohort")) for item in report["results"] if isinstance(item, Mapping)
    }
    if expected_cohort is not None:
        _require(cohorts_in_report == {expected_cohort},
                 f"metrics report is not exclusively {expected_cohort}")
    relevant = [row for row in rows if str(row["cohort"]) in cohorts_in_report]
    expected = evaluate_predictions(relevant)
    _verify_recomputed(report.get("n_prediction_rows"), expected["n_prediction_rows"],
                       "metrics.n_prediction_rows")
    _verify_recomputed(report["results"], expected["results"], "metrics.results")

    if config is None:
        return
    _require(expected_cohort is not None and len(cohorts_in_report) == 1,
             "protocol claim verification needs one explicit cohort")
    _require(report.get("config_sha256") == config.get("_config_hash"),
             "metrics report is not bound to the frozen configuration")
    _require(isinstance(manifest_audit, Mapping), "manifest audit is required for headline claims")
    headline_allowed = manifest_audit.get("exact_substitution_deletion_headline_allowed")
    _require(type(headline_allowed) is bool, "manifest audit lacks the headline stability decision")
    expected_headline = {
        "exact_substitution_deletion_headline_allowed": headline_allowed,
        "error_detection_status": "headline" if headline_allowed else "exploratory_only",
        "substitution_deletion_status": "headline" if headline_allowed else "exploratory_only",
        "minimum_stable_fraction": manifest_audit.get("minimum_stable_fraction"),
        "observed_patient_stable_fraction": manifest_audit.get("patient_stable_fraction"),
    }
    _verify_recomputed(report.get("headline_policy"), expected_headline,
                       "metrics.headline_policy")
    paired, all_paired, success = _recompute_paired_and_success(
        expected, config, expected_cohort, headline_allowed=bool(headline_allowed)
    )
    _verify_recomputed(report.get("paired_comparison"), paired,
                       "metrics.paired_comparison")
    _verify_recomputed(report.get("paired_comparisons"), all_paired,
                       "metrics.paired_comparisons")
    _verify_recomputed(report.get("success_assessment"), success,
                       "metrics.success_assessment")


def _expected_metrics_csv_rows(
    report: Mapping[str, object], level: str
) -> list[dict[str, object]]:
    results = report.get("results")
    _require(isinstance(results, list), "metrics results are unavailable for CSV verification")
    output: list[dict[str, object]] = []
    if level == "method":
        for result in results:
            _require(isinstance(result, Mapping), "metrics result is not an object")
            row: dict[str, object] = {
                "cohort": result["cohort"],
                "method": result["method"],
                "n_speakers": result["n_speakers"],
                "n_stable_tokens": result["n_stable_tokens"],
            }
            macro = result.get("macro")
            _require(isinstance(macro, Mapping), "metrics macro block is missing")
            for name, entry in macro.items():
                _require(isinstance(entry, Mapping), f"metrics macro {name} is invalid")
                row[f"macro_{name}"] = entry["value"]
                row[f"macro_{name}_n_speakers"] = entry["n_valid_speakers"]
            output.append(row)
        return output
    _require(level == "speaker", f"unknown metrics CSV level: {level}")
    for result in results:
        _require(isinstance(result, Mapping), "metrics result is not an object")
        speakers = result.get("per_speaker")
        _require(isinstance(speakers, list), "per-speaker metrics are missing")
        for speaker in speakers:
            _require(isinstance(speaker, Mapping), "per-speaker metric row is invalid")
            row = {"cohort": result["cohort"], "method": result["method"]}
            row.update(speaker)
            output.append(row)
    return output


def _csv_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def verify_metrics_csv(
    path: str | os.PathLike[str],
    *,
    report: Mapping[str, object] | None = None,
    level: str | None = None,
) -> None:
    item = Path(path)
    _require(item.is_file(), f"missing method metrics CSV: {item}")
    try:
        with item.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(reader.fieldnames is not None, f"CSV has no header: {item}")
            _require({"cohort", "method"}.issubset(reader.fieldnames),
                     f"CSV lacks cohort/method: {item}")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise VerificationError(f"invalid metrics CSV: {item}") from error
    _require(bool(rows), f"empty metrics CSV: {item}")
    if report is None:
        return
    _require(level in {"method", "speaker"}, "metrics CSV level is required")
    expected_rows = _expected_metrics_csv_rows(report, str(level))
    expected_fields = sorted({key for row in expected_rows for key in row})
    _require(reader.fieldnames == expected_fields,
             f"metrics CSV column inventory/order changed: {item}")
    serialized = [
        {key: _csv_string(row.get(key)) for key in expected_fields}
        for row in expected_rows
    ]
    _verify_recomputed(rows, serialized, f"metrics CSV {item.name}")


def _verify_severity_inference_inventory(
    observed: Sequence[Mapping[str, object]],
    regenerated: Sequence[Mapping[str, object]],
) -> None:
    """Compare regenerated severity inputs in their persisted canonical order."""

    expected = sorted(
        (dict(row) for row in regenerated),
        key=lambda row: (str(row["event_id"]), str(row["recording_id"])),
    )
    _verify_recomputed(list(observed), expected, "severity audited inference manifest")


def verify_severity_artifacts(
    evaluation_dir: Path,
    *,
    project_root: Path | None = None,
    config: Mapping[str, object] | None = None,
) -> dict[str, int]:
    """Recompute the complete matched eight-speaker secondary experiment."""

    from .severity import (
        EXPECTED_DYSARTHRIC_SPEAKERS,
        FROZEN_COMPARATOR_METHODS,
        NEW_METHOD,
        build_audited_severity_events,
        canonical_phone_sequences_from_audited_prompts,
        load_frozen_score_sources,
        run_severity_experiment,
        validate_severity_phone_scores,
    )

    score_path = evaluation_dir / "severity_phone_scores.jsonl.gz"
    provenance_path = evaluation_dir / "severity_phone_scores.provenance.json"
    metrics_path = evaluation_dir / "severity_metrics.json"
    csv_path = evaluation_dir / "severity_metrics.csv"
    for path in (score_path, provenance_path, metrics_path, csv_path):
        _require(path.is_file(), f"severity artifact is missing: {path}")
    verify_deterministic_gzip(score_path)
    score_rows = _read_jsonl(score_path)
    audited_recordings = None
    canonical_by_recording = None
    if project_root is not None:
        inference_path = project_root / "artifacts" / "manifests" / "severity_audited_recordings.jsonl"
        _require(inference_path.is_file(), "severity inference manifest is missing")
        audited_recordings = _read_jsonl(inference_path)
        try:
            canonical_by_recording = canonical_phone_sequences_from_audited_prompts(
                audited_recordings
            )
        except ValueError as error:
            raise VerificationError(str(error)) from error
    try:
        score_stats = validate_severity_phone_scores(
            score_rows,
            audited_recordings=audited_recordings,
            canonical_phones_by_recording=canonical_by_recording,
            expected_recordings=415,
            expected_events=329,
            expected_speakers=EXPECTED_DYSARTHRIC_SPEAKERS,
        )
    except ValueError as error:
        raise VerificationError(str(error)) from error

    provenance = read_json(provenance_path)
    _require(isinstance(provenance, Mapping), "severity provenance is not an object")
    _require(
        provenance.get("schema_version")
        == "da-cf-gop.severity-scoring-provenance.v1",
        "severity provenance schema changed",
    )
    _require(provenance.get("output_sha256") == sha256_file(score_path),
             "severity score hash disagrees with provenance")
    _require(provenance.get("n_recordings") == 415, "severity recording count changed")
    _require(provenance.get("n_events") == 329, "severity event count changed")
    _require(provenance.get("n_phone_rows") == len(score_rows),
             "severity phone-row count changed")
    _require(score_stats["n_recordings"] == 415,
             "severity scores do not cover 415 recordings")
    _require(provenance.get("legacy_cached_canonical_ids_used") is False,
             "severity scorer used historical canonical ids")
    _require(provenance.get("phn_or_textgrid_used_at_inference") is False,
             "severity inference used PHN/TextGrid")
    _require(provenance.get("severity_used_for_training") is False,
             "severity label entered model training")
    fold_hashes = provenance.get("fold_fit_hashes")
    _require(isinstance(fold_hashes, Mapping)
             and set(fold_hashes) == set(EXPECTED_DYSARTHRIC_SPEAKERS),
             "severity fold-fit hash inventory changed")
    _require(all(isinstance(value, str) and len(value) == 64 for value in fold_hashes.values()),
             "severity fold-fit hashes are invalid")
    if config is not None:
        _require(provenance.get("config_sha256") == config.get("_config_hash"),
                 "severity scores are not bound to the frozen config")

    report = read_json(metrics_path)
    _require(isinstance(report, Mapping), "severity metrics is not an object")
    _require(report.get("schema_version") == "da-cf-gop.severity.v1",
             "severity metrics schema changed")
    _require(report.get("comparison_level") == "system_level_cross_backend",
             "severity comparison level changed")
    _require(report.get("component_causal_claims_permitted") is False,
             "severity report permits invalid component claims")
    _require(report.get("severity_used_for_model_training") is False,
             "severity report says severity was used for training")
    scope = report.get("audit_scope")
    _require(isinstance(scope, Mapping), "severity audit scope is missing")
    _require(scope.get("n_manifest_rows") == 415, "severity audit is not 415 recordings")
    _require(scope.get("n_unique_reading_events") == 329,
             "severity audit is not 329 events")
    _require(scope.get("n_speakers") == 8, "severity audit is not eight speakers")
    _require(scope.get("strict_matched_coverage") is True,
             "severity methods are not strictly matched")
    methods = report.get("methods")
    _require(isinstance(methods, Mapping) and len(methods) == 5,
             "severity report must contain DA-CF and four comparators")
    _require(set(methods) == {NEW_METHOD, *FROZEN_COMPARATOR_METHODS},
             "severity report method inventory changed")
    for method, result in methods.items():
        _require(isinstance(result, Mapping), f"invalid severity method: {method}")
        coverage = result.get("coverage")
        _require(isinstance(coverage, Mapping), f"severity coverage missing: {method}")
        _require(coverage.get("n_events") == 329, f"severity event coverage changed: {method}")
        _require(coverage.get("n_speakers") == 8, f"severity speaker coverage changed: {method}")
        _require(float(coverage.get("event_coverage", 0.0)) == 1.0,
                 f"severity coverage is incomplete: {method}")
    resampling = report.get("resampling")
    _require(isinstance(resampling, Mapping), "severity resampling metadata missing")
    _require(resampling.get("bootstrap_replicates") == 10_000,
             "severity bootstrap count changed")
    _require(resampling.get("exact_test_assignments") == 256,
             "severity exact test is not the eight-speaker enumeration")
    expected_report: Mapping[str, object] | None = None
    if config is not None:
        _require(project_root is not None,
                 "project root is required for frozen severity recomputation")
        assert project_root is not None
        severity_config = config.get("severity")
        evaluation_config = config.get("evaluation")
        _require(isinstance(severity_config, Mapping)
                 and isinstance(evaluation_config, Mapping),
                 "severity/evaluation configuration is missing")
        _require(
            tuple(str(value) for value in severity_config.get("dysarthric_speakers", ()))
            == EXPECTED_DYSARTHRIC_SPEAKERS,
            "frozen severity speaker inventory/order changed",
        )
        _require(severity_config.get("audited_recordings") == 415
                 and severity_config.get("unique_reading_events") == 329
                 and severity_config.get("event_key") == ["speaker_id", "session", "stem"]
                 and severity_config.get("system_level_comparison_only") is True,
                 "frozen severity scope/policy changed")
        frozen_sources = severity_config.get("frozen_scores")
        _require(isinstance(frozen_sources, Mapping)
                 and set(frozen_sources) == {"mixgop", "uq_gop", "cagop", "ctc_sf"},
                 "frozen severity source inventory changed")
        for path in frozen_sources.values():
            _require(Path(str(path)).is_file(), f"frozen severity source is missing: {path}")
        audited_manifest = Path(str(severity_config["audited_manifest"]))
        _require(audited_manifest.is_file(), "audited severity manifest is missing")
        normalized = load_frozen_score_sources(
            {str(name): Path(str(path)) for name, path in frozen_sources.items()}
        )
        expected_report = run_severity_experiment(
            score_path,
            audited_manifest,
            normalized,
            new_scores_are_phone_level=True,
            new_method=NEW_METHOD,
            n_bootstrap=int(evaluation_config.get("bootstrap_replicates", 10_000)),
            seed=int(config.get("seed", 20260829)),
            expected_manifest_rows=415,
            expected_events=329,
            expected_speakers=8,
        )
        _require(
            set(report) == set(expected_report) | {
                "audited_inference", "frozen_result_sources",
            },
            "severity report key inventory changed",
        )
        for key, expected_value in expected_report.items():
            _require(key in report, f"severity metrics lacks recomputable field {key}")
            _verify_recomputed(report[key], expected_value, f"severity.{key}")
        expected_sources = {
            str(name): {"filename": Path(str(path)).name, "sha256": sha256_file(path)}
            for name, path in sorted(frozen_sources.items())
        }
        _verify_recomputed(report.get("frozen_result_sources"), expected_sources,
                           "severity.frozen_result_sources")
        expected_inference = {
            "manifest_filename": "severity_audited_recordings.jsonl",
            "manifest_sha256": sha256_file(
                project_root / "artifacts" / "manifests" / "severity_audited_recordings.jsonl"
            ),
            "official_logits_cache": str(severity_config["audited_logits_cache"]),
            "n_recordings": 415,
            "n_events": 329,
            "score_all_audited_recordings": True,
        }
        _verify_recomputed(report.get("audited_inference"), expected_inference,
                           "severity.audited_inference")

        # Rebuild the audited inference inventory as an independent coverage check.
        audited = build_audited_severity_events(
            audited_manifest,
            logits_cache=severity_config["audited_logits_cache"],
            expected_manifest_rows=415,
            expected_events=329,
            expected_speakers=8,
            require_files=True,
        )
        _require(audited["n_recordings"] == 415 and audited["n_events"] == 329,
                 "audited severity inventory changed")
        # ``build_severity_inference_manifest`` first discovers recordings in
        # recording-id order, while the persisted privacy-safe artifact has a
        # separate canonical order: event id, then recording id.  Compare the
        # regenerated rows in that documented serialized order so ordering is
        # still audited without treating two identical inventories as unequal.
        _verify_severity_inference_inventory(
            audited_recordings,
            audited["recordings"],
        )

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "method", "n_events", "n_speakers", "event_coverage",
                "kendall_tau_b", "abs_kendall_tau_b", "spearman_rho", "pearson_r",
            }
            _require(reader.fieldnames == [
                "method", "n_events", "n_speakers", "event_coverage",
                "kendall_tau_b", "abs_kendall_tau_b", "spearman_rho", "pearson_r",
            ] and set(reader.fieldnames) == required,
                     "severity CSV header inventory/order changed")
            csv_rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise VerificationError(f"invalid severity CSV: {csv_path}") from error
    _require(len(csv_rows) == 5, "severity CSV must contain five methods")
    _require([str(row.get("method")) for row in csv_rows] == sorted(methods),
             "severity CSV row order changed")
    csv_by_method = {str(row.get("method")): row for row in csv_rows}
    _require(set(csv_by_method) == set(methods), "severity CSV method inventory changed")
    for method, result in methods.items():
        row = csv_by_method[str(method)]
        coverage_block = result["coverage"]
        correlation_block = result.get("correlations")
        _require(isinstance(coverage_block, Mapping) and isinstance(correlation_block, Mapping),
                 f"severity method result is incomplete: {method}")
        _require(int(row["n_events"]) == coverage_block["n_events"],
                 f"severity CSV event count mismatch: {method}")
        _require(int(row["n_speakers"]) == coverage_block["n_speakers"],
                 f"severity CSV speaker count mismatch: {method}")
        _same_number(row["event_coverage"], coverage_block["event_coverage"],
                     f"severity CSV coverage {method}")
        for metric in ("kendall_tau_b", "abs_kendall_tau_b", "spearman_rho", "pearson_r"):
            _same_number(row[metric], correlation_block[metric],
                         f"severity CSV {method} {metric}")
    return {"n_phone_rows": len(score_rows), "n_recordings": score_stats["n_recordings"]}


def verify_manifest(path: str | os.PathLike[str]) -> None:
    item = Path(path)
    try:
        value = read_json(item)
    except ProvenanceError as error:
        raise VerificationError(f"invalid manifest {item}: {error}") from error
    _require(isinstance(value, (dict, list)), f"manifest must be an object or list: {item}")
    records: object = value
    if isinstance(value, dict):
        records = value.get("records", value.get("events", []))
    if not isinstance(records, list) or not records:
        return
    seen: set[tuple[object, ...]] = set()
    for row in records:
        _require(isinstance(row, Mapping), f"manifest record is not an object: {item}")
        if all(key in row for key in ("speaker", "session", "stem")):
            key = (row["speaker"], row["session"], row["stem"])
        elif "event" in row:
            key = (row["event"],)
        elif "event_id" in row:
            key = (row["event_id"],)
        else:
            continue
        _require(key not in seen, f"duplicate reading event in manifest {item}: {key}")
        seen.add(key)


def _verify_cohort_manifest_bundle(
    manifest_dir: Path,
    config: Mapping[str, object],
    cohort: str,
) -> tuple[list[dict[str, object]], Mapping[str, object]]:
    cohorts = config.get("cohorts")
    _require(isinstance(cohorts, Mapping), "frozen cohort configuration is missing")
    patients = tuple(str(value) for value in cohorts.get(cohort, ()))
    healthy = tuple(str(value) for value in cohorts.get("healthy_phn", ()))
    _require(len(patients) == (5 if cohort == "primary5" else 7),
             f"{cohort} speaker inventory changed")
    _require(len(set(patients)) == len(patients) and len(set(healthy)) == 4,
             f"{cohort} or healthy speaker inventory contains duplicates")
    manifest_path = manifest_dir / f"{cohort}.jsonl.gz"
    exclusions_path = manifest_dir / f"{cohort}.exclusions.jsonl.gz"
    audit_path = manifest_dir / f"{cohort}.audit.json"
    folds_path = manifest_dir / f"{cohort}.folds.json"
    for path in (manifest_path, exclusions_path, audit_path, folds_path):
        _require(path.is_file(), f"frozen cohort artifact is missing: {path}")
    verify_deterministic_gzip(manifest_path)
    verify_deterministic_gzip(exclusions_path)
    rows = _read_jsonl(manifest_path)
    exclusions = _read_jsonl(exclusions_path)
    expected_speakers = set(patients) | set(healthy)
    seen_events: set[str] = set()
    observed_speakers: set[str] = set()
    counts: dict[str, int] = {}
    stable_tokens = 0
    uncertain_tokens = 0
    canonical_tokens = 0
    patient_rows = 0
    patient_stable_tokens = 0
    patient_uncertain_tokens = 0
    patient_canonical_tokens = 0
    prompt_values: set[str] = set()
    canonical_sequences: set[tuple[str, ...]] = set()
    phn_microphones: Counter[str] = Counter()
    insertion_totals: Counter[str] = Counter()
    row_key_inventory = set(rows[0])
    _require(
        row_key_inventory == set(MANIFEST_ROW_KEYS)
        or row_key_inventory == set(MANIFEST_ROW_KEYS | MANIFEST_SOURCE_HASH_KEYS),
        f"{cohort} manifest row key inventory changed",
    )
    source_bound_manifest = row_key_inventory == set(
        MANIFEST_ROW_KEYS | MANIFEST_SOURCE_HASH_KEYS
    )
    for row in rows:
        _require(set(row) == row_key_inventory,
                 f"{cohort} manifest row key inventory changed")
        _require(row.get("schema_version") == "da-cf-gop.manifest.v1",
                 f"{cohort} manifest row schema changed")
        speaker = str(row.get("speaker_id", ""))
        event = str(row.get("reading_event_id", ""))
        _require(speaker in expected_speakers, f"{cohort} contains unexpected speaker {speaker}")
        _require(bool(event) and event not in seen_events,
                 f"{cohort} contains a missing or duplicate reading event {event}")
        _require(row.get("audio_microphone") == "headMic",
                 f"{cohort} contains non-headMic audio")
        _require(
            row.get("speaker_group")
            == ("dysarthric" if speaker in patients else "healthy"),
            f"{cohort} manifest speaker group changed for {speaker}",
        )
        session, stem = str(row.get("session", "")), str(row.get("stem", ""))
        _require(event == f"{speaker}_{session}_{stem}",
                 f"{cohort} reading-event key changed for {event}")
        phn_microphone = str(row.get("phn_microphone", ""))
        expected_phn_microphones = {"headMic", "arrayMic"} if speaker in {"F01", "M02"} else {"headMic"}
        _require(phn_microphone in expected_phn_microphones,
                 f"{cohort} PHN microphone policy changed for {speaker}")
        canonical = row.get("canonical_phones")
        tokens = row.get("stable_tokens")
        _require(isinstance(canonical, list) and bool(canonical),
                 f"{cohort} manifest canonical sequence is invalid")
        _require(isinstance(tokens, list), f"{cohort} stable-token list is invalid")
        indices: set[int] = set()
        row_stable_count = 0
        for token in tokens:
            _require(isinstance(token, Mapping) and set(token) == STABLE_TOKEN_KEYS,
                     f"{cohort} stable-token schema changed")
            index = token.get("phone_index")
            _require(type(index) is int and 0 <= int(index) < len(canonical)
                     and int(index) not in indices,
                     f"{cohort} stable-token index is invalid")
            indices.add(int(index))
            _require(type(token.get("stable")) is bool
                     and token.get("canonical_phone") == canonical[int(index)],
                     f"{cohort} token label disagrees with its canonical sequence")
            event_type = token.get("event_type")
            if token.get("stable") is True:
                row_stable_count += 1
                _require(event_type in {"match", "substitution", "deletion"}
                         and token.get("levenshtein_event_type") == event_type
                         and token.get("category_event_type") == event_type,
                         f"{cohort} stable token aligners disagree on event type")
                realized = token.get("realized_phone")
                _require(token.get("levenshtein_realized_phone") == realized
                         and token.get("category_realized_phone") == realized,
                         f"{cohort} stable token aligners disagree on realized phone")
            else:
                _require(event_type == "alignment_uncertain"
                         and token.get("realized_phone") is None,
                         f"{cohort} uncertain-token label changed")
        _require(indices == set(range(len(canonical))),
                 f"{cohort} token labels do not cover the canonical sequence")
        stable_count = row.get("stable_count")
        uncertain_count = row.get("uncertain_count")
        _require(type(stable_count) is int and stable_count == row_stable_count,
                 f"{cohort} stable token count changed")
        _require(type(uncertain_count) is int
                 and uncertain_count == len(canonical) - row_stable_count,
                 f"{cohort} uncertain token count changed")
        _same_number(row.get("stable_fraction"), row_stable_count / len(canonical),
                     f"{cohort} manifest stable fraction")
        seen_events.add(event)
        observed_speakers.add(speaker)
        counts[speaker] = counts.get(speaker, 0) + 1
        canonical_tokens += len(canonical)
        stable_tokens += row_stable_count
        uncertain_tokens += int(uncertain_count)
        prompt_values.add(str(row.get("prompt_id", "")))
        canonical_sequences.add(tuple(str(phone) for phone in canonical))
        phn_microphones[phn_microphone] += 1
        if source_bound_manifest:
            _require(all(_is_sha256(row.get(name)) for name in MANIFEST_SOURCE_HASH_KEYS),
                     f"{cohort} manifest source hash is invalid")
        for key in ("agreed_insertions", "category_insertions", "levenshtein_insertions"):
            value = row.get(key)
            _require(type(value) is int and int(value) >= 0,
                     f"{cohort} insertion count is invalid")
            insertion_totals[key] += int(value)
        if speaker in patients:
            patient_rows += 1
            patient_canonical_tokens += len(canonical)
            patient_stable_tokens += row_stable_count
            patient_uncertain_tokens += int(uncertain_count)
    _require(observed_speakers == expected_speakers,
             f"{cohort} manifest speaker coverage differs")

    exclusion_reasons: Counter[str] = Counter()
    excluded_events: set[str] = set()
    for row in exclusions:
        _require(set(row) == EXCLUSION_ROW_KEYS,
                 f"{cohort} exclusion row key inventory changed")
        speaker = str(row.get("speaker_id", ""))
        event = str(row.get("reading_event_id", ""))
        session, stem = str(row.get("session", "")), str(row.get("stem", ""))
        reason = str(row.get("reason", ""))
        _require(speaker in expected_speakers and event == f"{speaker}_{session}_{stem}"
                 and bool(reason), f"{cohort} exclusion identity/reason is invalid")
        _require(event not in seen_events and event not in excluded_events,
                 f"{cohort} exclusion event is duplicated or included")
        excluded_events.add(event)
        exclusion_reasons[reason] += 1

    audit = read_json(audit_path)
    _require(isinstance(audit, Mapping), f"{cohort} audit is not an object")
    expected_audit_keys = (
        MANIFEST_AUDIT_KEYS | MANIFEST_AUDIT_SOURCE_KEYS
        if source_bound_manifest else MANIFEST_AUDIT_KEYS
    )
    _require(set(audit) == expected_audit_keys, f"{cohort} audit key inventory changed")
    _require(audit.get("schema_version") == "da-cf-gop.manifest.v1",
             f"{cohort} audit schema changed")
    _require(audit.get("cohort") == cohort, f"{cohort} audit cohort changed")
    if source_bound_manifest:
        _require(audit.get("manifest_sha256") == _logical_json_sha256(rows),
                 f"{cohort} audit logical manifest hash mismatch")
        source_inventory = audit.get("input_source_hashes")
        g2p_provenance = audit.get("g2p_provenance")
        _require(isinstance(source_inventory, Mapping)
                 and set(source_inventory) == {"wav", "prompt", "phn"},
                 f"{cohort} source-hash inventory changed")
        for role, path_name, digest_name in (
            ("wav", "wav_path", "wav_sha256"),
            ("prompt", "prompt_path", "prompt_sha256"),
            ("phn", "phn_path", "phn_sha256"),
        ):
            record = source_inventory[role]
            _require(isinstance(record, Mapping)
                     and set(record) == {"n_files", "inventory_sha256"},
                     f"{cohort} {role} source inventory schema changed")
            entries = [
                {
                    "reading_event_id": row["reading_event_id"],
                    "path": row[path_name],
                    "sha256": row[digest_name],
                }
                for row in rows
            ]
            _verify_recomputed(record.get("n_files"), len(entries),
                               f"{cohort}.{role}.n_files")
            _verify_recomputed(
                record.get("inventory_sha256"),
                _logical_json_sha256(entries),
                               f"{cohort}.{role}.inventory_sha256")
        _require(isinstance(g2p_provenance, Mapping)
                 and _is_sha256(g2p_provenance.get("descriptor_sha256")),
                 f"{cohort} G2P provenance is invalid")
        _require({str(row["g2p_descriptor_sha256"]) for row in rows}
                 == {str(g2p_provenance["descriptor_sha256"])},
                 f"{cohort} rows disagree with the frozen G2P descriptor")
    else:
        _require(_is_sha256(audit.get("manifest_sha256")),
                 f"{cohort} legacy logical manifest hash is invalid")
    _require(audit.get("rows_by_speaker") == dict(sorted(counts.items())),
             f"{cohort} audit speaker counts cannot be recomputed")
    _require(audit.get("dysarthric_speakers_requested") == list(patients),
             f"{cohort} audit patient inventory changed")
    _require(audit.get("healthy_speakers_requested") == list(healthy),
             f"{cohort} audit healthy inventory changed")
    exact_audit_values: dict[str, object] = {
        "n_rows": len(rows),
        "n_patient_rows": patient_rows,
        "n_exclusions": len(exclusions),
        "rows_by_speaker": dict(sorted(counts.items())),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "canonical_tokens": canonical_tokens,
        "stable_tokens": stable_tokens,
        "uncertain_tokens": uncertain_tokens,
        "patient_canonical_tokens": patient_canonical_tokens,
        "patient_stable_tokens": patient_stable_tokens,
        "patient_uncertain_tokens": patient_uncertain_tokens,
        "n_unique_prompts": len(prompt_values),
        "n_unique_canonical_sequences": len(canonical_sequences),
        "phn_microphone_counts": dict(sorted(phn_microphones.items())),
        **dict(insertion_totals),
    }
    for key, expected in exact_audit_values.items():
        _verify_recomputed(audit.get(key), expected, f"{cohort}.audit.{key}")
    stable_fraction = stable_tokens / canonical_tokens
    patient_fraction = patient_stable_tokens / patient_canonical_tokens
    _same_number(audit.get("stable_fraction"), stable_fraction,
                 f"{cohort} audit stable fraction")
    _same_number(audit.get("patient_stable_fraction"), patient_fraction,
                 f"{cohort} audit patient stable fraction")
    labels = config.get("labels")
    _require(isinstance(labels, Mapping), "frozen label configuration is missing")
    minimum = float(labels.get("stable_alignment_min_fraction", 0.8))
    _same_number(audit.get("minimum_stable_fraction"), minimum,
                 f"{cohort} audit minimum stable fraction")
    headline_decision = audit.get("exact_substitution_deletion_headline_allowed")
    _require(type(headline_decision) is bool
             and headline_decision == (patient_fraction >= minimum),
             f"{cohort} audit headline decision cannot be recomputed")
    _require(audit.get("audio_policy") == "headMic only"
             and audit.get("reading_event_key") == ["speaker_id", "session", "stem"]
             and audit.get("phn_boundaries_serialized") is False,
             f"{cohort} audit data policy changed")

    fold_document = read_json(folds_path)
    _require(isinstance(fold_document, Mapping), f"{cohort} folds are not an object")
    _require(set(fold_document) == {"schema_version", "folds"},
             f"{cohort} fold-document key inventory changed")
    _require(fold_document.get("schema_version") == "da-cf-gop.folds.v1",
             f"{cohort} fold schema changed")
    folds = fold_document.get("folds")
    _require(isinstance(folds, list) and len(folds) == len(patients),
             f"{cohort} outer fold count changed")
    by_held_out: dict[str, Mapping[str, object]] = {}
    for fold in folds:
        _require(isinstance(fold, Mapping), f"{cohort} fold is not an object")
        _require(set(fold) == {
            "schema_version", "cohort", "fold_id", "held_out_patient",
            "training_patients", "healthy_references", "inner_folds",
        }, f"{cohort} outer-fold key inventory changed")
        _require(fold.get("schema_version") == "da-cf-gop.folds.v1",
                 f"{cohort} outer-fold schema changed")
        held_out = str(fold.get("held_out_patient", ""))
        _require(held_out in patients and held_out not in by_held_out,
                 f"{cohort} outer held-out speaker is invalid")
        by_held_out[held_out] = fold
        _require(fold.get("fold_id") == f"{cohort}__outer_{held_out}",
                 f"{cohort} fold id changed for {held_out}")
        _require(fold.get("cohort") == cohort, f"{cohort} fold cohort changed")
        _require(fold.get("training_patients") == [p for p in patients if p != held_out],
                 f"{cohort} training patients changed for {held_out}")
        _require(fold.get("healthy_references") == list(healthy),
                 f"{cohort} healthy references changed for {held_out}")
        inner = fold.get("inner_folds")
        training = [p for p in patients if p != held_out]
        _require(isinstance(inner, list) and len(inner) == len(training),
                 f"{cohort} inner fold count changed for {held_out}")
        inner_by_held = {
            str(item.get("held_out_patient")): item
            for item in inner if isinstance(item, Mapping)
        }
        _require(set(inner_by_held) == set(training),
                 f"{cohort} inner held-out coverage changed for {held_out}")
        for inner_held, item in inner_by_held.items():
            _require(set(item) == {
                "held_out_patient", "training_patients", "healthy_references",
            }, f"{cohort} inner-fold key inventory changed")
            _require(item.get("training_patients") == [p for p in training if p != inner_held],
                     f"{cohort} inner training speakers changed")
            _require(item.get("healthy_references") == list(healthy),
                     f"{cohort} inner healthy references changed")
    _require(set(by_held_out) == set(patients), f"{cohort} outer fold coverage changed")
    return rows, audit


def _verify_summary_outputs(
    summary_path: Path,
    project_root: Path,
    *,
    expected_config_sha256: str | None = None,
) -> None:
    value = read_json(summary_path)
    _require(isinstance(value, Mapping), f"stage summary is not an object: {summary_path}")
    _require(set(value) == STAGE_SUMMARY_KEYS,
             f"stage summary key inventory changed: {summary_path}")
    _require(value.get("schema_version") == "da-cf-gop.stage-summary.v1",
             f"stage summary schema changed: {summary_path}")
    _require(isinstance(value.get("stage"), str) and bool(value.get("stage")),
             f"stage summary name is invalid: {summary_path}")
    _require(_is_sha256(value.get("config_sha256")),
             f"stage summary config hash is invalid: {summary_path}")
    if expected_config_sha256 is not None:
        _require(value.get("config_sha256") == expected_config_sha256,
                 f"stage summary is stale for the frozen config: {summary_path}")
    _require(isinstance(value.get("training_speakers"), list)
             and value.get("training_speakers")
             == sorted(set(str(item) for item in value.get("training_speakers", []))),
             f"stage summary training-speaker inventory is invalid: {summary_path}")
    for field in ("source_files", "models", "manifests", "upstream_artifacts"):
        inventory = value.get(field)
        _require(isinstance(inventory, Mapping),
                 f"stage summary {field} is not an object: {summary_path}")
        _require(all(isinstance(name, str) and bool(name) and _is_sha256(digest)
                     for name, digest in inventory.items()),
                 f"stage summary {field} has an invalid hash: {summary_path}")
    exclusions = value.get("exclusions")
    _require(isinstance(exclusions, Mapping)
             and all(isinstance(name, str) and type(count) is int and count >= 0
                     for name, count in exclusions.items()),
             f"stage summary exclusions are invalid: {summary_path}")
    _require(isinstance(value.get("details"), Mapping),
             f"stage summary details are invalid: {summary_path}")
    _require("outputs" in value, f"stage summary lacks outputs: {summary_path}")
    outputs = value["outputs"]
    _require(isinstance(outputs, Mapping) and bool(outputs),
             f"summary output inventory is empty: {summary_path}")
    for name, metadata in outputs.items():
        if isinstance(metadata, str):
            details = value.get("details")
            cohort = str(details.get("cohort", "")) if isinstance(details, Mapping) else ""
            stage = str(value.get("stage", ""))
            artifacts = project_root / "artifacts"
            mappings: dict[str, Path] = {}
            if stage in {"build-manifests", "build-severity-training-manifest"}:
                stem = cohort or "severity_training8"
                mappings = {
                    "manifest": artifacts / "manifests" / f"{stem}.jsonl.gz",
                    "audit": artifacts / "manifests" / f"{stem}.audit.json",
                    "exclusions": artifacts / "manifests" / f"{stem}.exclusions.jsonl.gz",
                    "folds": artifacts / "manifests" / f"{stem}.folds.json",
                }
            elif stage == "extract-logits":
                mappings = {"logit_index": artifacts / "cache" / "logits.index.json"}
            elif stage == "extract-severity-training-logits":
                mappings = {
                    "severity_training_logit_index":
                        artifacts / "cache" / "severity_training8.logits.index.json"
                }
            elif stage == "run-phone-loso":
                base = artifacts / "evaluation" / cohort
                mappings = {
                    "phone_predictions": base / "phone_predictions.jsonl.gz",
                    "runtime_source_predictions": base / "runtime_source_predictions.jsonl.gz",
                    "inner_oof_predictions": base / "inner_oof_predictions.jsonl.gz",
                    "specific_output_gates": base / "specific_output_gates.json",
                    "metrics": base / "metrics.json",
                    "method_metrics": base / "method_metrics.csv",
                    "speaker_metrics": base / "speaker_metrics.csv",
                    "final_assessment": artifacts / "evaluation" / "final_assessment.json",
                }
            elif stage == "export-llm":
                mappings = {
                    "llm_phone_evidence": artifacts / "runtime" / "llm_phone_evidence.jsonl"
                }
            elif stage == "score-severity":
                mappings = {
                    "severity_phone_scores": artifacts / "evaluation" / "severity_phone_scores.jsonl.gz",
                    "severity_phone_scores_provenance": artifacts / "evaluation" / "severity_phone_scores.provenance.json",
                }
            elif stage == "run-severity":
                mappings = {
                    "severity_metrics": artifacts / "evaluation" / "severity_metrics.json",
                    "severity_metrics_csv": artifacts / "evaluation" / "severity_metrics.csv",
                    "severity_inference_manifest": artifacts / "manifests" / "severity_audited_recordings.jsonl",
                }
            _require(name in mappings, f"unknown summary output {stage}.{name}")
            path = mappings[name]
            _require(path.is_file(), f"summary output is missing: {path}")
            _require(_is_sha256(metadata) and sha256_file(path) == metadata,
                     f"summary output hash changed: {path}")
            continue
        _require(isinstance(metadata, Mapping), f"invalid output metadata: {name}")
        if set(metadata) == {"path", "kind", "bytes", "sha256"}:
            try:
                verify_path_record(metadata, root=project_root)
            except ProvenanceError as error:
                raise VerificationError(str(error)) from error
            continue
        _require({"bytes", "sha256"}.issubset(metadata),
                 f"output metadata lacks size/hash: {name}")
        path = (summary_path.parent / str(name)).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as error:
            raise VerificationError(f"summary output escapes project: {path}") from error
        _require(path.is_file(), f"summary output is missing: {path}")
        _require(path.stat().st_size == int(metadata["bytes"]),
                 f"summary output size changed: {path}")
        _require(sha256_file(path) == metadata["sha256"],
                 f"summary output hash changed: {path}")


def _prediction_paths(evaluation_dir: Path) -> list[Path]:
    combined = evaluation_dir / "phone_predictions.jsonl.gz"
    if combined.is_file():
        return [combined]
    return sorted(evaluation_dir.glob("**/phone_predictions.jsonl.gz"))


def _metrics_paths(evaluation_dir: Path) -> list[Path]:
    combined = evaluation_dir / "metrics.json"
    if combined.is_file():
        return [combined]
    return sorted(evaluation_dir.glob("**/metrics.json"))


def _verify_prediction_protocol(
    rows: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    manifests: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    methods = tuple(str(value) for value in config.get("methods", ())) + tuple(
        str(value) for value in config.get("ablations", ())
    )
    _require(methods == FROZEN_METHODS, "frozen 16-method inventory or order changed")
    cohorts = config.get("cohorts")
    _require(isinstance(cohorts, Mapping), "frozen cohorts are missing")
    for cohort in ("primary5", "sensitivity7"):
        local = [row for row in rows if row.get("cohort") == cohort]
        _require(bool(local), f"prediction artifact is missing {cohort}")
        observed_methods = {str(row.get("method")) for row in local}
        _require(observed_methods == set(FROZEN_METHODS),
                 f"{cohort} does not contain exactly 16 frozen methods")
        expected_speakers = set(str(value) for value in cohorts.get(cohort, ()))
        observed_speakers = {str(row.get("speaker")) for row in local}
        _require(observed_speakers == expected_speakers,
                 f"{cohort} held-out speaker coverage changed")
        for row in local:
            speaker = str(row.get("speaker", ""))
            _require(row.get("fold") == f"{cohort}__outer_{speaker}",
                      f"{cohort} prediction is assigned to the wrong outer fold")
            _require(row.get("stable") is True,
                     f"{cohort} evaluation contains a non-stable token")

        manifest_rows = manifests.get(cohort)
        _require(isinstance(manifest_rows, Sequence),
                 f"{cohort} manifest rows are unavailable for coverage verification")
        expected_tokens: dict[tuple[str, str, int, str], tuple[object, object, bool]] = {}
        for manifest_row in manifest_rows:
            speaker = str(manifest_row.get("speaker_id", ""))
            if speaker not in expected_speakers:
                continue
            event = str(manifest_row.get("reading_event_id", ""))
            stable_rows = manifest_row.get("stable_tokens")
            _require(isinstance(stable_rows, list),
                     f"{cohort} manifest stable-token coverage is unavailable")
            for token in stable_rows:
                assert isinstance(token, Mapping)  # validated by the manifest audit
                if token.get("stable") is not True:
                    continue
                target = str(token["canonical_phone"])
                phone_index = int(token["phone_index"])
                event_type = str(token["event_type"])
                gold_event = "correct" if event_type == "match" else event_type
                gold_realized = (
                    target if event_type == "match"
                    else token.get("realized_phone") if event_type == "substitution"
                    else None
                )
                key = (speaker, event, phone_index, target)
                _require(key not in expected_tokens,
                         f"{cohort} manifest has duplicate stable token {key}")
                expected_tokens[key] = (gold_event, gold_realized, True)

        reference_method = FROZEN_METHODS[0]
        reference_rows = [row for row in local if row.get("method") == reference_method]
        observed_tokens = {
            (
                str(row["speaker"]), str(row["event"]), int(row["phone_index"]),
                str(row["target"]),
            ): (row["gold_event"], row["gold_realized"], row["stable"])
            for row in reference_rows
        }
        _require(observed_tokens == expected_tokens,
                 f"{cohort} prediction coverage differs from frozen stable tokens")


def _expected_final_assessment(
    primary: Mapping[str, object], sensitivity: Mapping[str, object]
) -> dict[str, object]:
    primary_success_block = primary.get("success_assessment")
    sensitivity_success_block = sensitivity.get("success_assessment")
    sensitivity_paired = sensitivity.get("paired_comparison")
    _require(
        isinstance(primary_success_block, Mapping)
        and isinstance(sensitivity_success_block, Mapping)
        and isinstance(sensitivity_paired, Mapping),
        "cohort reports lack final-assessment inputs",
    )
    primary_success = bool(primary_success_block.get("cohort_success"))
    difference = sensitivity_success_block.get("auprc_difference")
    sensitivity_positive = difference is not None and float(difference) > 0.0
    n_improved = int(sensitivity_paired.get("n_improved", -1))
    bootstrap = sensitivity_paired.get("bootstrap")
    exact = sensitivity_paired.get("exact_sign_flip")
    _require(isinstance(bootstrap, Mapping) and isinstance(exact, Mapping),
             "sensitivity paired inference is missing")
    ci_low = float(bootstrap["ci_low"])
    p_value = float(exact["p_two_sided"])
    supported = bool(
        primary_success and sensitivity_positive and n_improved >= 5
        and ci_low > 0.0 and p_value < 0.05
    )
    return {
        "schema_version": "da-cf-gop.final-assessment.v1",
        "primary5_success": primary_success,
        "sensitivity7_direction_positive": sensitivity_positive,
        "sensitivity7_at_least_5_of_7_improved": n_improved >= 5,
        "sensitivity7_n_improved": n_improved,
        "sensitivity7_paired_bootstrap_ci_low": ci_low,
        "sensitivity7_exact_sign_flip_p_two_sided": p_value,
        "strictly_supported_superiority": supported,
        "permitted_wording": (
            "statistically_supported_superiority"
            if supported else "point_estimate_or_negative_result"
        ),
        "sensitivity7_cannot_rescue_primary5": True,
        "test_set_driven_retuning_permitted": False,
        "source_metrics_sha256": {},  # filled by caller from immutable files
    }


def _verify_llm_export(
    root: Path,
    config: Mapping[str, object],
    predictions: Sequence[Mapping[str, object]],
) -> int:
    from .cli import (
        RUNTIME_SOURCE_KEYS,
        _load_fold_gates,
        _recompute_fold_gates,
        _validate_runtime_source_inventory,
    )
    from .llm_schema import LOCAL_IDENTIFIER_SCHEME, SCHEMA_VERSION

    artifacts = root / "artifacts"
    cohort_dir = artifacts / "evaluation" / "primary5"
    source_path = cohort_dir / "runtime_source_predictions.jsonl.gz"
    inner_path = cohort_dir / "inner_oof_predictions.jsonl.gz"
    gates_path = cohort_dir / "specific_output_gates.json"
    runtime_path = artifacts / "runtime" / "llm_phone_evidence.jsonl"
    summary_path = artifacts / "summaries" / "export-llm.json"
    for path in (source_path, inner_path, gates_path, runtime_path, summary_path):
        _require(path.is_file(), f"LLM export dependency is missing: {path}")
    verify_deterministic_gzip(source_path)
    verify_deterministic_gzip(inner_path)
    source_rows = _read_jsonl(source_path)
    llm = config.get("llm_export")
    cohorts = config.get("cohorts")
    _require(isinstance(llm, Mapping) and isinstance(cohorts, Mapping),
             "LLM/cohort configuration is missing")
    _require(llm.get("schema_version") == "da-cf-gop.llm.v1"
             and llm.get("patient_facing") is False,
             "frozen LLM export policy changed")
    patients = tuple(str(value) for value in cohorts.get("primary5", ()))
    try:
        _validate_runtime_source_inventory(
            source_rows,
            predictions,
            _read_jsonl(artifacts / "manifests" / "primary5.jsonl.gz"),
            patients=patients,
        )
    except ValueError as error:
        raise VerificationError(f"invalid LLM runtime source: {error}") from error
    expected_fold_speakers = {
        f"primary5__outer_{held_out}": tuple(p for p in patients if p != held_out)
        for held_out in patients
    }
    precision = float(llm.get("precision_floor", 0.8))
    coverage = float(llm.get("coverage_floor", 0.1))
    margin = float(llm.get("alternative_margin_floor", 0.1))
    error_threshold = float(llm.get("error_threshold", 0.5))
    _require(precision >= 0.8 and coverage >= 0.1 and margin >= 0.1,
             "LLM disclosure safety floors were weakened")
    stored = _load_fold_gates(
        gates_path, precision_floor=precision,
        coverage_floor=coverage, margin_floor=margin,
    )
    _require(set(stored) == set(expected_fold_speakers),
             "LLM gates do not cover the exact five outer folds")
    recomputed = _recompute_fold_gates(
        _read_jsonl(inner_path),
        expected_fold_speakers=expected_fold_speakers,
        precision_floor=precision,
        coverage_floor=coverage,
        margin_floor=margin,
        error_threshold=error_threshold,
        threshold_grid=tuple(llm.get("threshold_grid", ())),
    )
    for fold in stored:
        _verify_recomputed(stored[fold].to_dict(), recomputed[fold].to_dict(),
                           f"LLM gate {fold}")

    identifier_namespace = f"{SCHEMA_VERSION}:{config['_config_hash']}"
    expected_records = [
        build_llm_record(
            row,
            recomputed[str(row["fold"])],
            identifier_namespace=identifier_namespace,
            error_threshold=error_threshold,
        )
        for row in source_rows
    ]
    expected_records.sort(
        key=lambda row: (str(row["utterance_record_id"]), int(row["phone_index"]))
    )
    try:
        records = read_llm_records(runtime_path)
    except ValueError as error:
        raise VerificationError(f"invalid LLM runtime artifact: {error}") from error
    _verify_recomputed(records, expected_records, "LLM runtime records/local identifiers")

    summary = read_json(summary_path)
    _require(isinstance(summary, Mapping), "LLM stage summary is not an object")
    upstream = summary.get("upstream_artifacts")
    details = summary.get("details")
    outputs = summary.get("outputs")
    _require(isinstance(upstream, Mapping) and isinstance(details, Mapping)
             and isinstance(outputs, Mapping), "LLM stage summary is incomplete")
    expected_hashes = {
        "phone_predictions": sha256_file(cohort_dir / "phone_predictions.jsonl.gz"),
        "runtime_source_predictions": sha256_file(source_path),
        "inner_oof_predictions": sha256_file(inner_path),
        "specific_output_gates": sha256_file(gates_path),
    }
    _verify_recomputed(upstream, expected_hashes, "LLM summary upstream hashes")
    _require(outputs.get("llm_phone_evidence") == sha256_file(runtime_path),
             "LLM summary output hash mismatch")
    _require(details.get("n_records") == len(records), "LLM record count mismatch")
    expected_decisions = dict(sorted(Counter(str(row["decision"]) for row in records).items()))
    expected_details = {
        "cohort": "primary5",
        "n_records": len(records),
        "n_outer_folds": 5,
        "decisions": expected_decisions,
        "deployment": "local_offline",
        "identifier_scheme": LOCAL_IDENTIFIER_SCHEME,
        "identifier_namespace": "schema_version+frozen_config_sha256",
        "error_threshold": error_threshold,
        "patient_facing": False,
        "llm_invoked": False,
    }
    _verify_recomputed(details, expected_details, "LLM summary details/gates/identifiers")
    _verify_recomputed(
        summary.get("manifests"),
        {"primary5": sha256_file(artifacts / "manifests" / "primary5.jsonl.gz")},
        "LLM summary manifest binding",
    )
    _verify_recomputed(summary.get("source_files"),
                       {"frozen_config": str(config["_config_hash"])},
                       "LLM summary config source")
    _verify_recomputed(summary.get("training_speakers"), [],
                       "LLM summary training speakers")
    _verify_recomputed(summary.get("models"), {}, "LLM summary models")
    _verify_recomputed(summary.get("exclusions"), {}, "LLM summary exclusions")
    return len(records)


def verify_project(
    project_root: str | os.PathLike[str],
    *,
    expected_cohorts: Sequence[str] = ("primary5", "sensitivity7"),
) -> dict[str, Any]:
    """Verify a complete frozen run and return a compact audit summary."""

    root = Path(project_root).resolve()
    full_protocol = (root / "configs" / "frozen_v1.json").is_file()
    config: Mapping[str, object] | None = None
    manifest_audits: dict[str, Mapping[str, object]] = {}
    manifest_rows_by_cohort: dict[str, Sequence[Mapping[str, object]]] = {}
    if full_protocol:
        try:
            config = load_config(root / "configs" / "frozen_v1.json")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise VerificationError(f"cannot load frozen configuration: {error}") from error
    artifacts = root / "artifacts"
    manifest_dir = artifacts / "manifests"
    cache_dir = artifacts / "cache"
    evaluation_dir = artifacts / "evaluation"
    runtime_path = artifacts / "runtime" / "llm_phone_evidence.jsonl"
    _require(root.is_dir(), f"project root is missing: {root}")
    for directory in (manifest_dir, cache_dir, evaluation_dir):
        _require(directory.is_dir(), f"artifact directory is missing: {directory}")

    if full_protocol:
        assert config is not None
        manifests = []
        for cohort in expected_cohorts:
            cohort_rows, audit = _verify_cohort_manifest_bundle(manifest_dir, config, cohort)
            manifest_rows_by_cohort[cohort] = cohort_rows
            manifest_audits[cohort] = audit
            manifests.append(manifest_dir / f"{cohort}.jsonl.gz")
    else:
        manifests = sorted(manifest_dir.glob("*.json"))
        _require(bool(manifests), "no frozen manifests found")
        for manifest in manifests:
            verify_manifest(manifest)

    forbidden_cache = [*cache_dir.rglob("*.pkl"), *cache_dir.rglob("*.pickle")]
    _require(not forbidden_cache, f"pickle cache is forbidden: {forbidden_cache[0] if forbidden_cache else ''}")
    npz_paths = sorted(cache_dir.rglob("*.npz"))
    _require(bool(npz_paths), "no NPZ cache artifacts found")
    for npz in npz_paths:
        verify_deterministic_npz(npz)
        descriptor_path = companion_descriptor_path(npz)
        _require(descriptor_path.is_file(), f"cache descriptor is missing: {descriptor_path}")
        descriptor = read_json(descriptor_path)
        _require(isinstance(descriptor, Mapping), f"cache descriptor is not an object: {descriptor_path}")
        artifact_record = descriptor.get("artifact")
        descriptor_root: Path | None = root
        if isinstance(artifact_record, Mapping) and Path(str(artifact_record.get("path", ""))).is_absolute():
            descriptor_root = None
        try:
            observed = validate_cache_descriptor(descriptor, root=descriptor_root)
        except ProvenanceError as error:
            raise VerificationError(f"invalid cache descriptor {descriptor_path}: {error}") from error
        _require(observed.resolve() == npz.resolve(),
                 f"descriptor points to a different artifact: {descriptor_path}")

    prediction_paths = _prediction_paths(evaluation_dir)
    _require(bool(prediction_paths), "phone prediction artifact is missing")
    predictions: list[dict[str, object]] = []
    for path in prediction_paths:
        predictions.extend(verify_evaluation_artifact(path))
    # Validate duplicates across separate cohort files as well.
    try:
        validate_prediction_rows(predictions)
    except MetricError as error:
        raise VerificationError(f"combined prediction artifacts are invalid: {error}") from error
    verify_matched_method_coverage(predictions)
    observed_cohorts = {str(row["cohort"]) for row in predictions}
    _require(set(expected_cohorts).issubset(observed_cohorts),
             f"missing expected cohorts: {sorted(set(expected_cohorts) - observed_cohorts)}")
    if full_protocol:
        assert config is not None
        _require(observed_cohorts == set(expected_cohorts), "unexpected prediction cohort found")
        _verify_prediction_protocol(predictions, config, manifest_rows_by_cohort)

    metric_paths = _metrics_paths(evaluation_dir)
    _require(bool(metric_paths), "metrics.json is missing")
    metric_groups: set[tuple[str, str]] = set()
    reports_by_cohort: dict[str, Mapping[str, object]] = {}
    for path in metric_paths:
        value = read_json(path)
        _require(isinstance(value, Mapping), f"metrics report is not an object: {path}")
        path_cohort = path.parent.name if path.parent != evaluation_dir else None
        verify_metrics_report(
            value,
            predictions,
            config=config if full_protocol else None,
            expected_cohort=path_cohort if full_protocol else None,
            manifest_audit=manifest_audits.get(str(path_cohort)),
        )
        if path_cohort is not None:
            _require(path_cohort not in reports_by_cohort,
                     f"duplicate metrics report for {path_cohort}")
            reports_by_cohort[path_cohort] = value
        metric_groups.update(
            (str(item["cohort"]), str(item["method"]))
            for item in value["results"] if isinstance(item, Mapping)
        )
        if full_protocol:
            verify_metrics_csv(
                path.parent / "method_metrics.csv", report=value, level="method"
            )
            verify_metrics_csv(
                path.parent / "speaker_metrics.csv", report=value, level="speaker"
            )
        else:
            verify_metrics_csv(path.parent / "method_metrics.csv")
    expected_groups = {(str(row["cohort"]), str(row["method"])) for row in predictions}
    _require(metric_groups == expected_groups, "metrics reports do not cover every method/cohort")

    severity_summary = {"n_phone_rows": 0, "n_recordings": 0}
    if full_protocol:
        _require(set(reports_by_cohort) == set(expected_cohorts),
                 "metrics reports do not cover the exact frozen cohorts")
        final_assessment = evaluation_dir / "final_assessment.json"
        _require(final_assessment.is_file(), "cross-cohort final assessment is missing")
        assessment = read_json(final_assessment)
        _require(isinstance(assessment, Mapping), "final assessment is not an object")
        _require(
            assessment.get("schema_version") == "da-cf-gop.final-assessment.v1",
            "final assessment schema changed",
        )
        expected_assessment = _expected_final_assessment(
            reports_by_cohort["primary5"], reports_by_cohort["sensitivity7"]
        )
        expected_assessment["source_metrics_sha256"] = {
            "primary5": sha256_file(evaluation_dir / "primary5" / "metrics.json"),
            "sensitivity7": sha256_file(evaluation_dir / "sensitivity7" / "metrics.json"),
        }
        _verify_recomputed(assessment, expected_assessment, "final_assessment")
        assert config is not None
        severity_summary = verify_severity_artifacts(
            evaluation_dir, project_root=root, config=config
        )
        runtime_count = _verify_llm_export(root, config, predictions)
        runtime_records: Sequence[object] = [None] * runtime_count
    else:
        _require(runtime_path.is_file(), f"runtime artifact is missing: {runtime_path}")
        try:
            runtime_records = read_llm_records(runtime_path)
        except ValueError as error:
            raise VerificationError(f"invalid LLM runtime artifact: {error}") from error

    summary_dir = artifacts / "summaries"
    summaries = sorted(summary_dir.glob("*.json")) if summary_dir.is_dir() else []
    if full_protocol:
        required_summaries = {
            "build-manifests.primary5.json", "build-manifests.sensitivity7.json",
            "extract-logits.json", "run-phone-loso.primary5.json",
            "run-phone-loso.sensitivity7.json", "score-severity.json",
            "run-severity.json", "export-llm.json",
        }
        _require(required_summaries.issubset({path.name for path in summaries}),
                 "required frozen stage summaries are missing")
    for summary in summaries:
        try:
            _verify_summary_outputs(
                summary,
                root,
                expected_config_sha256=(
                    str(config["_config_hash"])
                    if full_protocol and config is not None else None
                ),
            )
        except ProvenanceError as error:
            raise VerificationError(str(error)) from error

    return {
        "status": "PASS",
        "n_manifests": len(manifests),
        "n_npz": len(npz_paths),
        "n_prediction_rows": len(predictions),
        "n_runtime_rows": len(runtime_records),
        "n_severity_phone_rows": severity_summary["n_phone_rows"],
        "n_severity_recordings": severity_summary["n_recordings"],
        "cohorts": sorted(observed_cohorts),
        "methods": sorted({str(row["method"]) for row in predictions}),
    }


def verify_and_print(project_root: str | os.PathLike[str]) -> None:
    """Verify and print the sole success token required by the frozen protocol."""

    verify_project(project_root)
    print("PASS")
