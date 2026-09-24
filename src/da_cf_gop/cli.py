"""Unified command-line interface for the frozen DA-CF-GoP workflow.

The module deliberately keeps heavyweight acoustic-model imports behind the
individual command dispatches.  This makes ``--help``, artifact export, and
verification usable without loading PyTorch or Transformers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import sys
from typing import Any

from .artifacts import ArtifactLayout, read_jsonl, write_jsonl, write_stage_summary
from .config import CODE_ROOT, DEFAULT_CONFIG, load_config
from .llm_schema import (
    ARPABET_39,
    LOCAL_IDENTIFIER_SCHEME,
    LLMContractError,
    SCHEMA_VERSION,
    SpecificOutputGate,
    build_llm_record,
    fit_specific_output_gate,
)
from .provenance import read_json, sha256_file


COHORTS = ("primary5", "sensitivity7")
RUNTIME_SOURCE_KEYS = frozenset(
    {
        "cohort",
        "fold",
        "speaker",
        "event",
        "phone_index",
        "target",
        "p_error",
        "top_alt",
        "p_alt_given_error",
        "top2_margin",
    }
)
INNER_OOF_KEYS = frozenset(
    {
        "event", "fold", "gold_event", "gold_realized",
        "inner_validation_speaker", "p_alt_given_error", "p_error",
        "phone_index", "speaker", "stable", "target", "top2_margin", "top_alt",
    }
)
ADAPTED_METHOD = "da_cf_adapted"
DELETE = "<DEL>"


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen workflow parser without importing model code."""

    parser = argparse.ArgumentParser(
        prog="da-cf-gop",
        description="Run the frozen DA-CF-GoP data, scoring, and audit workflow.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="frozen JSON configuration (default: configs/frozen_v1.json)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "build-manifests",
        help="freeze TORGO event manifests, labels, and nested LOSO folds",
    )

    logits = subparsers.add_parser(
        "extract-logits",
        help="extract and content-bind official CTC-SF raw logits",
    )
    logits.add_argument(
        "--cohort",
        choices=COHORTS,
        default="sensitivity7",
        help="manifest to extract (sensitivity7 includes every patient event)",
    )
    logits.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="development-only positive utterance limit",
    )

    loso = subparsers.add_parser(
        "run-phone-loso",
        help="run the frozen nested patient-LOSO phone experiment",
    )
    loso.add_argument("--cohort", required=True, choices=COHORTS)

    parikh = subparsers.add_parser(
        "run-parikh2025",
        help="compare Parikh PP-AF RPS/UPS using existing frozen logits and folds",
    )
    parikh.add_argument("--cohort", choices=COHORTS, default="primary5")
    parikh.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parikh.add_argument("--artifacts", type=Path, default=CODE_ROOT / "artifacts")

    dual = subparsers.add_parser("run-dual-graph", help="run independent dual-graph CTC diagnosis and matched baselines")
    dual.add_argument("--cohort", choices=COHORTS, default="primary5")
    dual.add_argument("--artifacts", type=Path, default=CODE_ROOT / "artifacts")
    dual.add_argument("--dual-config", type=Path, default=CODE_ROOT / "configs" / "dual_graph_v1.json")
    dual.add_argument("--reuse-baselines", action="store_true")

    subparsers.add_parser(
        "run-severity",
        help="run the matched, speaker-first secondary severity experiment",
    )
    subparsers.add_parser(
        "export-llm",
        help="export the privacy-minimized primary5 runtime evidence contract",
    )

    verify = subparsers.add_parser(
        "verify",
        help="audit a complete frozen run; successful stdout is exactly PASS",
    )
    verify.add_argument(
        "--project-root",
        type=Path,
        default=CODE_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LLMContractError(f"{label} must be a JSON object")
    return value


def _load_fold_gates(
    path: Path,
    *,
    precision_floor: float,
    coverage_floor: float,
    margin_floor: float,
) -> dict[str, SpecificOutputGate]:
    try:
        document = _require_mapping(read_json(path), "specific-output gate document")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"primary5 specific-output gates are absent: {path}; "
            "run run-phone-loso --cohort primary5 first"
        ) from error
    if set(document) != {"schema_version", "cohort", "folds"}:
        raise LLMContractError("specific-output gate document key inventory changed")
    if document.get("schema_version") != "da-cf-gop.specific-output-gates.v1":
        raise LLMContractError("specific-output gate schema version changed")
    if document.get("cohort") != "primary5":
        raise LLMContractError("specific-output gates are not from primary5")
    raw_folds = _require_mapping(document.get("folds"), "specific-output gates.folds")
    if not raw_folds:
        raise LLMContractError("specific-output gate document has no outer folds")

    gates: dict[str, SpecificOutputGate] = {}
    for fold, raw_gate in raw_folds.items():
        fold_id = str(fold)
        if not fold_id or fold_id in gates:
            raise LLMContractError("specific-output gate fold ids are invalid")
        payload = _require_mapping(raw_gate, f"specific-output gate {fold_id}")
        expected_gate_keys = set(SpecificOutputGate.__dataclass_fields__)
        if set(payload) != expected_gate_keys:
            raise LLMContractError(
                f"specific-output gate {fold_id} key inventory changed"
            )
        try:
            gate = SpecificOutputGate(**dict(payload))
        except TypeError as error:
            raise LLMContractError(
                f"specific-output gate {fold_id} does not match the frozen schema"
            ) from error
        # Validate the fields that control disclosure even if no row happens to
        # exercise the enabled branch.
        if type(gate.enabled) is not bool:
            raise LLMContractError(f"specific-output gate {fold_id} enabled must be boolean")
        numeric_values = (gate.min_margin, gate.coverage)
        try:
            checked_values = tuple(float(value) for value in numeric_values)
        except (TypeError, ValueError) as error:
            raise LLMContractError(
                f"specific-output gate {fold_id} numeric fields are invalid"
            ) from error
        if not all(math.isfinite(value) for value in checked_values):
            raise LLMContractError(f"specific-output gate {fold_id} contains non-finite values")
        if not math.isclose(float(gate.min_margin), margin_floor, abs_tol=0.0, rel_tol=0.0):
            raise LLMContractError(f"specific-output gate {fold_id} margin is invalid")
        if not 0.0 <= float(gate.coverage) <= 1.0:
            raise LLMContractError(f"specific-output gate {fold_id} coverage is invalid")
        if type(gate.n_eligible) is not int or type(gate.n_authorized) is not int:
            raise LLMContractError(f"specific-output gate {fold_id} counts must be integers")
        if not 0 <= gate.n_authorized <= gate.n_eligible:
            raise LLMContractError(f"specific-output gate {fold_id} counts are invalid")
        if gate.reason is not None and not isinstance(gate.reason, str):
            raise LLMContractError(f"specific-output gate {fold_id} reason is invalid")
        if gate.enabled:
            try:
                threshold = float(gate.probability_threshold)  # type: ignore[arg-type]
                achieved_precision = float(gate.precision)  # type: ignore[arg-type]
            except (TypeError, ValueError) as error:
                raise LLMContractError(
                    f"enabled specific-output gate {fold_id} lacks numeric thresholds"
                ) from error
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise LLMContractError(
                    f"enabled specific-output gate {fold_id} lacks a probability threshold"
                )
            if not math.isfinite(achieved_precision) or achieved_precision < precision_floor:
                raise LLMContractError(
                    f"enabled specific-output gate {fold_id} violates the precision floor"
                )
            if float(gate.coverage) < coverage_floor:
                raise LLMContractError(
                    f"enabled specific-output gate {fold_id} violates the coverage floor"
                )
            if gate.reason is not None:
                raise LLMContractError(f"enabled specific-output gate {fold_id} has a reason")
        elif gate.probability_threshold is not None or gate.precision is not None:
            raise LLMContractError(
                f"disabled specific-output gate {fold_id} exposes a threshold/precision"
            )
        gates[fold_id] = gate
    return gates


def _recompute_fold_gates(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_fold_speakers: Mapping[str, Sequence[str]],
    precision_floor: float,
    coverage_floor: float,
    margin_floor: float,
    error_threshold: float,
    threshold_grid: Sequence[float],
) -> dict[str, SpecificOutputGate]:
    grouped: dict[str, list[Mapping[str, object]]] = {
        str(fold): [] for fold in expected_fold_speakers
    }
    seen: set[tuple[str, str, str, int]] = set()
    for index, row in enumerate(rows, start=1):
        if set(row) != INNER_OOF_KEYS:
            raise LLMContractError(
                f"inner-OOF row {index} must use the exact frozen key set"
            )
        fold = str(row.get("fold", ""))
        if fold not in grouped:
            raise LLMContractError(f"inner-OOF row {index} has unexpected fold {fold!r}")
        speaker = str(row.get("speaker", ""))
        if speaker != str(row.get("inner_validation_speaker", "")):
            raise LLMContractError("inner-OOF speaker and validation speaker disagree")
        if speaker not in set(str(value) for value in expected_fold_speakers[fold]):
            raise LLMContractError(
                f"inner-OOF fold {fold} contains an invalid validation speaker {speaker}"
            )
        phone_index = row.get("phone_index")
        if type(phone_index) is not int or int(phone_index) < 0:
            raise LLMContractError("inner-OOF phone_index must be non-negative")
        event = str(row.get("event", ""))
        target = str(row.get("target", ""))
        top_alt = row.get("top_alt")
        if not event or target not in ARPABET_39:
            raise LLMContractError("inner-OOF event/target is invalid")
        if top_alt not in ((ARPABET_39 - {target}) | {DELETE}):
            raise LLMContractError("inner-OOF top alternative is invalid")
        if row.get("stable") is not True:
            raise LLMContractError("inner-OOF gate rows must be stable tokens")
        gold_event = row.get("gold_event")
        gold_realized = row.get("gold_realized")
        if (
            gold_event not in {"correct", "substitution", "deletion"}
            or (gold_event == "correct" and gold_realized != target)
            or (
                gold_event == "substitution"
                and (gold_realized not in ARPABET_39 or gold_realized == target)
            )
            or (gold_event == "deletion" and gold_realized not in {None, DELETE})
        ):
            raise LLMContractError("inner-OOF gold event is invalid")
        for field in ("p_error", "p_alt_given_error", "top2_margin"):
            try:
                probability = float(row.get(field))
            except (TypeError, ValueError) as error:
                raise LLMContractError(f"inner-OOF {field} must be numeric") from error
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise LLMContractError(f"inner-OOF {field} must be a probability")
        key = (fold, speaker, event, int(phone_index))
        if key in seen:
            raise LLMContractError(f"duplicate inner-OOF token: {key}")
        seen.add(key)
        grouped[fold].append(row)
    if any(not values for values in grouped.values()):
        missing = sorted(fold for fold, values in grouped.items() if not values)
        raise LLMContractError(f"inner-OOF artifact has empty outer folds: {missing}")
    for fold, values in grouped.items():
        observed = {str(row["speaker"]) for row in values}
        expected = set(str(value) for value in expected_fold_speakers[fold])
        if observed != expected:
            raise LLMContractError(
                f"inner-OOF fold {fold} does not cover every inner validation speaker"
            )
    return {
        fold: fit_specific_output_gate(
            values,
            min_precision=precision_floor,
            min_coverage=coverage_floor,
            min_margin=margin_floor,
            error_threshold=error_threshold,
            threshold_grid=threshold_grid,
        )
        for fold, values in sorted(grouped.items())
    }


def _runtime_token_key(row: Mapping[str, object]) -> tuple[str, str, str, int, str]:
    phone_index = row.get("phone_index")
    if type(phone_index) is not int or int(phone_index) < 0:
        raise LLMContractError("runtime source phone_index must be non-negative")
    return (
        str(row.get("fold", "")),
        str(row.get("speaker", "")),
        str(row.get("event", "")),
        int(phone_index),
        str(row.get("target", "")),
    )


def _validate_runtime_source_inventory(
    rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    patients: Sequence[str],
) -> None:
    """Bind all inference tokens to prompts and stable rows to evaluation.

    ``runtime_source_predictions`` is written before held-out PHN is attached,
    so it correctly includes alignment-uncertain tokens that are absent from
    the stable-only evaluation artifact.  The full inventory is reconstructed
    solely from prompt-derived canonical phones; the stable subset must also
    match the adapted evaluation projection field for field.
    """

    patient_set = {str(value) for value in patients}
    expected_inventory: set[tuple[str, str, str, int, str]] = set()
    for manifest_row in manifest_rows:
        speaker = str(manifest_row.get("speaker_id", ""))
        if speaker not in patient_set:
            continue
        event = str(manifest_row.get("reading_event_id", ""))
        phones = manifest_row.get("canonical_phones")
        if not event or not isinstance(phones, list) or not phones:
            raise LLMContractError("primary5 manifest lacks canonical inference inventory")
        for phone_index, phone in enumerate(phones):
            target = str(phone)
            if target not in ARPABET_39:
                raise LLMContractError("primary5 manifest has an invalid canonical phone")
            key = (
                f"primary5__outer_{speaker}", speaker, event, phone_index, target
            )
            if key in expected_inventory:
                raise LLMContractError("primary5 manifest has duplicate canonical tokens")
            expected_inventory.add(key)

    source_by_key: dict[tuple[str, str, str, int, str], Mapping[str, object]] = {}
    for index, row in enumerate(rows, start=1):
        if set(row) != RUNTIME_SOURCE_KEYS:
            raise LLMContractError(
                f"runtime source row {index} must use the exact private key set"
            )
        key = _runtime_token_key(row)
        if key in source_by_key:
            raise LLMContractError(f"duplicate runtime source token: {key}")
        source_by_key[key] = row
    if set(source_by_key) != expected_inventory:
        raise LLMContractError(
            "runtime source does not cover the exact prompt-derived primary5 tokens"
        )

    stable_by_key: dict[tuple[str, str, str, int, str], Mapping[str, object]] = {}
    for row in prediction_rows:
        if row.get("cohort") != "primary5" or row.get("method") != ADAPTED_METHOD:
            continue
        projected = {key: row[key] for key in RUNTIME_SOURCE_KEYS}
        key = _runtime_token_key(projected)
        if key in stable_by_key:
            raise LLMContractError(f"duplicate adapted evaluation token: {key}")
        stable_by_key[key] = projected
    if not stable_by_key or not set(stable_by_key).issubset(source_by_key):
        raise LLMContractError(
            "runtime source is missing the stable adapted evaluation projection"
        )
    for key, expected in stable_by_key.items():
        if dict(source_by_key[key]) != dict(expected):
            raise LLMContractError(
                "runtime source disagrees with the stable adapted evaluation projection"
            )


def export_llm_stage(
    config: dict[str, Any] | None = None,
    *,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Export only privacy-minimized primary5 evidence using fold-local gates.

    Gold labels are intentionally unavailable to this stage.  The input is the
    private runtime-source projection written after outer-fold prediction, and
    each row is authorized only by its own outer fold's inner-OOF gate.
    """

    cfg = load_config() if config is None else config
    export_config = cfg.get("llm_export", {})
    if export_config.get("schema_version", "da-cf-gop.llm.v1") != "da-cf-gop.llm.v1":
        raise LLMContractError("frozen LLM export schema version changed")
    if export_config.get("patient_facing", False) is not False:
        raise LLMContractError("patient-facing export is forbidden in this study")
    error_threshold = float(export_config.get("error_threshold", 0.50))
    if not 0.0 <= error_threshold <= 1.0:
        raise LLMContractError("LLM error threshold must be a probability")
    precision_floor = float(export_config.get("precision_floor", 0.80))
    coverage_floor = float(export_config.get("coverage_floor", 0.10))
    margin_floor = float(export_config.get("alternative_margin_floor", 0.10))
    if precision_floor < 0.80 or coverage_floor < 0.10 or margin_floor < 0.10:
        raise LLMContractError("specific-output safety floors may not be weakened")
    # The evidence artifact is consumed only by a locally deployed LLM.  IDs
    # therefore need deterministic linkage, not a portable release secret.
    # Bind them to this schema and frozen configuration so repeated local runs
    # are byte-identical while raw identities remain outside the artifact.
    identifier_namespace = f"{SCHEMA_VERSION}:{cfg['_config_hash']}"

    layout = ArtifactLayout.from_path(cfg["paths"]["artifacts"])
    layout.create()
    cohort_dir = layout.evaluation / "primary5"
    source_path = cohort_dir / "runtime_source_predictions.jsonl.gz"
    predictions_path = cohort_dir / "phone_predictions.jsonl.gz"
    inner_oof_path = cohort_dir / "inner_oof_predictions.jsonl.gz"
    gates_path = cohort_dir / "specific_output_gates.json"
    manifest_path = layout.manifests / "primary5.jsonl.gz"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"primary5 runtime source is absent: {source_path}; "
            "run run-phone-loso --cohort primary5 first"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"primary5 manifest is absent: {manifest_path}; run build-manifests first"
        )
    for required in (predictions_path, inner_oof_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"primary5 export dependency is absent: {required}; "
                "run run-phone-loso --cohort primary5 first"
            )

    rows = read_jsonl(source_path)
    if not rows:
        raise LLMContractError("primary5 runtime source is empty")
    gates = _load_fold_gates(
        gates_path,
        precision_floor=precision_floor,
        coverage_floor=coverage_floor,
        margin_floor=margin_floor,
    )
    patients = tuple(str(value) for value in cfg.get("cohorts", {}).get("primary5", ()))
    if len(patients) != 5 or len(set(patients)) != 5:
        raise LLMContractError("frozen primary5 speaker inventory is invalid")
    expected_fold_speakers = {
        f"primary5__outer_{held_out}": tuple(
            speaker for speaker in patients if speaker != held_out
        )
        for held_out in patients
    }
    if set(gates) != set(expected_fold_speakers):
        raise LLMContractError("specific-output gates do not cover the exact primary5 folds")
    inner_rows = read_jsonl(inner_oof_path)
    recomputed_gates = _recompute_fold_gates(
        inner_rows,
        expected_fold_speakers=expected_fold_speakers,
        precision_floor=precision_floor,
        coverage_floor=coverage_floor,
        margin_floor=margin_floor,
        error_threshold=error_threshold,
        threshold_grid=tuple(export_config.get("threshold_grid", ())),
    )
    for fold in sorted(gates):
        if gates[fold].to_dict() != recomputed_gates[fold].to_dict():
            raise LLMContractError(
                f"specific-output gate {fold} cannot be recomputed from inner OOF"
            )

    prediction_rows = read_jsonl(predictions_path)
    manifest_rows = read_jsonl(manifest_path)
    _validate_runtime_source_inventory(
        rows, prediction_rows, manifest_rows, patients=patients
    )
    source_folds: set[str] = set()
    records: list[dict[str, object]] = []
    decisions: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        if set(row) != RUNTIME_SOURCE_KEYS:
            raise LLMContractError(
                f"runtime source row {index} must use the exact private key set"
            )
        if row.get("cohort") != "primary5":
            raise LLMContractError(f"runtime source row {index} is not primary5")
        fold = str(row.get("fold", ""))
        if not fold or fold not in gates:
            raise LLMContractError(
                f"runtime source row {index} has no matching inner-OOF gate: {fold!r}"
            )
        expected_speaker = fold.removeprefix("primary5__outer_")
        if str(row.get("speaker", "")) != expected_speaker:
            raise LLMContractError(
                f"runtime source row {index} speaker does not match outer fold"
            )
        source_folds.add(fold)
        record = build_llm_record(
            row,
            gates[fold],
            identifier_namespace=identifier_namespace,
            error_threshold=error_threshold,
        )
        records.append(record)
        decisions[str(record["decision"])] += 1

    if source_folds != set(gates):
        missing = sorted(set(gates) - source_folds)
        raise LLMContractError(
            f"runtime source does not cover every gated outer fold: {missing}"
        )
    record_ids = [str(record["record_id"]) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise LLMContractError("runtime source produced duplicate local phone identifiers")
    records.sort(
        key=lambda item: (str(item["utterance_record_id"]), int(item["phone_index"]))
    )

    destination = layout.runtime / "llm_phone_evidence.jsonl"
    output_sha256 = write_jsonl(destination, records)
    summary = write_stage_summary(
        layout.summaries / "export-llm.json",
        stage="export-llm",
        config_sha256=str(cfg["_config_hash"]),
        training_speakers=[],
        source_files={"frozen_config": str(cfg["_config_hash"])},
        models={},
        manifests={"primary5": sha256_file(manifest_path)},
        upstream_artifacts={
            "phone_predictions": sha256_file(predictions_path),
            "runtime_source_predictions": sha256_file(source_path),
            "inner_oof_predictions": sha256_file(inner_oof_path),
            "specific_output_gates": sha256_file(gates_path),
        },
        exclusions={},
        outputs={"llm_phone_evidence": output_sha256},
        details={
            "cohort": "primary5",
            "n_records": len(records),
            "n_outer_folds": len(source_folds),
            "decisions": dict(sorted(decisions.items())),
            "deployment": "local_offline",
            "identifier_scheme": LOCAL_IDENTIFIER_SCHEME,
            "identifier_namespace": "schema_version+frozen_config_sha256",
            "error_threshold": error_threshold,
            "patient_facing": False,
            "llm_invoked": False,
        },
    )
    if progress is not None:
        progress(f"exported {len(records)} safe phone records to {destination}")
    return summary


def _dispatch(arguments: argparse.Namespace) -> None:
    command = arguments.command
    if command == "verify":
        from .verify import verify_and_print

        verify_and_print(arguments.project_root)
        return

    if command == "run-parikh2025":
        from .parikh_experiment import run

        run(arguments.artifacts, cohort=arguments.cohort, device=arguments.device)
        return

    if command == "run-dual-graph":
        from .dual_experiment import run

        run(arguments.artifacts, cohort=arguments.cohort, config_path=arguments.dual_config,
            reuse_baselines=arguments.reuse_baselines)
        return

    cfg = load_config(arguments.config)
    if command == "build-manifests":
        from .stages import build_manifests_stage

        build_manifests_stage(cfg)
    elif command == "extract-logits":
        from .stages import extract_logits_stage

        extract_logits_stage(cfg, cohort=arguments.cohort, limit=arguments.limit)
    elif command == "run-phone-loso":
        from .experiment import run_phone_loso_stage

        run_phone_loso_stage(cfg, cohort=arguments.cohort)
    elif command == "run-severity":
        from .severity_scoring import run_severity_scoring_stage
        from .severity import run_severity_stage

        run_severity_scoring_stage(cfg)
        run_severity_stage(cfg)
    elif command == "export-llm":
        export_llm_stage(cfg, progress=print)
    else:  # pragma: no cover - argparse makes this unreachable.
        raise RuntimeError(f"unhandled command: {command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and execute one command, converting failures to exit status 1."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        _dispatch(arguments)
    except KeyboardInterrupt:
        parser.exit(130, "error: interrupted\n")
    except Exception as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
