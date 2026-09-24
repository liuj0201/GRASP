"""Matched, speaker-first evaluation for the secondary severity experiment.

This module intentionally treats the historical code9/code10 outputs as
immutable *data*.  It never imports their Python packages.  The audited
manifest is used only to recover the frozen reading-event identity
``(speaker, session, stem)`` and the paper severity label.

The comparison is system-level: the frozen methods use different acoustic
backends and different supervision.  Consequently, no result produced here
supports a causal claim about an individual DA-CF-GoP component.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from scipy import stats

from .artifacts import write_stage_summary
from .metrics import PHONE_VOCAB


SCHEMA_VERSION = "da-cf-gop.severity.v1"
NEW_METHOD = "DA-CF-GoP/adapted"
FROZEN_COMPARATOR_METHODS = (
    "MixGoP/XLS-R19",
    "UQ-GoP/maxlogit_prior",
    "CaGOP/full",
    "CTC-SF/ctc_sf_sd_norm",
)
EXPECTED_DYSARTHRIC_SPEAKERS = (
    "F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05",
)
SYSTEM_LEVEL_WARNING = (
    "System-level comparison only: methods differ in acoustic backend and/or "
    "supervision, so differences cannot be attributed causally to a DA-CF-GoP "
    "algorithm component."
)
SEVERITY_PHONE_SCORE_KEYS = frozenset(
    {
        "schema_version", "method", "cohort", "fold", "speaker", "event",
        "recording_id", "phone_index", "canonical_phone", "gop",
    }
)


class SeverityError(ValueError):
    """Raised when a severity artifact violates the frozen comparison contract."""


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SeverityError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise SeverityError(f"{name} must be finite")
    return result


def _deterministic_mean(values: Iterable[object], name: str) -> float:
    """Order-independent finite mean for reproducible JSON byte hashes."""

    checked = sorted(_finite(value, name) for value in values)
    if not checked:
        raise SeverityError(f"{name} has no values")
    return math.fsum(checked) / len(checked)


def canonical_event_id(speaker_id: str, session: str, stem: str) -> str:
    """Encode the frozen reading-event key without microphone information."""

    values = tuple(str(value).strip() for value in (speaker_id, session, stem))
    if any(not value for value in values):
        raise SeverityError("speaker_id, session, and stem must be non-empty")
    if any("|" in value for value in values):
        raise SeverityError("event-key components may not contain '|'")
    return "|".join(values)


def split_event_id(event_id: str, speaker_hint: str | None = None) -> tuple[str, str, str]:
    """Decode either the canonical pipe form or the new manifest underscore form."""

    raw = str(event_id).strip()
    if raw.count("|") == 2:
        speaker, session, stem = raw.split("|", 2)
        canonical_event_id(speaker, session, stem)
        return speaker, session, stem

    hint = str(speaker_hint or "").strip()
    prefix = f"{hint}_" if hint else ""
    remainder = raw[len(prefix):] if prefix and raw.startswith(prefix) else raw
    parts = remainder.split("_", 1)
    if hint and len(parts) == 2 and parts[0].lower().startswith("session"):
        return hint, parts[0], parts[1]
    raise SeverityError(
        "event must be 'speaker|session|stem', or the DA-CF manifest form "
        "'speaker_SessionN_stem' with a speaker field"
    )


def _open_jsonl(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL or JSONL.GZ and require an object on every non-empty line."""

    source = Path(path)
    rows: list[dict[str, Any]] = []
    with _open_jsonl(source) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise SeverityError(f"invalid JSON at {source}:{line_number}") from error
            if not isinstance(value, dict):
                raise SeverityError(f"JSONL row at {source}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise SeverityError(f"no rows found in {source}")
    return rows


def write_jsonl(
    rows: Iterable[Mapping[str, object]], path: str | Path
) -> Path:
    """Write deterministic JSONL, including deterministic gzip headers."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        json.dumps(
            dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        for row in rows
    )
    if destination.suffix.lower() == ".gz":
        with destination.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
                handle.write(payload)
    else:
        destination.write_bytes(payload)
    return destination


def _coerce_rows(source: str | Path | Iterable[Mapping[str, object]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        return read_jsonl(source)
    rows = [dict(row) for row in source]
    if not rows:
        raise SeverityError("input contains no rows")
    return rows


def validate_severity_phone_scores(
    rows: Sequence[Mapping[str, object]],
    *,
    audited_recordings: Sequence[Mapping[str, object]] | None = None,
    canonical_phones_by_recording: Mapping[str, Sequence[str]] | None = None,
    expected_recordings: int | None = 415,
    expected_events: int | None = 329,
    expected_speakers: Sequence[str] | None = EXPECTED_DYSARTHRIC_SPEAKERS,
) -> dict[str, int]:
    """Validate the exact dedicated severity inference schema and coverage."""

    if not rows:
        raise SeverityError("severity phone-score artifact is empty")
    audit_by_recording: dict[str, Mapping[str, object]] | None = None
    if audited_recordings is not None:
        audit_by_recording = {}
        for row in audited_recordings:
            recording = str(row.get("recording_id", ""))
            if not recording or recording in audit_by_recording:
                raise SeverityError("audited severity recordings are missing or duplicated")
            audit_by_recording[recording] = row
    allowed_speakers = (
        set(str(value) for value in expected_speakers)
        if expected_speakers is not None else None
    )
    indices: dict[str, set[int]] = defaultdict(set)
    observed_phones: dict[str, dict[int, str]] = defaultdict(dict)
    events: set[str] = set()
    speakers: set[str] = set()
    metadata: dict[str, tuple[str, str]] = {}
    for line, row in enumerate(rows, start=1):
        if set(row) != SEVERITY_PHONE_SCORE_KEYS:
            raise SeverityError(f"severity phone row {line} has a non-frozen key set")
        if row.get("schema_version") != "da-cf-gop.severity-phone-score.v1":
            raise SeverityError("severity phone-score schema changed")
        if row.get("method") != NEW_METHOD or row.get("cohort") != "severity8":
            raise SeverityError("severity phone-score method/cohort changed")
        speaker = str(row.get("speaker", ""))
        event = str(row.get("event", ""))
        recording = str(row.get("recording_id", ""))
        if not speaker or not event or not recording:
            raise SeverityError("severity phone row lacks identity fields")
        if allowed_speakers is not None and speaker not in allowed_speakers:
            raise SeverityError(f"unexpected severity speaker {speaker}")
        if row.get("fold") != f"severity8__outer_{speaker}":
            raise SeverityError("severity phone row is assigned to the wrong LOSO fold")
        event_speaker, _session, _stem = split_event_id(event, speaker)
        if event_speaker != speaker:
            raise SeverityError("severity event and speaker disagree")
        index = row.get("phone_index")
        if type(index) is not int or int(index) < 0:
            raise SeverityError("severity phone index is invalid")
        if int(index) in indices[recording]:
            raise SeverityError(f"duplicate severity phone token {(recording, index)}")
        indices[recording].add(int(index))
        canonical = str(row.get("canonical_phone", ""))
        if canonical not in PHONE_VOCAB:
            raise SeverityError("severity canonical phone is outside the model vocabulary")
        _finite(row.get("gop"), "severity GoP")
        observed_phones[recording][int(index)] = canonical
        current = (speaker, event)
        if recording in metadata and metadata[recording] != current:
            raise SeverityError("severity recording metadata changes between phone rows")
        metadata[recording] = current
        if audit_by_recording is not None:
            audit = audit_by_recording.get(recording)
            if audit is None:
                raise SeverityError(f"severity score has unknown recording {recording}")
            if speaker != str(audit.get("speaker_id")) or event != str(audit.get("event_id")):
                raise SeverityError("severity score disagrees with audited recording metadata")
        events.add(event)
        speakers.add(speaker)
    for recording, observed in indices.items():
        if observed != set(range(max(observed) + 1)):
            raise SeverityError(f"severity token indices are not contiguous for {recording}")
    if canonical_phones_by_recording is not None:
        if set(canonical_phones_by_recording) != set(indices):
            raise SeverityError("canonical prompt sequences lack exact severity recording coverage")
        for recording, sequence in canonical_phones_by_recording.items():
            expected = tuple(str(phone).upper() for phone in sequence)
            if not expected or any(phone not in PHONE_VOCAB for phone in expected):
                raise SeverityError(f"invalid prompt-derived phone sequence for {recording}")
            observed = observed_phones[recording]
            if set(observed) != set(range(len(expected))):
                raise SeverityError(f"severity token count disagrees with prompt G2P for {recording}")
            if any(observed[index] != phone for index, phone in enumerate(expected)):
                raise SeverityError(f"severity canonical phones disagree with prompt G2P for {recording}")
    if audit_by_recording is not None and set(indices) != set(audit_by_recording):
        raise SeverityError("severity scores lack exact audited recording coverage")
    if expected_recordings is not None and len(indices) != expected_recordings:
        raise SeverityError("severity recording count changed")
    if expected_events is not None and len(events) != expected_events:
        raise SeverityError("severity event count changed")
    if allowed_speakers is not None and speakers != allowed_speakers:
        raise SeverityError("severity speaker coverage changed")
    return {
        "n_phone_rows": len(rows), "n_recordings": len(indices),
        "n_events": len(events), "n_speakers": len(speakers),
    }


def canonical_phone_sequences_from_audited_prompts(
    audited_recordings: Sequence[Mapping[str, object]],
    *,
    g2p: Any | None = None,
) -> dict[str, tuple[str, ...]]:
    """Independently rebuild each audited recording's canonical prompt phones."""

    if not audited_recordings:
        raise SeverityError("audited severity recording list is empty")
    if g2p is None:
        from .phonology import texts_to_phones

        g2p = texts_to_phones
    prompts = list(dict.fromkeys(str(row.get("prompt", "")).strip() for row in audited_recordings))
    if any(not prompt for prompt in prompts):
        raise SeverityError("audited severity prompt is empty")
    mapped, unknown = g2p(prompts)
    if len(mapped) != len(prompts) or len(unknown) != len(prompts):
        raise SeverityError("severity G2P returned a different number of prompt rows")
    by_prompt: dict[str, tuple[str, ...]] = {}
    for prompt, sequence, unknown_phones in zip(prompts, mapped, unknown):
        phones = tuple(str(phone).upper() for phone in sequence)
        invalid = sorted(set(phones) - PHONE_VOCAB)
        if unknown_phones or not phones or invalid:
            raise SeverityError(
                f"severity prompt cannot be mapped to the frozen vocabulary: "
                f"unknown={list(unknown_phones)}, invalid={invalid}"
            )
        by_prompt[prompt] = phones
    result: dict[str, tuple[str, ...]] = {}
    for row in audited_recordings:
        recording = str(row.get("recording_id", ""))
        prompt = str(row.get("prompt", "")).strip()
        if not recording or recording in result:
            raise SeverityError("audited severity recording ids are missing or duplicated")
        result[recording] = by_prompt[prompt]
    return result


def build_audited_event_index(
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    expected_manifest_rows: int | None = 415,
    expected_events: int | None = 329,
    expected_speakers: int | None = 8,
) -> dict[str, Any]:
    """Index the audited dysarthric rows and collapse them to reading events.

    The original 415-row benchmark intentionally contains repeated reading
    events from different microphone recordings.  This experiment freezes the
    independent event key and gives each event one vote within a speaker.
    """

    selected: list[Mapping[str, object]] = []
    for row in manifest_rows:
        group = str(row.get("speaker_group", "")).strip().lower()
        severity_value = row.get("paper_severity", row.get("severity"))
        if group:
            if group != "dysarthric":
                continue
        elif severity_value is None or _finite(severity_value, "paper_severity") <= 0:
            continue
        selected.append(row)

    if expected_manifest_rows is not None and len(selected) != expected_manifest_rows:
        raise SeverityError(
            f"audited dysarthric manifest must contain {expected_manifest_rows} rows; "
            f"got {len(selected)}"
        )

    by_utterance: dict[str, dict[str, Any]] = {}
    events: dict[str, dict[str, Any]] = {}
    for row in selected:
        required = ("utterance_id", "speaker_id", "session", "stem")
        missing = [name for name in required if not str(row.get(name, "")).strip()]
        if missing:
            raise SeverityError(f"audited manifest row is missing fields: {missing}")
        utterance_id = str(row["utterance_id"])
        if utterance_id in by_utterance:
            raise SeverityError(f"duplicate audited utterance_id: {utterance_id}")
        speaker = str(row["speaker_id"])
        session = str(row["session"])
        stem = str(row["stem"])
        severity = _finite(
            row.get("paper_severity", row.get("severity")), "paper_severity"
        )
        if severity <= 0:
            raise SeverityError("the severity experiment accepts dysarthric speakers only")
        event_id = canonical_event_id(speaker, session, stem)
        metadata = {
            "event_id": event_id,
            "speaker_id": speaker,
            "session": session,
            "stem": stem,
            "severity": severity,
            "audio_key": (
                str(Path(str(row["wav_path"])).resolve()).casefold()
                if str(row.get("wav_path", "")).strip()
                else f"utterance:{utterance_id}"
            ),
        }
        by_utterance[utterance_id] = metadata
        existing = events.setdefault(
            event_id,
            {
                **{name: metadata[name] for name in (
                    "event_id", "speaker_id", "session", "stem", "severity"
                )},
                "source_utterance_ids": [],
                "source_audio_keys": [],
            },
        )
        if any(existing[name] != metadata[name] for name in (
            "event_id", "speaker_id", "session", "stem", "severity"
        )):
            raise SeverityError(f"inconsistent metadata for reading event {event_id}")
        existing["source_utterance_ids"].append(utterance_id)
        existing["source_audio_keys"].append(metadata["audio_key"])

    for event in events.values():
        event["source_utterance_ids"] = sorted(event["source_utterance_ids"])
        event["source_audio_keys"] = sorted(set(event["source_audio_keys"]))

    speakers = sorted({str(value["speaker_id"]) for value in events.values()})
    if expected_events is not None and len(events) != expected_events:
        raise SeverityError(
            f"audited manifest must contain {expected_events} reading events; got {len(events)}"
        )
    if expected_speakers is not None and len(speakers) != expected_speakers:
        raise SeverityError(
            f"audited manifest must contain {expected_speakers} dysarthric speakers; "
            f"got {len(speakers)}"
        )
    if expected_speakers == 8 and tuple(speakers) != EXPECTED_DYSARTHRIC_SPEAKERS:
        raise SeverityError(f"unexpected frozen speaker set: {speakers}")
    return {
        "by_utterance": by_utterance,
        "events": events,
        "n_manifest_rows": len(selected),
        "n_events": len(events),
        "speakers": speakers,
    }


def build_severity_inference_manifest(
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    logits_cache: str | Path | None = None,
    expected_manifest_rows: int | None = 415,
    expected_events: int | None = 329,
    expected_speakers: int | None = 8,
    require_audio_files: bool = True,
) -> list[dict[str, Any]]:
    """Create the label-free input for dedicated eight-speaker inference.

    One row is emitted for each of the 415 audited recording identifiers.
    Severity, PHN, TextGrid, and segment boundaries are deliberately absent.
    Repeated physical paths are intentionally retained because the frozen
    comparator artifacts are keyed by these exact 415 identifiers; they are
    averaged inside the 329 reading events before speaker aggregation.
    """

    index = build_audited_event_index(
        manifest_rows,
        expected_manifest_rows=expected_manifest_rows,
        expected_events=expected_events,
        expected_speakers=expected_speakers,
    )
    by_utterance = index["by_utterance"]
    selected = sorted(
        (row for row in manifest_rows if str(row.get("utterance_id", "")) in by_utterance),
        key=lambda row: str(row["utterance_id"]),
    )
    cache_root = Path(logits_cache).resolve() if logits_cache is not None else None
    output: list[dict[str, Any]] = []
    for source in selected:
        utterance_id = str(source["utterance_id"])
        event_id = str(by_utterance[utterance_id]["event_id"])
        event = index["events"][event_id]
        prompt = str(source.get("prompt", "")).strip()
        if not prompt:
            raise SeverityError(f"missing prompt for {utterance_id}")
        wav_path = Path(str(source.get("wav_path", "")))
        if not str(source.get("wav_path", "")).strip():
            raise SeverityError("severity inference source has no wav_path")
        if require_audio_files and not wav_path.is_file():
            raise SeverityError(f"severity inference audio is missing: {wav_path}")
        record = {
            "schema_version": "da-cf-gop.severity-inference-manifest.v1",
            "recording_id": utterance_id,
            "event_id": event_id,
            "speaker_id": str(event["speaker_id"]),
            "session": str(event["session"]),
            "stem": str(event["stem"]),
            "wav_path": str(wav_path.resolve()),
            "prompt": prompt,
            "prompt_id": str(source.get("prompt_id", "")),
            "audio_microphone": str(source.get("microphone", "unknown")),
            "canonical_source": "frozen_local_espeak_prompt_g2p",
            "phn_or_textgrid_permitted": False,
            "severity_available_to_scorer": False,
        }
        if cache_root is not None:
            logits_path = cache_root / f"{utterance_id}.npz"
            if require_audio_files and not logits_path.is_file():
                raise SeverityError(f"audited official logits are missing: {logits_path}")
            record["official_logits_npz"] = str(logits_path)
        output.append(record)
    event_ids = {str(row["event_id"]) for row in output}
    if event_ids != set(index["events"]):
        raise SeverityError("severity inference manifest lost an audited reading event")
    return output


def build_audited_severity_events(
    audited_manifest: str | Path | Iterable[Mapping[str, object]],
    *,
    logits_cache: str | Path,
    expected_manifest_rows: int | None = 415,
    expected_events: int | None = 329,
    expected_speakers: int | None = 8,
    require_files: bool = True,
) -> dict[str, Any]:
    """Expose the exact audited inference inputs and their 329 event groups."""

    manifest_rows = _coerce_rows(audited_manifest)
    recordings = build_severity_inference_manifest(
        manifest_rows,
        logits_cache=logits_cache,
        expected_manifest_rows=expected_manifest_rows,
        expected_events=expected_events,
        expected_speakers=expected_speakers,
        require_audio_files=require_files,
    )
    grouped: dict[str, list[str]] = defaultdict(list)
    event_metadata: dict[str, tuple[str, str, str]] = {}
    for row in recordings:
        event_id = str(row["event_id"])
        grouped[event_id].append(str(row["recording_id"]))
        event_metadata[event_id] = (
            str(row["speaker_id"]), str(row["session"]), str(row["stem"])
        )
    events = [
        {
            "event_id": event_id,
            "speaker_id": event_metadata[event_id][0],
            "session": event_metadata[event_id][1],
            "stem": event_metadata[event_id][2],
            "recording_ids": sorted(grouped[event_id]),
            "n_recordings": len(grouped[event_id]),
        }
        for event_id in sorted(grouped)
    ]
    return {
        "schema_version": "da-cf-gop.severity-audited-events.v1",
        "recording_policy": (
            "score all audited recordings; arithmetic mean within reading event; "
            "arithmetic mean within speaker"
        ),
        "n_recordings": len(recordings),
        "n_events": len(events),
        "n_speakers": len({row["speaker_id"] for row in recordings}),
        "recordings": recordings,
        "events": events,
    }


def write_severity_inference_manifest(
    rows: Sequence[Mapping[str, object]], path: str | Path
) -> Path:
    """Write the dedicated inference manifest in deterministic JSONL form."""

    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row["event_id"]), str(row["recording_id"])),
    )
    return write_jsonl(ordered, path)


def load_frozen_score_sources(
    sources: Mapping[str, str | Path],
) -> list[dict[str, Any]]:
    """Normalize the four predeclared code9 raw result files as immutable data."""

    specifications = {
        "mixgop": ("mixgop", "clarity", "MixGoP/XLS-R19"),
        "uq_gop": ("UQ-GoP/maxlogit_prior", "score", "UQ-GoP/maxlogit_prior"),
        "cagop": ("CaGOP/full", "score", "CaGOP/full"),
        "ctc_sf": ("CTC-SF/ctc_sf_sd_norm", "score", "CTC-SF/ctc_sf_sd_norm"),
    }
    if set(sources) != set(specifications):
        raise SeverityError(
            f"frozen score sources must be exactly {sorted(specifications)}"
        )
    output: list[dict[str, Any]] = []
    for source_name in sorted(specifications):
        source_method, score_field, normalized_method = specifications[source_name]
        selected = 0
        for row in read_jsonl(sources[source_name]):
            if str(row.get("method", "")) != source_method:
                continue
            output.append({
                "utterance_id": str(row.get("utterance_id", "")),
                "method": normalized_method,
                "score": _finite(row.get(score_field), f"{normalized_method} score"),
                "score_direction": "higher_is_better",
            })
            selected += 1
        if selected == 0:
            raise SeverityError(f"no {source_method} rows found in {sources[source_name]}")
    return output


def normalize_frozen_comparators(
    score_rows: Sequence[Mapping[str, object]],
    audited_index: Mapping[str, object],
    *,
    methods: Sequence[str] = FROZEN_COMPARATOR_METHODS,
) -> dict[str, list[dict[str, Any]]]:
    """Load frozen comparator rows and average repeated recordings per event."""

    by_utterance = audited_index["by_utterance"]
    events = audited_index["events"]
    if not isinstance(by_utterance, Mapping) or not isinstance(events, Mapping):
        raise SeverityError("invalid audited event index")
    expected_utterances = set(str(value) for value in by_utterance)
    expected_events = set(str(value) for value in events)
    selected_methods = tuple(str(value) for value in methods)
    if not selected_methods or len(set(selected_methods)) != len(selected_methods):
        raise SeverityError("comparator method names must be unique and non-empty")

    raw: dict[str, dict[str, float]] = {method: {} for method in selected_methods}
    for row in score_rows:
        method = str(row.get("method", ""))
        if method not in raw:
            continue
        utterance_id = str(row.get("utterance_id", ""))
        if utterance_id not in expected_utterances:
            # Raw frozen files also contain the healthy reference rows.  They
            # are not part of the eight-speaker severity evaluation.
            continue
        if utterance_id in raw[method]:
            raise SeverityError(f"duplicate ({method}, {utterance_id}) score")
        direction = str(row.get("score_direction", row.get("direction", "higher_is_better")))
        if direction != "higher_is_better":
            raise SeverityError(f"{method} has an unexpected score direction: {direction}")
        raw[method][utterance_id] = _finite(row.get("score"), f"{method} score")

    output: dict[str, list[dict[str, Any]]] = {}
    for method in selected_methods:
        observed = set(raw[method])
        if observed != expected_utterances:
            missing = sorted(expected_utterances - observed)[:5]
            extra = sorted(observed - expected_utterances)[:5]
            raise SeverityError(
                f"{method} does not have exact audited-row coverage; "
                f"missing={missing}, extra={extra}"
            )
        grouped: dict[str, list[float]] = defaultdict(list)
        for utterance_id, score in raw[method].items():
            metadata = by_utterance[utterance_id]
            grouped[str(metadata["event_id"])].append(score)
        if set(grouped) != expected_events:
            raise SeverityError(f"{method} does not have exact audited-event coverage")
        output[method] = [
            {
                "method": method,
                "event_id": event_id,
                "speaker_id": str(events[event_id]["speaker_id"]),
                "session": str(events[event_id]["session"]),
                "stem": str(events[event_id]["stem"]),
                "severity": float(events[event_id]["severity"]),
                "score": _deterministic_mean(grouped[event_id], f"{method} event score"),
                "n_source_recordings": len(grouped[event_id]),
            }
            for event_id in sorted(grouped)
        ]
    return output


def event_scores_from_phone_rows(
    phone_rows: Sequence[Mapping[str, object]],
    *,
    method: str = NEW_METHOD,
    cohort: str | None = None,
    score_field: str = "gop",
    stable_only: bool = True,
) -> list[dict[str, Any]]:
    """Calculate one arithmetic mean :math:`G_i` per reading event.

    If a ``stable`` field is present, the default uses only stable canonical
    tokens.  Duplicate phone positions are rejected so that accidentally
    concatenating two runs cannot silently change an utterance mean.
    """

    recording_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    metadata: dict[str, tuple[str, str, str]] = {}
    seen_tokens: set[tuple[str, str, int]] = set()
    selected = 0
    for row in phone_rows:
        if "method" in row and str(row["method"]) != method:
            continue
        if cohort is not None and str(row.get("cohort", "")) != cohort:
            continue
        if stable_only and "stable" in row and row["stable"] is not True:
            continue
        speaker = str(row.get("speaker_id", row.get("speaker", ""))).strip()
        event_raw = str(
            row.get("event_id", row.get("reading_event_id", row.get("event", "")))
        ).strip()
        if not speaker or not event_raw:
            raise SeverityError("phone row needs speaker and reading-event identity")
        event_speaker, session, stem = split_event_id(event_raw, speaker)
        if event_speaker != speaker:
            raise SeverityError(f"speaker disagrees with event identity: {event_raw}")
        event_id = canonical_event_id(speaker, session, stem)
        recording_id = str(
            row.get("recording_id", row.get("utterance_id", event_id))
        ).strip()
        if not recording_id:
            raise SeverityError("recording_id must be non-empty")
        if "phone_index" in row:
            if type(row["phone_index"]) is not int or int(row["phone_index"]) < 0:
                raise SeverityError("phone_index must be a non-negative integer")
            token_key = (event_id, recording_id, int(row["phone_index"]))
            if token_key in seen_tokens:
                raise SeverityError(f"duplicate phone score: {token_key}")
            seen_tokens.add(token_key)
        recording_scores[(event_id, recording_id)].append(
            _finite(row.get(score_field), score_field)
        )
        metadata[event_id] = (speaker, session, stem)
        selected += 1
    if selected == 0:
        raise SeverityError(f"no phone rows selected for method {method!r}")
    by_event: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for (event_id, recording_id), scores in sorted(recording_scores.items()):
        by_event[event_id].append((
            recording_id,
            _deterministic_mean(scores, "recording phone GoP"),
            len(scores),
        ))
    return [
        {
            "method": method,
            "event_id": event_id,
            "speaker_id": metadata[event_id][0],
            "session": metadata[event_id][1],
            "stem": metadata[event_id][2],
            "score": _deterministic_mean(
                (recording[1] for recording in by_event[event_id]), "recording mean GoP"
            ),
            "n_recordings": len(by_event[event_id]),
            "recording_ids": sorted(recording[0] for recording in by_event[event_id]),
            "n_phones": sum(recording[2] for recording in by_event[event_id]),
        }
        for event_id in sorted(by_event)
    ]


def normalize_new_event_scores(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str = NEW_METHOD,
    score_field: str = "score",
) -> list[dict[str, Any]]:
    """Validate already-aggregated new-method event scores."""

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if "method" in row and str(row["method"]) != method:
            continue
        speaker = str(row.get("speaker_id", row.get("speaker", ""))).strip()
        event_raw = str(
            row.get("event_id", row.get("reading_event_id", row.get("event", "")))
        ).strip()
        event_speaker, session, stem = split_event_id(event_raw, speaker)
        if event_speaker != speaker:
            raise SeverityError(f"speaker disagrees with event identity: {event_raw}")
        event_id = canonical_event_id(speaker, session, stem)
        if event_id in seen:
            raise SeverityError(f"duplicate new-method event score: {event_id}")
        seen.add(event_id)
        output.append({
            "method": method,
            "event_id": event_id,
            "speaker_id": speaker,
            "session": session,
            "stem": stem,
            "score": _finite(row.get(score_field), score_field),
            "recording_ids": sorted(str(value) for value in row.get("recording_ids", ())),
        })
    if not output:
        raise SeverityError(f"no event scores selected for method {method!r}")
    return sorted(output, key=lambda value: value["event_id"])


def match_new_scores_to_audit(
    rows: Sequence[Mapping[str, object]],
    audited_index: Mapping[str, object],
    *,
    method: str = NEW_METHOD,
) -> list[dict[str, Any]]:
    """Attach evaluation-only labels after enforcing exact event coverage."""

    events = audited_index["events"]
    if not isinstance(events, Mapping):
        raise SeverityError("invalid audited event index")
    indexed: dict[str, Mapping[str, object]] = {}
    for row in rows:
        event_id = str(row.get("event_id", ""))
        if not event_id:
            raise SeverityError("new-method event score has no event_id")
        if event_id in indexed:
            raise SeverityError(f"duplicate new-method event score: {event_id}")
        indexed[event_id] = row
    expected = set(str(value) for value in events)
    observed = set(indexed)
    if observed != expected:
        raise SeverityError(
            "new method does not have exact audited-event coverage; "
            f"missing={sorted(expected - observed)[:5]}, "
            f"extra={sorted(observed - expected)[:5]}"
        )
    output: list[dict[str, Any]] = []
    for event_id in sorted(expected):
        audit = events[event_id]
        row = indexed[event_id]
        if str(row.get("speaker_id", "")) != str(audit["speaker_id"]):
            raise SeverityError(f"speaker mismatch for {event_id}")
        expected_recordings = sorted(str(value) for value in audit["source_utterance_ids"])
        observed_recordings = sorted(str(value) for value in row.get("recording_ids", ()))
        if observed_recordings != expected_recordings:
            raise SeverityError(
                f"new method does not have exact audited-recording coverage for {event_id}; "
                f"expected={expected_recordings}, observed={observed_recordings}"
            )
        output.append({
            "method": method,
            "event_id": event_id,
            "speaker_id": str(audit["speaker_id"]),
            "session": str(audit["session"]),
            "stem": str(audit["stem"]),
            "severity": float(audit["severity"]),
            "score": _finite(row.get("score"), "new-method event score"),
            "n_source_recordings": len(observed_recordings),
            "n_phones": int(row.get("n_phones", 0)),
        })
    return output


def speaker_first_scores(
    event_rows: Sequence[Mapping[str, object]], method: str
) -> list[dict[str, Any]]:
    """Average event scores within speaker, giving every speaker equal weight."""

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    seen_events: set[str] = set()
    for row in event_rows:
        event_id = str(row.get("event_id", ""))
        if not event_id or event_id in seen_events:
            raise SeverityError(f"missing or duplicate event_id: {event_id}")
        seen_events.add(event_id)
        grouped[str(row.get("speaker_id", ""))].append(row)
    output: list[dict[str, Any]] = []
    for speaker in sorted(grouped):
        if not speaker:
            raise SeverityError("empty speaker_id")
        rows = grouped[speaker]
        severities = {_finite(row.get("severity"), "severity") for row in rows}
        if len(severities) != 1:
            raise SeverityError(f"inconsistent severity for speaker {speaker}")
        output.append({
            "method": method,
            "speaker_id": speaker,
            "severity": severities.pop(),
            "score": _deterministic_mean(
                (row.get("score") for row in rows), f"score[{speaker}]"
            ),
            "n_events": len(rows),
        })
    if not output:
        raise SeverityError("speaker aggregation received no events")
    return output


def severity_correlations(
    severity: Sequence[float], scores: Sequence[float]
) -> dict[str, float]:
    """Compute the frozen signed and absolute speaker-level correlations."""

    labels = np.asarray(severity, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or values.shape != labels.shape or len(labels) < 3:
        raise SeverityError("severity correlations require at least three paired speakers")
    if not np.isfinite(labels).all() or not np.isfinite(values).all():
        raise SeverityError("severity and score arrays must be finite")
    if len(np.unique(labels)) < 2 or len(np.unique(values)) < 2:
        raise SeverityError("severity correlations are undefined for a constant input")
    tau = float(stats.kendalltau(labels, values, variant="b").statistic)
    rho = float(stats.spearmanr(labels, values).statistic)
    pearson = float(stats.pearsonr(labels, values).statistic)
    if not all(math.isfinite(value) for value in (tau, rho, pearson)):
        raise SeverityError("a severity correlation is undefined")
    return {
        "kendall_tau_b": tau,
        "abs_kendall_tau_b": abs(tau),
        "spearman_rho": rho,
        "pearson_r": pearson,
    }


def _paired_speaker_arrays(
    rows_a: Sequence[Mapping[str, object]],
    rows_b: Sequence[Mapping[str, object]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    def index(rows: Sequence[Mapping[str, object]], label: str) -> dict[str, Mapping[str, object]]:
        result: dict[str, Mapping[str, object]] = {}
        for row in rows:
            speaker = str(row.get("speaker_id", ""))
            if not speaker or speaker in result:
                raise SeverityError(f"{label} has an empty or duplicate speaker: {speaker}")
            result[speaker] = row
        return result

    a = index(rows_a, "method A")
    b = index(rows_b, "method B")
    if set(a) != set(b) or not a:
        raise SeverityError("paired methods need the same non-empty speaker set")
    speakers = sorted(a)
    labels = np.asarray([_finite(a[s]["severity"], "severity") for s in speakers])
    labels_b = np.asarray([_finite(b[s]["severity"], "severity") for s in speakers])
    if not np.array_equal(labels, labels_b):
        raise SeverityError("paired methods disagree on speaker severity")
    score_a = np.asarray([_finite(a[s]["score"], "score A") for s in speakers])
    score_b = np.asarray([_finite(b[s]["score"], "score B") for s in speakers])
    return speakers, labels, score_a, score_b


def paired_speaker_bootstrap(
    rows_a: Sequence[Mapping[str, object]],
    rows_b: Sequence[Mapping[str, object]],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20260829,
) -> dict[str, Any]:
    """Severity-stratified paired speaker bootstrap of metric(A)-metric(B)."""

    if n_bootstrap < 1:
        raise SeverityError("n_bootstrap must be positive")
    speakers, labels, score_a, score_b = _paired_speaker_arrays(rows_a, rows_b)
    metric_a = severity_correlations(labels, score_a)
    metric_b = severity_correlations(labels, score_b)
    point = {name: metric_a[name] - metric_b[name] for name in metric_a}
    strata = [np.flatnonzero(labels == value) for value in sorted(np.unique(labels))]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(n_bootstrap):
        draw = np.concatenate([
            members[rng.integers(0, len(members), size=len(members))]
            for members in strata
        ])
        try:
            draw_a = severity_correlations(labels[draw], score_a[draw])
            draw_b = severity_correlations(labels[draw], score_b[draw])
        except SeverityError:
            continue
        for name in point:
            value = draw_a[name] - draw_b[name]
            if math.isfinite(value):
                samples[name].append(value)
    metrics: dict[str, Any] = {}
    for name, difference in point.items():
        values = samples[name]
        if not values:
            raise SeverityError(f"all bootstrap replicates were undefined for {name}")
        low, high = np.quantile(np.asarray(values), [0.025, 0.975])
        metrics[name] = {
            "difference_a_minus_b": float(difference),
            "ci_low": float(low),
            "ci_high": float(high),
            "valid_replicates": len(values),
        }
    return {
        "unit": "paired_speaker",
        "stratified_by": "paper_severity",
        "n_speakers": len(speakers),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "metrics": metrics,
    }


def exact_paired_sign_flip_test(
    rows_a: Sequence[Mapping[str, object]],
    rows_b: Sequence[Mapping[str, object]],
    *,
    expected_speakers: int | None = 8,
) -> dict[str, Any]:
    """Exact within-speaker method swap (paired sign-flip) test.

    For every sign pattern, the two method scores are swapped within selected
    speakers and the correlation difference is recomputed.  With the frozen
    eight-speaker cohort this enumerates all ``2**8 == 256`` assignments.
    """

    speakers, labels, score_a, score_b = _paired_speaker_arrays(rows_a, rows_b)
    if expected_speakers is not None and len(speakers) != expected_speakers:
        raise SeverityError(
            f"exact test requires {expected_speakers} speakers; got {len(speakers)}"
        )
    if len(speakers) > 20:
        raise SeverityError("exact enumeration is limited to 20 speakers")
    observed_a = severity_correlations(labels, score_a)
    observed_b = severity_correlations(labels, score_b)
    metric_names = ("kendall_tau_b", "abs_kendall_tau_b")
    observed = {name: observed_a[name] - observed_b[name] for name in metric_names}
    null: dict[str, list[float]] = {name: [] for name in metric_names}
    midpoint = (score_a + score_b) / 2.0
    half_difference = (score_a - score_b) / 2.0
    for signs in itertools.product((-1.0, 1.0), repeat=len(speakers)):
        signed = half_difference * np.asarray(signs, dtype=np.float64)
        perm_a = severity_correlations(labels, midpoint + signed)
        perm_b = severity_correlations(labels, midpoint - signed)
        for name in metric_names:
            null[name].append(perm_a[name] - perm_b[name])
    tests = {}
    for name in metric_names:
        distribution = np.asarray(null[name], dtype=np.float64)
        p_value = float(np.mean(
            np.abs(distribution) >= abs(observed[name]) - 1e-12
        ))
        tests[name] = {
            "difference_a_minus_b": float(observed[name]),
            "p_two_sided": p_value,
        }
    return {
        "unit": "paired_speaker",
        "operation": "within_speaker_method_swap_equivalent_to_paired_sign_flip",
        "n_speakers": len(speakers),
        "n_permutations": 2 ** len(speakers),
        "tests": tests,
    }


def evaluate_severity_systems(
    new_event_rows: Sequence[Mapping[str, object]],
    frozen_event_rows: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    n_audited_manifest_rows: int,
    n_audited_events: int,
    new_method: str = NEW_METHOD,
    n_bootstrap: int = 10_000,
    seed: int = 20260829,
    expected_speakers: int | None = 8,
) -> dict[str, Any]:
    """Evaluate matched systems with event-first, then speaker-first weighting."""

    method_events: dict[str, Sequence[Mapping[str, object]]] = {
        new_method: new_event_rows,
        **{str(method): rows for method, rows in frozen_event_rows.items()},
    }
    if len(method_events) != 1 + len(frozen_event_rows):
        raise SeverityError("new method name collides with a frozen comparator")
    reference_events = {str(row["event_id"]) for row in new_event_rows}
    if len(reference_events) != n_audited_events:
        raise SeverityError("new-method event count disagrees with audited event count")
    for method, rows in method_events.items():
        event_ids = [str(row.get("event_id", "")) for row in rows]
        if len(event_ids) != len(set(event_ids)):
            raise SeverityError(f"{method} contains duplicate reading events")
        if set(event_ids) != reference_events:
            raise SeverityError(f"{method} does not have exact matched event coverage")

    speaker_rows = {
        method: speaker_first_scores(rows, method)
        for method, rows in sorted(method_events.items())
    }
    expected_set = set(str(row["speaker_id"]) for row in speaker_rows[new_method])
    if expected_speakers is not None and len(expected_set) != expected_speakers:
        raise SeverityError(
            f"severity experiment requires {expected_speakers} speakers; got {len(expected_set)}"
        )
    for method, rows in speaker_rows.items():
        if {str(row["speaker_id"]) for row in rows} != expected_set:
            raise SeverityError(f"{method} does not have exact matched speaker coverage")

    methods: dict[str, Any] = {}
    for method in sorted(method_events):
        rows = speaker_rows[method]
        methods[method] = {
            "coverage": {
                "n_events": len(method_events[method]),
                "n_audited_events": n_audited_events,
                "event_coverage": len(method_events[method]) / n_audited_events,
                "n_speakers": len(rows),
                "per_speaker_events": {
                    str(row["speaker_id"]): int(row["n_events"]) for row in rows
                },
            },
            "correlations": severity_correlations(
                [float(row["severity"]) for row in rows],
                [float(row["score"]) for row in rows],
            ),
            "speaker_scores": rows,
        }

    comparisons: dict[str, Any] = {}
    for comparator in sorted(frozen_event_rows):
        comparisons[comparator] = {
            "method_a": new_method,
            "method_b": comparator,
            "paired_bootstrap": paired_speaker_bootstrap(
                speaker_rows[new_method], speaker_rows[comparator],
                n_bootstrap=n_bootstrap, seed=seed,
            ),
            "exact_paired_sign_flip": exact_paired_sign_flip_test(
                speaker_rows[new_method], speaker_rows[comparator],
                expected_speakers=expected_speakers,
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "secondary_severity",
        "comparison_level": "system_level_cross_backend",
        "component_causal_claims_permitted": False,
        "warning": SYSTEM_LEVEL_WARNING,
        "severity_used_for_model_training": False,
        "direction_policy": (
            "raw higher-is-better scores; signed correlations are not flipped or "
            "selected using severity labels"
        ),
        "aggregation": {
            "event_key": ["speaker_id", "session", "stem"],
            "within_event": "arithmetic mean over repeated audited recordings",
            "within_speaker": "arithmetic mean over unique reading events",
            "between_speakers": "equal speaker weight",
        },
        "audit_scope": {
            "n_manifest_rows": n_audited_manifest_rows,
            "n_unique_reading_events": n_audited_events,
            "n_speakers": len(expected_set),
            "speaker_ids": sorted(expected_set),
            "strict_matched_coverage": True,
        },
        "resampling": {
            "bootstrap_replicates": n_bootstrap,
            "bootstrap_seed": seed,
            "bootstrap_unit": "paired_speaker",
            "exact_test_assignments": 2 ** len(expected_set),
        },
        "methods": methods,
        "paired_comparisons_vs_new_method": comparisons,
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_severity_outputs(
    report: Mapping[str, object],
    json_path: str | Path,
    csv_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Write deterministic JSON and an optional one-row-per-method CSV."""

    destination = Path(json_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    destination.write_text(payload, encoding="utf-8", newline="\n")
    csv_destination: Path | None = None
    if csv_path is not None:
        csv_destination = Path(csv_path)
        csv_destination.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            "method", "n_events", "n_speakers", "event_coverage",
            "kendall_tau_b", "abs_kendall_tau_b", "spearman_rho", "pearson_r",
        )
        with csv_destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            methods = report.get("methods")
            if not isinstance(methods, Mapping):
                raise SeverityError("severity report has no methods mapping")
            for method in sorted(methods):
                result = methods[method]
                coverage = result["coverage"]
                correlations = result["correlations"]
                writer.writerow({
                    "method": method,
                    "n_events": coverage["n_events"],
                    "n_speakers": coverage["n_speakers"],
                    "event_coverage": coverage["event_coverage"],
                    **correlations,
                })
    return destination, csv_destination


def run_severity_experiment(
    new_scores: str | Path | Iterable[Mapping[str, object]],
    audited_manifest: str | Path | Iterable[Mapping[str, object]],
    frozen_scores: str | Path | Iterable[Mapping[str, object]],
    *,
    new_scores_are_phone_level: bool = True,
    new_method: str = NEW_METHOD,
    cohort: str | None = None,
    score_field: str | None = None,
    comparator_methods: Sequence[str] = FROZEN_COMPARATOR_METHODS,
    n_bootstrap: int = 10_000,
    seed: int = 20260829,
    expected_manifest_rows: int | None = 415,
    expected_events: int | None = 329,
    expected_speakers: int | None = 8,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    """End-to-end deterministic API used by the ``run-severity`` CLI stage."""

    manifest_rows = _coerce_rows(audited_manifest)
    index = build_audited_event_index(
        manifest_rows,
        expected_manifest_rows=expected_manifest_rows,
        expected_events=expected_events,
        expected_speakers=expected_speakers,
    )
    raw_new = _coerce_rows(new_scores)
    if new_scores_are_phone_level:
        new_events = event_scores_from_phone_rows(
            raw_new,
            method=new_method,
            cohort=cohort,
            score_field=score_field or "gop",
        )
    else:
        new_events = normalize_new_event_scores(
            raw_new, method=new_method, score_field=score_field or "score"
        )
    matched_new = match_new_scores_to_audit(new_events, index, method=new_method)
    frozen_events = normalize_frozen_comparators(
        _coerce_rows(frozen_scores), index, methods=comparator_methods
    )
    report = evaluate_severity_systems(
        matched_new,
        frozen_events,
        n_audited_manifest_rows=int(index["n_manifest_rows"]),
        n_audited_events=int(index["n_events"]),
        new_method=new_method,
        n_bootstrap=n_bootstrap,
        seed=seed,
        expected_speakers=expected_speakers,
    )
    provenance: dict[str, Any] = {}
    for name, source in (
        ("new_scores", new_scores),
        ("audited_manifest", audited_manifest),
        ("frozen_scores", frozen_scores),
    ):
        if isinstance(source, (str, Path)):
            provenance[name] = {
                "filename": Path(source).name,
                "sha256": _sha256(source),
            }
    report["input_provenance"] = provenance
    if output_json is not None:
        write_severity_outputs(report, output_json, output_csv)
    elif output_csv is not None:
        raise SeverityError("output_csv requires output_json")
    return report


def run_severity_stage(
    config: Mapping[str, object],
    *,
    scorer: Any | None = None,
) -> dict[str, Any]:
    """Run the frozen secondary stage or fail with the precise missing input.

    ``scorer`` is an optional provider callable receiving the label-free list
    of 415 recording dictionaries returned by
    :func:`build_audited_severity_events`.  It must return phone-score rows (or
    a JSONL path) using ``recording_id`` values from that list.  Without a
    provider, the stage consumes
    ``artifacts/evaluation/severity_phone_scores.jsonl.gz``.
    """

    severity = config.get("severity")
    paths = config.get("paths")
    if not isinstance(severity, Mapping) or not isinstance(paths, Mapping):
        raise SeverityError("resolved config must contain paths and severity mappings")
    artifacts = Path(str(paths["artifacts"]))
    manifest_path = Path(str(severity["audited_manifest"]))
    logits_cache = Path(str(severity["audited_logits_cache"]))
    frozen_sources = severity.get("frozen_scores")
    if not isinstance(frozen_sources, Mapping):
        raise SeverityError("severity.frozen_scores must be a mapping")
    expected_recordings = int(severity["audited_recordings"])
    expected_events = int(severity["unique_reading_events"])
    expected_speakers = len(tuple(severity["dysarthric_speakers"]))
    n_bootstrap = int(
        config.get("evaluation", {}).get("bootstrap_replicates", 10_000)  # type: ignore[union-attr]
    )
    seed = int(config.get("seed", 20260829))

    audited = build_audited_severity_events(
        manifest_path,
        logits_cache=logits_cache,
        expected_manifest_rows=expected_recordings,
        expected_events=expected_events,
        expected_speakers=expected_speakers,
        require_files=True,
    )
    inference_manifest = artifacts / "manifests" / "severity_audited_recordings.jsonl"
    write_severity_inference_manifest(audited["recordings"], inference_manifest)

    score_path = artifacts / "evaluation" / "severity_phone_scores.jsonl.gz"
    if scorer is not None:
        produced = scorer(audited["recordings"])
        if isinstance(produced, (str, Path)):
            score_path = Path(produced)
            if not score_path.is_file():
                raise SeverityError(f"severity scorer returned a missing path: {score_path}")
        else:
            write_jsonl(produced, score_path)
    elif not score_path.is_file():
        raise FileNotFoundError(
            "dedicated eight-speaker severity scores are missing: "
            f"{score_path}. Generate one phone-score row per canonical token for all "
            f"{expected_recordings} audited recordings using the supplied recording_id, "
            "event_id, official_logits_npz, and prompt. Required fields are method="
            f"{NEW_METHOD!r}, speaker, event, recording_id, phone_index, and gop. "
            "The primary5/sensitivity7 prediction artifact is invalid here because it "
            "omits M03."
        )

    score_rows = _coerce_rows(score_path)
    canonical_by_recording = canonical_phone_sequences_from_audited_prompts(
        audited["recordings"]
    )
    validate_severity_phone_scores(
        score_rows,
        audited_recordings=audited["recordings"],
        canonical_phones_by_recording=canonical_by_recording,
        expected_recordings=expected_recordings,
        expected_events=expected_events,
        expected_speakers=tuple(str(value) for value in severity["dysarthric_speakers"]),
    )

    normalized_frozen = load_frozen_score_sources({
        str(name): Path(str(path)) for name, path in frozen_sources.items()
    })
    report = run_severity_experiment(
        score_path,
        manifest_path,
        normalized_frozen,
        new_scores_are_phone_level=True,
        new_method=NEW_METHOD,
        n_bootstrap=n_bootstrap,
        seed=seed,
        expected_manifest_rows=expected_recordings,
        expected_events=expected_events,
        expected_speakers=expected_speakers,
    )
    report["audited_inference"] = {
        "manifest_filename": inference_manifest.name,
        "manifest_sha256": _sha256(inference_manifest),
        "official_logits_cache": str(logits_cache),
        "n_recordings": audited["n_recordings"],
        "n_events": audited["n_events"],
        "score_all_audited_recordings": True,
    }
    report["frozen_result_sources"] = {
        str(name): {"filename": Path(str(path)).name, "sha256": _sha256(path)}
        for name, path in sorted(frozen_sources.items())
    }
    json_path = artifacts / "evaluation" / "severity_metrics.json"
    csv_path = artifacts / "evaluation" / "severity_metrics.csv"
    write_severity_outputs(report, json_path, csv_path)
    write_stage_summary(
        artifacts / "summaries" / "run-severity.json",
        stage="run-severity",
        config_sha256=str(config.get("_config_hash", "")),
        training_speakers=[],
        source_files={
            "audited_manifest": _sha256(manifest_path),
            **{
                f"frozen_{name}": _sha256(path)
                for name, path in sorted(frozen_sources.items())
            },
        },
        models={},
        manifests={"severity_audited_recordings": _sha256(inference_manifest)},
        upstream_artifacts={
            "severity_phone_scores": _sha256(score_path),
            "severity_phone_scores_provenance": _sha256(
                artifacts / "evaluation" / "severity_phone_scores.provenance.json"
            ),
        },
        exclusions={},
        outputs={
            "severity_metrics": _sha256(json_path),
            "severity_metrics_csv": _sha256(csv_path),
            "severity_inference_manifest": _sha256(inference_manifest),
        },
        details={
            "n_recordings": expected_recordings,
            "n_events": expected_events,
            "n_speakers": expected_speakers,
            "event_first_then_speaker_first": True,
            "severity_used_for_model_training": False,
            "system_level_comparison_only": True,
        },
    )
    return report


def legacy_severity_paths(legacy_code_root: str | Path) -> dict[str, Path]:
    """Return the frozen code9 data paths without importing historical code."""

    root = Path(legacy_code_root)
    return {
        "audited_manifest": root / "code9" / "manifests" / "torgo_mixgop_release.jsonl",
        "audited_logits_cache": root / "code9" / "cache" / "ctc_sf_official",
        "frozen_scores": (
            root / "code9" / "results" / "common_evaluation" / "inputs"
            / "headline_dys_415.jsonl"
        ),
    }
