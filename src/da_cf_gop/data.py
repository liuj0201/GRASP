"""Auditable TORGO manifest construction for DA-CF-GoP.

The acoustic input is always head-microphone audio.  PHN is retained only as
an evaluation/training-label sidecar; F01 and M02 may use the sequence from the
same event's array microphone because TORGO has no head-microphone PHN for
those speakers.  No PHN timing is serialized into the manifest.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .phonology import (
    ARPABET_39,
    StableTokenLabel,
    derive_stable_token_labels,
    g2p_provenance,
    normalize_prompt,
    read_phn,
    texts_to_phones,
    validate_prompt,
)


PRIMARY5 = ("F03", "F04", "M01", "M04", "M05")
SENSITIVITY7 = ("F01", "F03", "F04", "M01", "M02", "M04", "M05")
HEALTHY_PHN = ("MC01", "MC02", "MC03", "MC04")
COHORTS = {"primary5": PRIMARY5, "sensitivity7": SENSITIVITY7}
CROSS_MIC_PHN_SPEAKERS = frozenset({"F01", "M02"})
MANIFEST_SCHEMA_VERSION = "da-cf-gop.manifest.v1"

G2PFunction = Callable[
    [Sequence[str]], tuple[list[list[str]], list[list[str]]]
]
AudioProbe = Callable[[Path], tuple[float, int]]


def _sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ManifestRow:
    schema_version: str
    reading_event_id: str
    speaker_id: str
    speaker_group: str
    sex: str
    session: str
    stem: str
    wav_path: str
    audio_microphone: str
    duration_s: float
    sample_rate: int
    prompt_path: str
    prompt: str
    prompt_id: str
    canonical_phones: tuple[str, ...]
    phn_path: str
    phn_microphone: str
    phn_raw_labels: tuple[str, ...]
    phn_timit_phones: tuple[str, ...]
    phn_model_phones: tuple[str, ...]
    stable_tokens: tuple[StableTokenLabel, ...]
    stable_count: int
    uncertain_count: int
    stable_fraction: float
    levenshtein_insertions: int
    category_insertions: int
    agreed_insertions: int
    severity_rank: float | None
    severity_label: str
    # Content hashes freeze the three physical sources used to construct this
    # row.  Empty strings are accepted only when reading a legacy manifest;
    # every newly built manifest populates all four bindings.
    wav_sha256: str = ""
    prompt_sha256: str = ""
    phn_sha256: str = ""
    g2p_descriptor_sha256: str = ""

    @property
    def utterance_id(self) -> str:
        return self.reading_event_id

    @property
    def transcript(self) -> str:
        return self.prompt

    @property
    def phones(self) -> list[str]:
        return list(self.canonical_phones)

    @property
    def has_phn(self) -> bool:
        return bool(self.phn_path)

    def to_dict(self) -> dict[str, object]:
        """Return a standard-JSON-compatible record (no NaN, no Path)."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ManifestRow":
        raw = dict(value)
        # Backward-safe in-memory migration for manifests created before source
        # hashes were serialized.  This never rewrites an artifact implicitly.
        for digest_name, path_name in (
            ("wav_sha256", "wav_path"),
            ("prompt_sha256", "prompt_path"),
            ("phn_sha256", "phn_path"),
        ):
            if digest_name not in raw:
                source = Path(str(raw.get(path_name, "")))
                raw[digest_name] = _sha256_file(source) if source.is_file() else ""
        raw.setdefault("g2p_descriptor_sha256", "")
        raw["canonical_phones"] = tuple(raw.get("canonical_phones", ()))
        raw["phn_raw_labels"] = tuple(raw.get("phn_raw_labels", ()))
        raw["phn_timit_phones"] = tuple(raw.get("phn_timit_phones", ()))
        raw["phn_model_phones"] = tuple(raw.get("phn_model_phones", ()))
        raw["stable_tokens"] = tuple(
            token if isinstance(token, StableTokenLabel) else StableTokenLabel.from_dict(token)
            for token in raw.get("stable_tokens", ())  # type: ignore[union-attr]
        )
        return cls(**raw)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Exclusion:
    speaker_id: str
    session: str
    stem: str
    reading_event_id: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Exclusion":
        return cls(**value)  # type: ignore[arg-type]


def speaker_set(speaker_id: str) -> str:
    match = re.fullmatch(r"([A-Z]+)[0-9]+", speaker_id.upper())
    if not match:
        raise ValueError(f"invalid TORGO speaker id: {speaker_id!r}")
    return match.group(1)


def reading_event_id(speaker_id: str, session: str, stem: str) -> str:
    """Canonical event key: the tuple is losslessly encoded with separators."""

    if any(not str(value).strip() for value in (speaker_id, session, stem)):
        raise ValueError("speaker_id, session, and stem must be non-empty")
    return f"{speaker_id}_{session}_{stem}"


def _score_letter(value: str) -> float | None:
    scores = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    parts = [part.strip().lower() for part in (value or "").replace("\\", "/").split("/")]
    values = [scores[part] for part in parts if part in scores]
    return sum(values) / len(values) if values else None


def read_notes(speaker_dir: Path, *, healthy: bool) -> tuple[float | None, str]:
    if healthy:
        return 0.0, "control"
    intelligibility: dict[str, str] = {}
    for path in sorted((speaker_dir / "Notes").glob("*.csv")):
        section = ""
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.reader(handle):
                cells = [cell.strip() for cell in row]
                if not any(cells):
                    continue
                first = cells[0].lower() if cells else ""
                if first.startswith("intel"):
                    section = "intel"
                elif first:
                    section = first
                if section == "intel" and len(cells) >= 3 and cells[1]:
                    intelligibility[cells[1].lower()] = cells[2]
    values = [_score_letter(value) for value in intelligibility.values()]
    finite = [value for value in values if value is not None]
    label = ";".join(f"{key}={value}" for key, value in sorted(intelligibility.items()))
    return (sum(finite) / len(finite) if finite else None), label


def _index_files(directory: Path, suffix: str) -> dict[str, tuple[Path, ...]]:
    indexed: dict[str, list[Path]] = {}
    if not directory.is_dir():
        return {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() == suffix.lower():
            indexed.setdefault(path.stem.casefold(), []).append(path)
    return {stem: tuple(paths) for stem, paths in indexed.items()}


def _audio_info(path: Path) -> tuple[float, int]:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.duration), int(info.samplerate)


def _cohort_selection(
    dys_speakers: Sequence[str] | str | None,
    healthy_speakers: Sequence[str] | None,
    cohort: str | None,
    sensitivity: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    if isinstance(dys_speakers, str):
        if cohort is not None:
            raise ValueError("cohort was supplied twice")
        cohort = dys_speakers
        dys_speakers = None
    selected_name = cohort or ("sensitivity7" if sensitivity else "primary5")
    if dys_speakers is None:
        if selected_name not in COHORTS:
            raise ValueError(f"unknown cohort {selected_name!r}; expected one of {sorted(COHORTS)}")
        dys = COHORTS[selected_name]
    else:
        dys = tuple(str(value).upper() for value in dys_speakers)
        selected_name = cohort or "custom"
    healthy_values = HEALTHY_PHN if healthy_speakers is None else healthy_speakers
    healthy = tuple(str(value).upper() for value in healthy_values)
    if not dys:
        raise ValueError("at least one dysarthric speaker is required")
    if len(set((*dys, *healthy))) != len(dys) + len(healthy):
        raise ValueError("dysarthric and healthy speaker lists must be unique and disjoint")
    return tuple(dys), healthy, selected_name


def build_torgo_manifest(
    data_root: str | Path,
    dys_speakers: Sequence[str] | str | None = None,
    healthy_speakers: Sequence[str] | None = None,
    *,
    cohort: str | None = None,
    sensitivity: bool = False,
    minimum_stable_fraction: float = 0.80,
    min_duration_s: float = 0.05,
    g2p: G2PFunction | None = None,
    audio_probe: AudioProbe | None = None,
    allow_cross_mic_phn_for: Iterable[str] = CROSS_MIC_PHN_SPEAKERS,
) -> tuple[list[ManifestRow], list[Exclusion], dict[str, object]]:
    """Build the frozen headMic + independent-PHN TORGO manifest.

    ``dys_speakers`` may be a speaker sequence or the strings ``primary5`` /
    ``sensitivity7`` for a concise CLI call.  The returned rows include stable
    labels, but never PHN boundaries.  Callers must strip all ``phn_*`` and gold
    fields before constructing an acoustic inference artifact.
    """

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"TORGO data root not found: {root}")
    if not 0.0 <= minimum_stable_fraction <= 1.0:
        raise ValueError("minimum_stable_fraction must lie in [0, 1]")
    dys, healthy, cohort_name = _cohort_selection(
        dys_speakers, healthy_speakers, cohort, sensitivity
    )
    cross_mic = frozenset(str(value).upper() for value in allow_cross_mic_phn_for)
    requested = tuple((*dys, *healthy))
    probe = audio_probe or _audio_info
    g2p_function = g2p or texts_to_phones
    if g2p is None:
        g2p_details = g2p_provenance()
    else:
        g2p_details = {
            "backend": "injected_g2p_callable",
            "callable": (
                f"{getattr(g2p, '__module__', '')}."
                f"{getattr(g2p, '__qualname__', getattr(g2p, '__name__', type(g2p).__name__))}"
            ).strip("."),
        }
        g2p_details["descriptor_sha256"] = _sha256_json(g2p_details)
    g2p_descriptor_sha256 = str(g2p_details["descriptor_sha256"])
    exclusions: list[Exclusion] = []
    provisional: list[dict[str, object]] = []
    seen_event_ids: set[str] = set()

    for speaker_id in requested:
        is_healthy = speaker_id in healthy
        speaker_dir = root / speaker_set(speaker_id) / speaker_id
        if not speaker_dir.is_dir():
            exclusions.append(Exclusion(
                speaker_id, "", "", "", "missing_speaker_directory", str(speaker_dir)
            ))
            continue
        severity_rank, severity_label = read_notes(speaker_dir, healthy=is_healthy)
        sessions = sorted(
            (path for path in speaker_dir.iterdir()
             if path.is_dir() and path.name.lower().startswith("session")),
            key=lambda path: path.name.lower(),
        )
        if not sessions:
            exclusions.append(Exclusion(
                speaker_id, "", "", "", "missing_session_directory", str(speaker_dir)
            ))
        for session_dir in sessions:
            wavs = _index_files(session_dir / "wav_headMic", ".wav")
            prompts = _index_files(session_dir / "prompts", ".txt")
            head_phn = _index_files(session_dir / "phn_headMic", ".phn")
            array_phn = _index_files(session_dir / "phn_arrayMic", ".phn")
            for stem_key, wav_candidates in wavs.items():
                stem = wav_candidates[0].stem
                event_id = reading_event_id(speaker_id, session_dir.name, stem)
                if len(wav_candidates) != 1:
                    exclusions.append(Exclusion(
                        speaker_id, session_dir.name, stem, event_id,
                        "ambiguous_headmic_audio", ";".join(str(path) for path in wav_candidates),
                    ))
                    continue
                if event_id in seen_event_ids:
                    exclusions.append(Exclusion(
                        speaker_id, session_dir.name, stem, event_id, "duplicate_reading_event"
                    ))
                    continue
                seen_event_ids.add(event_id)
                prompt_candidates = prompts.get(stem_key, ())
                if len(prompt_candidates) != 1:
                    reason = "missing_prompt" if not prompt_candidates else "ambiguous_prompt"
                    exclusions.append(Exclusion(
                        speaker_id, session_dir.name, stem, event_id, reason,
                        ";".join(str(path) for path in prompt_candidates),
                    ))
                    continue
                prompt = normalize_prompt(
                    prompt_candidates[0].read_text(encoding="utf-8", errors="replace")
                )
                valid, reason = validate_prompt(prompt)
                if not valid:
                    exclusions.append(Exclusion(
                        speaker_id, session_dir.name, stem, event_id, reason
                    ))
                    continue

                phn_candidates = head_phn.get(stem_key, ())
                phn_microphone = "headMic"
                if not phn_candidates and speaker_id in cross_mic:
                    phn_candidates = array_phn.get(stem_key, ())
                    phn_microphone = "arrayMic"
                if len(phn_candidates) != 1:
                    reason = "missing_phn_sequence" if not phn_candidates else "ambiguous_phn_sequence"
                    exclusions.append(Exclusion(
                        speaker_id, session_dir.name, stem, event_id, reason,
                        ";".join(str(path) for path in phn_candidates),
                    ))
                    continue
                try:
                    duration_s, sample_rate = probe(wav_candidates[0])
                except Exception as exc:
                    exclusions.append(Exclusion(
                        speaker_id, session_dir.name, stem, event_id,
                        "unreadable_audio", f"{type(exc).__name__}: {exc}",
                    ))
                    continue
                if not math.isfinite(duration_s) or duration_s < min_duration_s or sample_rate <= 0:
                    exclusions.append(Exclusion(
                        speaker_id, session_dir.name, stem, event_id, "invalid_audio_metadata",
                        f"duration_s={duration_s};sample_rate={sample_rate}",
                    ))
                    continue
                try:
                    phn = read_phn(phn_candidates[0], strict=True)
                except (OSError, ValueError) as exc:
                    exclusions.append(Exclusion(
                        speaker_id, session_dir.name, stem, event_id,
                        "invalid_phn", f"{type(exc).__name__}: {exc}",
                    ))
                    continue
                if not phn.model_phones:
                    exclusions.append(Exclusion(
                        speaker_id, session_dir.name, stem, event_id, "empty_phn_sequence"
                    ))
                    continue
                provisional.append({
                    "reading_event_id": event_id,
                    "speaker_id": speaker_id,
                    "speaker_group": "healthy" if is_healthy else "dysarthric",
                    "sex": "female" if speaker_id.startswith("F") else "male",
                    "session": session_dir.name,
                    "stem": stem,
                    "wav_path": str(wav_candidates[0].resolve()),
                    "wav_sha256": _sha256_file(wav_candidates[0]),
                    "duration_s": float(duration_s),
                    "sample_rate": int(sample_rate),
                    "prompt_path": str(prompt_candidates[0].resolve()),
                    "prompt_sha256": _sha256_file(prompt_candidates[0]),
                    "prompt": prompt,
                    "phn_path": str(phn_candidates[0].resolve()),
                    "phn_sha256": _sha256_file(phn_candidates[0]),
                    "phn_microphone": phn_microphone,
                    "phn_raw_labels": phn.raw_labels,
                    "phn_timit_phones": phn.timit_phones,
                    "phn_model_phones": phn.model_phones,
                    "severity_rank": severity_rank,
                    "severity_label": severity_label,
                    "g2p_descriptor_sha256": g2p_descriptor_sha256,
                })

    unique_prompts = list(dict.fromkeys(str(item["prompt"]) for item in provisional))
    mapped_rows, unknown_rows = g2p_function(unique_prompts)
    if len(mapped_rows) != len(unique_prompts) or len(unknown_rows) != len(unique_prompts):
        raise ValueError("G2P returned a different number of rows than prompts")
    phones_by_prompt = {
        prompt: tuple(str(phone).upper() for phone in phones)
        for prompt, phones in zip(unique_prompts, mapped_rows)
    }
    unknown_by_prompt = {
        prompt: tuple(str(phone) for phone in phones)
        for prompt, phones in zip(unique_prompts, unknown_rows)
    }

    rows: list[ManifestRow] = []
    for item in provisional:
        prompt = str(item["prompt"])
        unknown = unknown_by_prompt[prompt]
        canonical = phones_by_prompt[prompt]
        event_id = str(item["reading_event_id"])
        if unknown:
            exclusions.append(Exclusion(
                str(item["speaker_id"]), str(item["session"]), str(item["stem"]), event_id,
                "unmapped_g2p_phone", " ".join(unknown),
            ))
            continue
        if not canonical:
            exclusions.append(Exclusion(
                str(item["speaker_id"]), str(item["session"]), str(item["stem"]), event_id,
                "empty_canonical_sequence",
            ))
            continue
        invalid = sorted(set(canonical) - set(ARPABET_39))
        if invalid:
            exclusions.append(Exclusion(
                str(item["speaker_id"]), str(item["session"]), str(item["stem"]), event_id,
                "canonical_outside_model_vocab", " ".join(invalid),
            ))
            continue
        stable = derive_stable_token_labels(
            canonical, item["phn_model_phones"],  # type: ignore[arg-type]
            minimum_stable_fraction=minimum_stable_fraction,
        )
        rows.append(ManifestRow(
            schema_version=MANIFEST_SCHEMA_VERSION,
            canonical_phones=canonical,
            prompt_id=hashlib.sha256(prompt.lower().encode("utf-8")).hexdigest()[:16],
            audio_microphone="headMic",
            stable_tokens=stable.tokens,
            stable_count=stable.stable_count,
            uncertain_count=stable.uncertain_count,
            stable_fraction=stable.stable_fraction,
            levenshtein_insertions=stable.levenshtein_insertions,
            category_insertions=stable.category_insertions,
            agreed_insertions=stable.agreed_insertions,
            **item,  # type: ignore[arg-type]
        ))

    rows.sort(key=lambda row: (row.speaker_id, row.session.lower(), row.stem.lower()))
    ids = [row.reading_event_id for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate reading_event_id survived manifest construction")
    canonical_tokens = sum(len(row.canonical_phones) for row in rows)
    stable_tokens = sum(row.stable_count for row in rows)
    stable_fraction = stable_tokens / canonical_tokens if canonical_tokens else 0.0
    patient_rows = [row for row in rows if row.speaker_group == "dysarthric"]
    patient_canonical_tokens = sum(len(row.canonical_phones) for row in patient_rows)
    patient_stable_tokens = sum(row.stable_count for row in patient_rows)
    patient_stable_fraction = (
        patient_stable_tokens / patient_canonical_tokens
        if patient_canonical_tokens else 0.0
    )
    exclusion_counts = Counter(exclusion.reason for exclusion in exclusions)
    rows_by_speaker = Counter(row.speaker_id for row in rows)
    phn_mic_counts = Counter(row.phn_microphone for row in rows)
    serialized_rows = [row.to_dict() for row in rows]
    source_hash_inventory: dict[str, dict[str, object]] = {}
    for role, path_name, digest_name in (
        ("wav", "wav_path", "wav_sha256"),
        ("prompt", "prompt_path", "prompt_sha256"),
        ("phn", "phn_path", "phn_sha256"),
    ):
        entries = [
            {
                "reading_event_id": row.reading_event_id,
                "path": getattr(row, path_name),
                "sha256": getattr(row, digest_name),
            }
            for row in rows
        ]
        source_hash_inventory[role] = {
            "n_files": len(entries),
            "inventory_sha256": _sha256_json(entries),
        }
    manifest_hash = hashlib.sha256(
        json.dumps(
            serialized_rows, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    audit: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "cohort": cohort_name,
        "data_root": str(root),
        "dysarthric_speakers_requested": list(dys),
        "healthy_speakers_requested": list(healthy),
        "audio_policy": "headMic only",
        "phn_policy": "same-event headMic; arrayMic sequence only for F01/M02",
        "phn_boundaries_serialized": False,
        "reading_event_key": ["speaker_id", "session", "stem"],
        "n_rows": len(rows),
        "n_patient_rows": len(patient_rows),
        "n_unique_prompts": len({row.prompt_id for row in rows}),
        "n_unique_canonical_sequences": len({row.canonical_phones for row in rows}),
        "n_exclusions": len(exclusions),
        "rows_by_speaker": dict(sorted(rows_by_speaker.items())),
        "phn_microphone_counts": dict(sorted(phn_mic_counts.items())),
        "exclusion_reasons": dict(sorted(exclusion_counts.items())),
        "canonical_tokens": canonical_tokens,
        "stable_tokens": stable_tokens,
        "uncertain_tokens": canonical_tokens - stable_tokens,
        "stable_fraction": stable_fraction,
        "patient_canonical_tokens": patient_canonical_tokens,
        "patient_stable_tokens": patient_stable_tokens,
        "patient_uncertain_tokens": patient_canonical_tokens - patient_stable_tokens,
        "patient_stable_fraction": patient_stable_fraction,
        "minimum_stable_fraction": minimum_stable_fraction,
        "exact_substitution_deletion_headline_allowed": (
            patient_canonical_tokens > 0
            and patient_stable_fraction >= minimum_stable_fraction
        ),
        "headline_stability_population": "dysarthric canonical tokens only",
        "levenshtein_insertions": sum(row.levenshtein_insertions for row in rows),
        "category_insertions": sum(row.category_insertions for row in rows),
        "agreed_insertions": sum(row.agreed_insertions for row in rows),
        "manifest_sha256": manifest_hash,
        "input_source_hashes": source_hash_inventory,
        "g2p_provenance": g2p_details,
    }
    return rows, exclusions, audit


def build_manifest(
    data_root: str | Path,
    cohort: str = "primary5",
    **kwargs,
) -> tuple[list[ManifestRow], list[Exclusion], dict[str, object]]:
    """Convenience alias used by the CLI."""

    return build_torgo_manifest(data_root, cohort=cohort, **kwargs)


def _open_text(path: Path, mode: str):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def write_manifest(rows: Iterable[ManifestRow], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(destination, "w") as handle:
        for row in sorted(rows, key=lambda value: value.reading_event_id):
            handle.write(json.dumps(
                row.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False,
                separators=(",", ":"),
            ) + "\n")


def read_manifest(path: str | Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with _open_text(Path(path), "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(ManifestRow.from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid manifest row {line_number} in {path}") from exc
    ids = [row.reading_event_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate reading_event_id in {path}")
    return rows


def write_exclusions(rows: Iterable[Exclusion], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(destination, "w") as handle:
        for row in rows:
            handle.write(json.dumps(
                row.to_dict(), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ) + "\n")


def write_audit(audit: dict[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
