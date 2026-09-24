"""Read-only TORGO provenance, coverage, and symbol audit for the dual graph.

This audit does not regenerate gold or consume acoustic predictions.  It writes
new reports only and never changes a frozen manifest or the source PHN files.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from .phonology import ARPABET_39, PHN_ALLOPHONE_MAP, PHN_DROP, read_phn


PROVENANCE = {
    "status": "corpus_level_manual_error_annotation_supported",
    "per_file_independent_annotation_audit": "not_available",
    "audio_spot_checks_this_run": "not_performed",
    "sources": [
        {
            "url": "https://catalog.ldc.upenn.edu/docs/LDC2012S02/README.txt",
            "location": "TORGO directory structure, phn_*",
            "supports": "Wavesurfer TIMIT transcription of audio; head/array microphone association; closure labels",
            "does_not_establish": "independent human error decisions for every released file",
        },
        {
            "url": "https://www.cs.toronto.edu/~frank/Download/Papers/rudzicz_nips10.pdf",
            "location": "Section 2, printed page 2, last paragraph",
            "supports": "TORGO phoneme boundaries and pronunciation errors transcribed by a speech-language pathologist to TIMIT",
            "does_not_establish": "per-file annotator identity or inter-rater reliability in the current release",
        },
        {
            "url": "https://aclanthology.org/P10-1007.pdf",
            "location": "Section 2.2, printed pages 61-62",
            "supports": "SLP-set TORGO boundaries in that experiment",
            "does_not_establish": "dysarthric error labels: that experiment explicitly excluded cerebrally palsied speech",
        },
    ],
    "interpretation": (
        "Human actual-pronunciation annotation is supported at corpus level; "
        "do not call this newly independently adjudicated clinical gold. "
        "Differences from canonical G2P were not used as provenance evidence."
    ),
}

# Nonstandard vcl/eng-cl are not automatically treated as paired plosives.
CLOSURE_RELEASE = {phone + "cl": phone for phone in ("b", "d", "g", "k", "p", "t")}
CLOSURE_LABELS = frozenset((*CLOSURE_RELEASE, "vcl", "eng-cl"))


def closure_audit(segments) -> list[dict]:
    """Pair only immediately contiguous, same-place closure/release labels.

    Pairing is metadata only.  Unpaired closure is flagged, never interpreted
    as an insertion or silently turned into an observed released plosive.
    """
    rows = []
    for index, segment in enumerate(segments):
        label = segment.raw_label.lower()
        if label not in CLOSURE_LABELS:
            continue
        following = segments[index + 1] if index + 1 < len(segments) else None
        expected = CLOSURE_RELEASE.get(label)
        paired = bool(
            expected is not None and following is not None
            and following.raw_label.lower() == expected
            and segment.end_sample == following.start_sample
        )
        rows.append({
            "raw_index": index, "closure": label,
            "following_label": following.raw_label.lower() if following else None,
            "contiguous_matching_release": paired,
            "status": "paired_closure_release" if paired else "unpaired_closure_requires_review",
        })
    return rows


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def corpus_inventory(data_root: Path) -> dict[str, dict]:
    """Count same-event file availability, including speakers not in a cohort."""
    inventory = {}
    for group in sorted(data_root.iterdir()):
        if not group.is_dir():
            continue
        for speaker in sorted(group.iterdir()):
            if not speaker.is_dir():
                continue
            counts = Counter()
            for session in sorted(speaker.glob("Session*")):
                if not session.is_dir():
                    continue
                def stems(directory, suffix):
                    return {path.stem.casefold() for path in directory.iterdir()
                            if path.is_file() and path.suffix.lower() == suffix} if directory.is_dir() else set()
                audio = stems(session / "wav_headMic", ".wav")
                head = stems(session / "phn_headMic", ".phn")
                array = stems(session / "phn_arrayMic", ".phn")
                prompt = stems(session / "prompts", ".txt")
                counts["headmic_audio_events"] += len(audio)
                counts["headmic_with_headmic_phn"] += len(audio & head)
                counts["headmic_with_arraymic_phn"] += len(audio & array)
                counts["headmic_with_any_phn"] += len(audio & (head | array))
                counts["headmic_with_prompt"] += len(audio & prompt)
            inventory[speaker.name] = dict(counts)
    return inventory


def audit_manifest(manifest: Path, exclusions: Path | None = None) -> tuple[list[dict], dict]:
    rows = read_jsonl(manifest)
    seen = set()
    records = []
    speakers = defaultdict(Counter)
    all_raw = Counter()
    for row in rows:
        key = row["reading_event_id"]
        if key in seen:
            raise ValueError(f"duplicate reading event: {key}")
        seen.add(key)
        speaker = row["speaker_id"]
        counts = speakers[speaker]
        counts["included_utterances"] += 1
        counts["canonical_tokens"] += len(row["canonical_phones"])
        counts["stable_tokens_legacy"] += row["stable_count"]
        counts["uncertain_tokens_legacy"] += row["uncertain_count"]
        counts["insertions_legacy_agreed"] += row["agreed_insertions"]
        counts[f"phn_microphone_{row['phn_microphone']}"] += 1
        for label in row["stable_tokens"]:
            if label["stable"]:
                counts[f"legacy_{label['event_type']}"] += 1

        phn_path = Path(row["phn_path"])
        phn = read_phn(phn_path)
        closures = closure_audit(phn.segments)
        unpaired = [item for item in closures if not item["contiguous_matching_release"]]
        raw_counts = Counter(segment.raw_label.lower() for segment in phn.segments)
        all_raw.update(raw_counts)
        counts["closures"] += len(closures)
        counts["unpaired_closures"] += len(unpaired)
        counts["utterances_with_unpaired_closures"] += bool(unpaired)
        counts["raw_dx"] += raw_counts["dx"]
        counts["raw_ax_or_ax_h"] += raw_counts["ax"] + raw_counts["ax-h"]
        counts["raw_q_dropped"] += raw_counts["q"]
        counts["duration_milliseconds"] += round(1000 * row["duration_s"])

        missing = [field for field in ("wav_path", "prompt_path", "phn_path")
                   if not Path(row[field]).is_file()]
        same_event = (phn_path.stem.casefold() == str(row["stem"]).casefold()
                      and phn_path.parent.parent.name == row["session"]
                      and phn_path.parent.parent.parent.name == speaker)
        model_sequence_matches = tuple(row["phn_model_phones"]) == phn.model_phones
        record = {
            "reading_event_id": key, "speaker_id": speaker,
            "session": row["session"], "stem": row["stem"],
            "audio_microphone": row["audio_microphone"],
            "phn_microphone": row["phn_microphone"],
            "wav_path": row["wav_path"], "prompt_path": row["prompt_path"],
            "phn_path": row["phn_path"], "prompt": row["prompt"],
            "sample_rate": row["sample_rate"], "duration_s": row["duration_s"],
            "annotation_provenance": PROVENANCE["status"],
            "annotation_provenance_scope": "corpus_level_not_per_file_adjudication",
            "source_paths_exist": not missing, "same_event_phn": same_event,
            "raw_model_sequence_matches_manifest": model_sequence_matches,
            "closure_pairs": closures, "has_unpaired_closure": bool(unpaired),
            "quality_exclusion": (
                "source_integrity_failure" if missing or not same_event or not model_sequence_matches
                else "unpaired_or_nonmatching_closure_release" if unpaired else None
            ),
            "raw_symbol_counts": dict(sorted(raw_counts.items())),
            "mapping_limited_symbols": {symbol: raw_counts[symbol] for symbol in
                ("dx", "ax", "ax-h", "q", "nx", "el", "em", "en", "eng", "axr", "ix", "ux", "hv")
                if raw_counts[symbol]},
            "usable_for_frozen_model_space_comparison": not missing and same_event and model_sequence_matches,
            "exact_transcription_scope_note": "39-phone projected transcription only; unpaired closures require separate review",
        }
        if not record["usable_for_frozen_model_space_comparison"]:
            counts["source_integrity_failures"] += 1
        records.append(record)

    if exclusions is not None and exclusions.exists():
        for row in read_jsonl(exclusions):
            speakers[row["speaker_id"]][f"excluded_{row['reason']}"] += 1
            speakers[row["speaker_id"]]["excluded_utterances"] += 1
    for counts in speakers.values():
        total = counts["included_utterances"] + counts["excluded_utterances"]
        counts["candidate_headmic_events"] = total
        counts["manifest_coverage"] = counts["included_utterances"] / total if total else 0
    summary = {
        "schema_version": "da-cf-gop.dual-data-audit.v1",
        "cohort": manifest.name.split(".")[0],
        "annotation_provenance": PROVENANCE,
        "corpus_file_availability": corpus_inventory(Path(rows[0]["phn_path"]).parents[4]) if rows else {},
        "utterances": len(rows), "unique_events": len(seen),
        "speakers": {key: dict(value) for key, value in sorted(speakers.items())},
        "raw_symbol_counts": dict(sorted(all_raw.items())),
        "model_inventory": list(ARPABET_39), "blank_id": 0,
        "symbol_mapping": {
            "AX": {"model": "AH", "lost_contrast": "schwa versus AH"},
            "DX": {"model": "T", "lost_contrast": "tap versus T; normal D flapping may look like D-to-T"},
            "other_allophone_folds": PHN_ALLOPHONE_MAP,
            "legacy_dropped_symbols": sorted(PHN_DROP),
            "q": {"model": None, "lost_contrast": "observed glottal stop removed whereas canonical G2P glottal stop maps to T"},
        },
        "policy": {
            "audit_never_changes_source_labels": True,
            "same_event_cross_mic_sequences_only": True,
            "closure_pairing_requires_contiguous_matching_release": True,
            "unpaired_closures_are_not_insertions": True,
            "first_run_label_policy": "exclude quality_exclusion rows uniformly from training/development labels and test evaluation; inference may still predict all audio/prompt rows",
            "legacy_event_counts_not_new_dual_graph_gold": True,
            "excluded_denominator": "headMic events considered by the frozen manifest builder; not all corpus recordings",
            "boundary_info_used_only_by_audit_not_inference": True,
        },
    }
    return records, summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
    parser.add_argument("--cohort", choices=("primary5", "sensitivity7", "all"), default="all")
    args = parser.parse_args(argv)
    cohorts = ("primary5", "sensitivity7") if args.cohort == "all" else (args.cohort,)
    output = args.artifacts / "evaluation" / "dual_graph" / "data_audit"
    output.mkdir(parents=True, exist_ok=True)
    for cohort in cohorts:
        manifest_root = args.artifacts / "manifests"
        records, summary = audit_manifest(manifest_root / f"{cohort}.jsonl.gz",
                                         manifest_root / f"{cohort}.exclusions.jsonl.gz")
        (output / f"{cohort}.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with gzip.open(output / f"{cohort}.records.jsonl.gz", "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"cohort": cohort, "utterances": len(records), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
