"""Observed, training-patient-only substitution neighbors for sparse graphs.

No smoothing adds unobserved edges. A missing reference phone has no empirical
neighbors and is left to the acoustic selector. ``max_neighbors`` is an explicit
candidate budget, not a phonological or dysarthria-specific law.
"""

from collections import Counter, defaultdict

from .phonology import CTC_ID_TO_PHONE, PHONE_TO_CTC_ID


def estimate_neighbors(records, training_speakers, max_neighbors=4, min_count=1):
    """Return 40 directed neighbor rows and auditable supporting observations.

    Each training patient's stable, diagnostic tokens have total weight one.
    Only substitution tokens contribute edges, so the balanced rate of p->q is
    the mean over training patients of count(p->q)/diagnostic_token_count.
    Correct and deleted tokens count as exposure, matching the patient weighting
    used by the existing error-prior estimator. This rate is not P(q | p, error).

    Non-training records are filtered before their labels are read. The caller
    must exclude a training patient's own records when producing that patient's
    training features; development/test selectors use all training patients.
    """
    speakers = tuple(sorted(set(training_speakers)))
    if not speakers or max_neighbors < 0 or min_count < 1:
        raise ValueError("nonempty training speakers, nonnegative budget and positive min_count required")
    counts = {speaker: Counter() for speaker in speakers}
    exposure = Counter({speaker: 0 for speaker in speakers})
    evidence = defaultdict(list)
    selected_records = 0
    for record in records:
        speaker = record["speaker"]
        if speaker not in counts or record.get("quality_exclusion") is not None:
            continue
        selected_records += 1
        for token in record["labels"]["tokens"]:
            if not token["stable_event"] or not token["diagnostic"]:
                continue
            exposure[speaker] += 1
            if token["event_type"] != "substitution":
                continue
            position = token["phone_index"]
            source = PHONE_TO_CTC_ID[record["canonical_phones"][position]]
            target = PHONE_TO_CTC_ID[token["realized_phone"]]
            if source == target or source == 0 or target == 0:
                continue
            counts[speaker][source, target] += 1
            evidence[source, target].append({
                "speaker": speaker, "event": record["event"], "phone_index": position,
            })

    neighbors = [[] for _ in range(40)]
    by_source = defaultdict(list)
    for (source, target), supporting in evidence.items():
        raw_count = len(supporting)
        balanced = sum(
            counts[speaker][source, target] / exposure[speaker]
            for speaker in speakers if exposure[speaker]
        ) / len(speakers)
        by_source[source].append({
            "source_phone": CTC_ID_TO_PHONE[source], "source_id": source,
            "target_phone": CTC_ID_TO_PHONE[target], "target_id": target,
            "raw_count": raw_count,
            "speaker_count": sum(counts[speaker][source, target] > 0 for speaker in speakers),
            "balanced_rate": balanced,
            "supporting_training_events": sorted(
                supporting, key=lambda row: (row["speaker"], row["event"], row["phone_index"]),
            ),
        })

    sources = []
    for source in sorted(by_source):
        ranked = sorted(by_source[source], key=lambda row: (
            -row["balanced_rate"], -row["raw_count"], row["target_id"],
        ))
        eligible = [row for row in ranked if row["raw_count"] >= min_count]
        neighbors[source] = [row["target_id"] for row in eligible[:max_neighbors]]
        for rank, row in enumerate(ranked, 1):
            row.update({"rank": rank, "selected": row["target_id"] in neighbors[source]})
            sources.append(row)

    return {
        "neighbors": neighbors,
        "sources": sources,
        "training_speakers": list(speakers),
        "kind": "training_observed_directed_substitutions",
        "max_neighbors": max_neighbors,
        "min_count": min_count,
        "selected_training_records": selected_records,
        "diagnostic_tokens_by_speaker": dict(exposure),
        "balanced_rate_definition": "mean_s(count_s(source,target) / stable_diagnostic_tokens_s)",
        "unobserved_source_policy": "empty; acoustic candidates supplied separately",
        "smoothing": "none; every edge has observed training evidence",
    }
