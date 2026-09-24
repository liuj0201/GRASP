"""Speaker-balanced edit priors for the dual-graph experiment.

Only explicitly supplied training rows enter this module. Acceptability is
external and frozen: observed frequent patient realizations cannot add OK arcs.
"""

from collections import defaultdict

import numpy as np

from .phonology import PHONE_TO_CTC_ID


def generic_prior(vocab_size=40, operation=(0.90, 0.07, 0.03), insertion=0.02):
    op = np.tile(np.asarray(operation, dtype=float), (vocab_size, 1))
    op /= op.sum(axis=1, keepdims=True)
    sub = np.ones((vocab_size, vocab_size), dtype=float)
    sub[:, 0] = 0
    np.fill_diagonal(sub, 0)
    sub /= sub.sum(axis=1, keepdims=True)
    ins = np.ones(vocab_size, dtype=float)
    ins[0] = 0
    ins /= ins.sum()
    return {"op_probs": op, "sub_probs": sub, "insertion_prob": float(insertion),
            "insert_phone_probs": ins, "training_speakers": [], "kind": "generic"}


def estimate_error_prior(rows, *, training_speakers, alpha=20.0, effective_tokens=1000.0,
                         min_phone_tokens=20, min_phone_speakers=2, max_insertions=2):
    """Separate error occurrence from conditional substitution identity.

    Each speaker contributes a total of effective_tokens before averaging.
    Counts from ambiguous/non-diagnostic tokens and gaps are not learned.
    Rare source phones fall back to global operation and target distributions.
    """
    speakers = tuple(sorted(training_speakers))
    if not speakers or alpha <= 0:
        raise ValueError("nonempty training speakers and positive smoothing required")
    grouped = defaultdict(list)
    for row in rows:
        if row["speaker"] not in speakers:
            raise ValueError("non-training speaker passed into error-prior estimation")
        grouped[row["speaker"]].append(row)
    base = generic_prior()
    v = len(base["op_probs"])
    op = np.zeros((v, 3))
    sub = np.zeros((v, v))
    ins = np.zeros(v)
    insertion_success = insertion_failure = 0.0
    raw_phone_counts = np.zeros(v)
    phone_speakers = np.zeros(v)
    contribution = {}
    for speaker in speakers:
        local_op = np.zeros_like(op)
        local_sub = np.zeros_like(sub)
        local_ins = np.zeros_like(ins)
        success = failure = gap_count = over_limit = 0
        for row in grouped[speaker]:
            labels = row["labels"]
            for token in labels["tokens"]:
                if not token["stable_event"] or not token["diagnostic"]:
                    continue
                p = PHONE_TO_CTC_ID[row["canonical_phones"][token["phone_index"]]]
                event = {"match": 0, "substitution": 1, "deletion": 2}[token["event_type"]]
                local_op[p, event] += 1
                if event == 1:
                    local_sub[p, PHONE_TO_CTC_ID[token["realized_phone"]]] += 1
            for gap in labels["gaps"]:
                if not gap["stable"] or not gap["diagnostic"]:
                    continue
                phones = gap["inserted_phones"]
                gap_count += 1
                success += min(len(phones), max_insertions)
                failure += len(phones) < max_insertions
                over_limit += len(phones) > max_insertions
                for phone in phones[:max_insertions]:
                    local_ins[PHONE_TO_CTC_ID[phone]] += 1
        n = float(local_op.sum())
        if n == 0:
            raise ValueError(f"no diagnostic training tokens for {speaker}")
        scale = effective_tokens / n / len(speakers)
        op += local_op * scale
        sub += local_sub * scale
        raw_phone_counts += local_op.sum(axis=1)
        phone_speakers += local_op.sum(axis=1) > 0
        if gap_count:
            gap_scale = effective_tokens / gap_count / len(speakers)
            insertion_success += success * gap_scale
            insertion_failure += failure * gap_scale
            ins += local_ins * gap_scale
        contribution[speaker] = {"stable_event_tokens": int(n), "stable_gaps": gap_count,
                                 "effective_token_weight": effective_tokens / len(speakers),
                                 "insertions_longer_than_cap": over_limit}
    global_op = op.sum(axis=0) + alpha * base["op_probs"][1]
    global_op /= global_op.sum()
    global_sub = sub.sum(axis=0)
    fallback = []
    for p in range(v):
        if raw_phone_counts[p] < min_phone_tokens or phone_speakers[p] < min_phone_speakers:
            base["op_probs"][p] = global_op
            targets = global_sub.copy()
            fallback.append(p)
        else:
            counts = op[p] + alpha * base["op_probs"][p]
            base["op_probs"][p] = counts / counts.sum()
            targets = sub[p].copy()
        targets[0] = targets[p] = 0
        targets += alpha * base["sub_probs"][p]
        base["sub_probs"][p] = targets / targets.sum()
    base["insertion_prob"] = (insertion_success + alpha * base["insertion_prob"]) / (
        insertion_success + insertion_failure + alpha)
    ins += alpha * base["insert_phone_probs"]
    base["insert_phone_probs"] = ins / ins.sum()
    base.update({"kind": "speaker_equal_learned", "training_speakers": list(speakers),
                 "alpha": alpha, "contribution": contribution,
                 "fallback_phone_ids": fallback, "raw_phone_counts": raw_phone_counts,
                 "raw_phone_speakers": phone_speakers})
    return base


def serializable_prior(prior):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in prior.items()}
