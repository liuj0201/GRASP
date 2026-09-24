"""Run the full and sparse graph scorers without audio, models, or datasets.

From the repository root, after ``python -m pip install -e .``:

    python examples/synthetic_graph.py

The logits and reference phones below are invented. This is an API example,
not a trained detector, paper-result reproduction, or clinical assessment.
"""

from __future__ import annotations

import numpy as np

from da_cf_gop.candidate_selection import select_candidates
from da_cf_gop.dual_decode import decode
from da_cf_gop.dual_prior import generic_prior
from da_cf_gop.phonology import ARPABET_39, CTC_ID_TO_PHONE, PHONE_TO_CTC_ID


def synthetic_log_probs() -> np.ndarray:
    """Create ten frames with blank, K/G, AE/EH, and T/D support."""
    logits = np.full((10, len(ARPABET_39) + 1), -9.0)
    logits[:, 0] = 6.0
    for frames, preferred, alternative in (
        ((1, 2), "K", "G"),
        ((4, 5), "AE", "EH"),
        ((7, 8), "T", "D"),
    ):
        for frame in frames:
            logits[frame, 0] = 1.0
            logits[frame, PHONE_TO_CTC_ID[preferred]] = 6.0
            logits[frame, PHONE_TO_CTC_ID[alternative]] = 3.0
    # Stable log-softmax: the decoder consumes log probabilities, not logits.
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def main() -> None:
    canonical = ("K", "AE", "T")
    canonical_ids = [PHONE_TO_CTC_ID[phone] for phone in canonical]
    acceptable = [[phone_id] for phone_id in canonical_ids]
    log_probs = synthetic_log_probs()
    selection = select_candidates(
        log_probs,
        acceptable,
        policy="frame_topk",
        top_k=1,
        posterior_threshold=0.01,
    )
    # This fixed generic prior uses no training examples or patient labels.
    prior = generic_prior()
    full = decode(log_probs, acceptable, prior, canonical_ids=canonical_ids)
    sparse = decode(
        log_probs,
        acceptable,
        prior,
        canonical_ids=canonical_ids,
        substitution_candidates=selection.substitution_phone_ids,
        insertion_candidates=selection.insertion_phone_ids,
    )

    print("Synthetic API demo: no audio, trained detector, or clinical claims.")
    print("Reference phones:", " ".join(canonical))
    print("Active candidate phones:", " ".join(
        CTC_ID_TO_PHONE[phone_id] for phone_id in selection.active_phone_ids
    ))
    print("SUB candidates per anchor:", selection.substitution_counts)
    print("INS candidates per gap:", selection.insertion_counts)
    print(f"Retained nonblank acoustic mass: {selection.retained_acoustic_mass:.6f}")
    print("This mass statistic is not CTC path coverage or error recall.")
    print()
    print(f"{'Graph':<8} {'States':>8} {'CTC states':>11} {'Arcs':>8} "
          f"{'Log partition':>15} {'Viterbi score':>15}")
    for name, result in (("full", full), ("sparse", sparse)):
        print(f"{name:<8} {result['graph_state_count']:>8} "
              f"{result['ctc_state_count']:>11} {result['graph_arc_count']:>8} "
              f"{result['log_partition']:>15.6f} {result['viterbi_score']:>15.6f}")
    print()
    print("Sparse graph token scores (natural-log odds of OK versus SUB + DEL):")
    print(f"{'Phone':<6} {'Log odds':>10} {'P(OK)':>10} {'P(SUB)':>10} {'P(DEL)':>10}")
    for phone, score, posterior in zip(
        canonical, sparse["token_log_odds"], sparse["token_posteriors"]
    ):
        print(f"{phone:<6} {score:>10.6f} "
              f"{posterior[0]:>10.6f} {posterior[1]:>10.6f} {posterior[2]:>10.6f}")


if __name__ == "__main__":
    main()
