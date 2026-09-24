"""CPU graph-only timing on text-length-stratified recordings, without gold."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from da_cf_gop.candidate_selection import select_candidates
from da_cf_gop.dual_decode import decode
from da_cf_gop.dual_experiment import phone_ids
from da_cf_gop.dual_prior import generic_prior
from da_cf_gop.parikh_experiment import read_jsonl, write_json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/candidate_pruning_20260909/pilot.json"
STRATA = {"short": (1, 5), "medium": (6, 15), "long": (16, 40)}
METHODS = ("legacy_full", "full_compact", "acoustic_top1", "acoustic_top2", "mass99")
LEARNED_METHOD = "hybrid_top1_train4"
GRAPH_FIELDS = ("graph_state_count", "ctc_state_count", "graph_arc_count", "graph_array_bytes",
                "sum_forward_bytes", "max_history_trace_bytes", "log_partition")


def population_and_selection():
    evidence = {r["event"] for r in read_jsonl(ROOT / "artifacts/evaluation/final_v2/fixed_evidence_without_gold.jsonl.gz")}
    rows = [r for r in read_jsonl(ROOT / "artifacts/evaluation/dual_graph/dual_graph_v1/sensitivity7/acoustic_inputs.jsonl.gz")
            if r["event"] in evidence and r["event"] != "F04_Session2_0009"
            and 1 <= len(r["canonical_phones"]) <= 40]
    selected = []
    for speaker in sorted({r["speaker"] for r in rows}):
        for stratum, (low, high) in STRATA.items():
            own = sorted((r for r in rows if r["speaker"] == speaker and low <= len(r["canonical_phones"]) <= high),
                         key=lambda r: (len(r["canonical_phones"]), r["event"]))
            if own:
                selected.append({**own[len(own) // 2], "stratum": stratum})
    # Three additional fixed representatives balance lengths, without outcomes.
    for stratum, (low, high) in STRATA.items():
        remaining = sorted((r for r in rows if low <= len(r["canonical_phones"]) <= high
                            and r["event"] not in {s["event"] for s in selected}),
                           key=lambda r: (len(r["canonical_phones"]), r["event"]))
        if remaining:
            selected.append({**remaining[len(remaining) // 2], "stratum": stratum})
    return rows, selected


def load_log_probs(event):
    path = ROOT / "artifacts/cache/logits" / (hashlib.sha256(event.encode()).hexdigest()[:24] + ".npz")
    with np.load(path, allow_pickle=False) as data:
        assert str(data["utterance_id"].item()) == event
        logits = data["logits"].astype(np.float64)
    return logits - logsumexp(logits, axis=1, keepdims=True)


def load_training_context(speaker, run_output=OUT.parent):
    """Use only the outer fold's training patients for its test speaker."""
    directory = Path(run_output).resolve()
    fold_path = directory / "fold_indices" / f"sensitivity7_test_{speaker}.json"
    index = json.loads(fold_path.read_text())
    fold = index["fold"]
    context = index["speaker_context"][speaker]
    map_path = directory / "training_maps" / (context + ".json")
    training = json.loads(map_path.read_text())
    assert fold["test_patient"] == speaker
    assert set(training["training_speakers"]) == set(fold["training_patients"])
    assert speaker not in training["training_speakers"]
    assert fold["development_patient"] not in training["training_speakers"]
    assert training["max_neighbors"] == 4
    return training["neighbors"], {
        "fold": fold["fold"], "test_patient": speaker,
        "development_patient": fold["development_patient"],
        "speaker_context": context, "training_patients": training["training_speakers"],
        "map_kind": training["kind"], "max_neighbors": training["max_neighbors"],
        "fold_index": str(fold_path),
        "fold_index_sha256": hashlib.sha256(fold_path.read_bytes()).hexdigest(),
        "training_map": str(map_path),
        "training_map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
    }


def infer(method, lp, acceptable, canonical, prior, learned_neighbors=None):
    start = time.perf_counter()
    selection = None
    kwargs = {}
    if method == "full_compact":
        phones = tuple(range(1, lp.shape[1]))
        kwargs = {"substitution_candidates": [phones] * len(acceptable),
                  "insertion_candidates": [phones] * (len(acceptable) + 1)}
    elif method != "legacy_full":
        if method == LEARNED_METHOD and learned_neighbors is None:
            raise ValueError("Hybrid inference requires the outer-training neighbor map")
        selection = select_candidates(lp, acceptable, policy="mass_cover" if method == "mass99" else "frame_topk",
                                      top_k=1 if method in ("acoustic_top1", LEARNED_METHOD) else 2,
                                      posterior_threshold=.01, mass_coverage=.99,
                                      learned_neighbors=learned_neighbors if method == LEARNED_METHOD else None)
        kwargs = {"substitution_candidates": selection.substitution_phone_ids,
                  "insertion_candidates": selection.insertion_phone_ids}
    selected_at = time.perf_counter()
    result = decode(lp, acceptable, prior, canonical_ids=canonical, include_viterbi=True, **kwargs)
    end = time.perf_counter()
    return {
        "total_s": end - start,
        "selection_s": selected_at - start,
        "compile_and_decode_s": end - selected_at,
        "retained_acoustic_mass": selection.retained_acoustic_mass if selection else 1.0,
        "active_phone_count": len(selection.active_phone_ids) if selection else lp.shape[1] - 1,
        "mean_substitution_candidates": float(np.mean(selection.substitution_counts)) if selection else
            float(np.mean([lp.shape[1] - 1 - len(set(a)) for a in acceptable])),
        **{key: result[key] for key in GRAPH_FIELDS},
    }


def summarize(samples, methods=METHODS):
    summary = {}
    for method in methods:
        values = [np.mean([r["total_s"] for r in row["measurements"][method]]) for row in samples]
        baseline = [np.mean([r["total_s"] for r in row["measurements"]["full_compact"]]) for row in samples]
        summary[method] = {"mean_total_s": float(np.mean(values)), "median_total_s": float(np.median(values)),
                           "total_time_ratio_to_full_compact": float(np.sum(values) / np.sum(baseline)),
                           "median_event_speedup_vs_full_compact": float(np.median(np.asarray(baseline) / values))}
        for key in (*GRAPH_FIELDS[:-1], "retained_acoustic_mass", "active_phone_count", "mean_substitution_candidates"):
            data = [row["measurements"][method][0][key] for row in samples]
            summary[method][key] = {"mean": float(np.mean(data)), "median": float(np.median(data)),
                                    "min": float(np.min(data)), "max": float(np.max(data))}
    return summary


def verify_historical_evidence(selected, output):
    """Recompute three graph records; this is correctness QA, not timing."""
    events = {row["event"] for row in selected}
    evidence_path = ROOT / "artifacts/evaluation/final_v2/fixed_evidence_without_gold.jsonl.gz"
    evidence = {(row["event"], row["phone_index"]): row for row in read_jsonl(evidence_path)
                if row["event"] in events}
    checked = []
    for row in selected:
        lp = load_log_probs(row["event"])
        actual = decode(lp, [phone_ids(a) for a in row["acceptable_phones"]], generic_prior(),
                        canonical_ids=phone_ids(row["canonical_phones"]))
        expected = [evidence[row["event"], i] for i in range(len(row["canonical_phones"]))]
        differences = {}
        for result_key, stored_key in (("token_log_odds", "gop"),
                                       ("max_marginal_token_log_odds", "viterbi_gop"),
                                       ("token_posteriors", "event_posteriors"),
                                       ("substitution_posteriors", "substitution_posteriors")):
            reference = np.asarray([item[stored_key] for item in expected])
            np.testing.assert_allclose(actual[result_key], reference, rtol=0, atol=1e-8)
            differences[stored_key] = float(np.max(np.abs(actual[result_key] - reference)))
        checked.append({"event": row["event"], "n_reference_phones": len(expected),
                        "n_frames": len(lp), "max_absolute_difference": differences})
    write_json(output, {"passed": True, "purpose": "Correctness of reusing historical raw full-graph evidence; no timing claim",
                        "evidence_file": str(evidence_path.relative_to(ROOT)),
                        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                        "decoder_sha256": hashlib.sha256((ROOT / "src/da_cf_gop/dual_decode.py").read_bytes()).hexdigest(),
                        "samples": checked})
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-output", type=Path, default=OUT.parent,
                        help="Experiment directory containing fold_indices/ and training_maps/")
    parser.add_argument("--include-learned", action="store_true")
    parser.add_argument("--verify-history-only", action="store_true")
    args = parser.parse_args()
    threadpool_limits(limits=2)
    population, selected = population_and_selection()
    if args.verify_history_only:
        output = args.output or (args.run_output / "full_graph_history_verification.json")
        print(json.dumps(verify_historical_evidence(selected[:3], output)), flush=True)
        return
    selected = selected[:args.limit]
    methods = tuple(method for method in METHODS if method != "mass99") + (LEARNED_METHOD,) if args.include_learned else METHODS
    args.output = args.output or (args.run_output / ("pilot_learned.json" if args.include_learned else "pilot.json"))
    training_contexts = {speaker: load_training_context(speaker, args.run_output) for speaker in
                         sorted({row["speaker"] for row in selected})} if args.include_learned else {}
    prior = generic_prior()
    warm = load_log_probs(selected[0]["event"])[:20]
    decode(warm, [[1], [2]], prior)
    decode(warm, [[1], [2]], prior, substitution_candidates=[[3], []], insertion_candidates=[[]] * 3)
    metadata = {
        "selection_rule": "Per speaker median reference-length item within bins 1-5, 6-15, 16-40; plus one remaining global median in each bin; ties by event ID",
        "population_count": len(population), "selected_count": len(selected),
        "speaker_counts": dict(Counter(r["speaker"] for r in selected)),
        "stratum_counts": dict(Counter(r["stratum"] for r in selected)),
        "gold_or_accuracy_used_for_selection": False, "excluded_event": "F04_Session2_0009",
        "cpu_threads": 2, "process_count": 1, "platform": platform.platform(),
        "repetitions_per_event_method": 2,
        "order": "Rotate method order by event index, then reverse for second repetition",
        "include_viterbi": True, "logits_loading_and_normalization_excluded": True,
        "methods": methods, "resource_only_methods": ["mass99"] if "mass99" in methods else [],
        "learned_method": "Union acoustic top1 >=.01 with at most four observed substitutions per source phone from outer-training patients; SUB only" if args.include_learned else None,
        "training_map_loading_excluded": True,
        "run_output": str(args.run_output.resolve()),
        "training_contexts": {speaker: context[1] for speaker, context in training_contexts.items()},
        "acoustic_encoder_excluded": True, "numba_warmup_excluded": True,
        "selection_graph_compilation_and_sum_max_inference_included": True,
        "actual_rss_measured": False,
        "memory_fields": "Array byte counts; sum_forward_bytes only (T+1)*S float64 history; max_history_trace_bytes additionally three T*S int32 traces. Not peak RSS. Initial uncompressed compilation temporary arrays excluded from graph_array_bytes.",
        "scope": "Graph-only CPU pilot on cached CTC outputs; other host tasks may run; no edge-device or end-to-end latency claim.",
        "source_sha256": {name: hashlib.sha256((ROOT / "src/da_cf_gop" / name).read_bytes()).hexdigest()
                          for name in ("dual_decode.py", "candidate_selection.py")},
    }
    samples = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(selected):
        lp = load_log_probs(row["event"])
        acceptable = [phone_ids(a) for a in row["acceptable_phones"]]
        canonical = phone_ids(row["canonical_phones"])
        order = methods[index % len(methods):] + methods[:index % len(methods)]
        measurements = {method: [] for method in methods}
        neighbors, context = training_contexts.get(row["speaker"], (None, None))
        orders = [order, tuple(reversed(order))]
        for ordered_methods in orders:
            for method in ordered_methods:
                measurements[method].append(infer(method, lp, acceptable, canonical, prior, neighbors))
        full_z = measurements["legacy_full"][0]["log_partition"]
        for method in methods:
            for result in measurements[method]:
                result["retained_graph_partition_ratio"] = float(np.exp(result["log_partition"] - full_z))
        assert abs(measurements["full_compact"][0]["log_partition"] - full_z) < 1e-8
        samples.append({"event": row["event"], "speaker": row["speaker"], "stratum": row["stratum"],
                        "n_reference_phones": len(canonical), "n_frames": len(lp), "orders": orders,
                        "learned_source_context": context, "measurements": measurements})
        write_json(args.output, {**metadata, "completed_count": len(samples), "summary": summarize(samples, methods), "samples": samples})
        print(json.dumps({"event": row["event"], "n": len(canonical), "frames": len(lp),
                          "mean_seconds": {method: round(float(np.mean([r["total_s"] for r in measurements[method]])), 4)
                                           for method in methods}}), flush=True)


if __name__ == "__main__":
    main()
