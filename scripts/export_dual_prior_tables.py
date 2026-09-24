"""Export fitted dual-graph base error priors without refitting or scoring.

Usage (from dysarthria/code):
  python scripts/export_dual_prior_tables.py
  python scripts/export_dual_prior_tables.py --cohort primary5

Outputs are <root>/<cohort>/candidate_prior_tables/*.csv plus README.md and
export_summary.json. Rerun after more folds finish to refresh a partial export.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))
from da_cf_gop.phonology import PHONE_TO_CTC_ID  # noqa: E402

ID_TO_PHONE = {value: key for key, value in PHONE_TO_CTC_ID.items()}
EDGE_COLUMNS = ["fold", "train_speakers", "canonical_phone", "operation", "alternative_phone",
                "operation_probability", "conditional_alternative_probability", "joint_prior_probability",
                "raw_train_count", "fallback"]
INSERT_COLUMNS = ["fold", "train_speakers", "operation", "alternative_phone", "operation_probability",
                  "conditional_alternative_probability", "joint_prior_probability", "raw_train_count", "fallback"]
RATE_COLUMNS = ["fold", "train_speakers", "insertion_probability", "stop_probability", "maximum_insertions_per_gap",
                "raw_training_gaps", "raw_capped_insertion_successes", "raw_insertion_stop_count",
                "raw_gaps_over_cap", "fallback"]


def jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def training_counts(rows, training_speakers, max_insertions=2):
    """Count observed training events, never development/test events or predictions."""
    allowed_speakers = set(training_speakers)
    edges, insertion, source_totals = Counter(), Counter(), Counter()
    gaps = successes = stops = over_cap = 0
    for row in rows:
        if row["speaker"] not in allowed_speakers or row["quality_exclusion"] is not None:
            continue
        for token in row["labels"]["tokens"]:
            if not token["stable_event"] or not token["diagnostic"]:
                continue
            canonical = row["canonical_phones"][token["phone_index"]]
            operation = {"match": "OK", "substitution": "SUB", "deletion": "DEL"}[token["event_type"]]
            alternative = token["realized_phone"] if operation == "SUB" else (canonical if operation == "OK" else "<DEL>")
            edges[(canonical, operation, alternative)] += 1
            source_totals[canonical] += 1
        for gap in row["labels"]["gaps"]:
            if not gap["stable"] or not gap["diagnostic"]:
                continue
            phones = gap["inserted_phones"]
            gaps += 1
            successes += min(len(phones), max_insertions)
            stops += len(phones) < max_insertions
            over_cap += len(phones) > max_insertions
            insertion.update(phones[:max_insertions])
    return {
        "edges": edges, "insertion": insertion, "source_totals": source_totals,
        "raw_training_gaps": gaps, "raw_capped_insertion_successes": successes,
        "raw_insertion_stop_count": stops, "raw_gaps_over_cap": over_cap,
    }


def fit_tables(fit, counts=None, max_insertions=2):
    prior = fit["prior"]
    training = sorted(prior["training_speakers"])
    if training != sorted(fit["fold"]["training_patients"]):
        raise ValueError("Fit prior speakers do not match the frozen training fold")
    if set(training) & {fit["fold"]["test_patient"], fit["fold"]["development_patient"]}:
        raise ValueError("Training speakers overlap development/test patients")
    fold = fit["fold"]["fold"]
    speakers = ";".join(training)
    fallback_ids = set(prior.get("fallback_phone_ids", []))
    edges = []
    for p in sorted(ID_TO_PHONE):
        canonical = ID_TO_PHONE[p]
        if counts is not None:
            observed = counts["source_totals"][canonical]
            if observed != prior["raw_phone_counts"][p]:
                raise ValueError(f"{fold}/{canonical}: training counts differ from saved prior ({observed})")
        fallback = "global" if p in fallback_ids else "phone_specific"
        for operation, index in (("OK", 0), ("SUB", 1), ("DEL", 2)):
            operation_probability = float(prior["op_probs"][p][index])
            alternatives = [(canonical, 1.0)] if operation == "OK" else (
                [("<DEL>", 1.0)] if operation == "DEL" else
                [(ID_TO_PHONE[q], float(prior["sub_probs"][p][q])) for q in sorted(ID_TO_PHONE) if q != p]
            )
            for alternative, conditional in alternatives:
                if operation_probability <= 0 or conditional <= 0:
                    raise ValueError(f"{fold}/{canonical}/{operation}/{alternative}: non-positive base support")
                edges.append({
                    "fold": fold, "train_speakers": speakers, "canonical_phone": canonical,
                    "operation": operation, "alternative_phone": alternative,
                    "operation_probability": operation_probability,
                    "conditional_alternative_probability": conditional,
                    "joint_prior_probability": operation_probability * conditional,
                    "raw_train_count": counts["edges"][(canonical, operation, alternative)] if counts is not None else "",
                    "fallback": fallback,
                })
    rate = float(prior["insertion_prob"])
    insertion = []
    for q in sorted(ID_TO_PHONE):
        conditional = float(prior["insert_phone_probs"][q])
        insertion.append({
            "fold": fold, "train_speakers": speakers, "operation": "INS", "alternative_phone": ID_TO_PHONE[q],
            "operation_probability": rate, "conditional_alternative_probability": conditional,
            "joint_prior_probability": rate * conditional,
            "raw_train_count": counts["insertion"][ID_TO_PHONE[q]] if counts is not None else "",
            "fallback": "global",
        })
    rate_row = {
        "fold": fold, "train_speakers": speakers, "insertion_probability": rate, "stop_probability": 1 - rate,
        "maximum_insertions_per_gap": max_insertions, "fallback": "global",
        **{name: counts[name] if counts is not None else "" for name in RATE_COLUMNS if name.startswith("raw_")},
    }
    return edges, insertion, rate_row


def write_csv(path: Path, columns, rows):
    # UTF-8 BOM lets Windows Excel display phone/metadata text without guessing.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export_cohort(directory: Path):
    fit_files = sorted((directory / "folds").glob("*.fit.json"))
    label_path = directory / "evaluation_labels.jsonl.gz"
    edges, insertion, rates, exported = [], [], [], []
    for fit_path in fit_files:
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        counts = training_counts(jsonl(label_path), fit["prior"]["training_speakers"]) if label_path.is_file() else None
        e, i, r = fit_tables(fit, counts)
        edges.extend(e)
        insertion.extend(i)
        rates.append(r)
        exported.append({
            "fold": fit["fold"]["fold"], "source_fit": str(fit_path.resolve()),
            "training_speakers": fit["prior"]["training_speakers"],
            "alpha": fit["prior"].get("alpha"),
            "selected_edit_scale_not_applied_in_tables": fit["selection"]["edit_scale"],
            "raw_edge_counts_available": counts is not None,
            "raw_counts_match_saved_source_phone_totals": counts is not None,
        })
    output = directory / "candidate_prior_tables"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "directed_phone_edges.csv", EDGE_COLUMNS, edges)
    write_csv(output / "insertion_phone_distribution.csv", INSERT_COLUMNS, insertion)
    write_csv(output / "insertion_rates.csv", RATE_COLUMNS, rates)
    folds_path = directory / "folds.json"
    frozen_folds = json.loads(folds_path.read_text(encoding="utf-8")) if folds_path.is_file() else None
    # Existing protocol stores folds as a list; no inference is made if absent.
    expected = len(frozen_folds) if isinstance(frozen_folds, list) else None
    summary = {
        "schema_version": "dual-graph.prior-table-export.v1", "cohort": directory.name,
        "exported_fold_count": len(exported), "expected_fold_count": expected,
        "export_is_partial": expected is None or len(exported) != expected,
        "edge_rows": len(edges), "insertion_rows": len(insertion), "folds": exported,
        "raw_counts_source": str(label_path.resolve()) if label_path.is_file() else None,
        "probability_source": "Saved fitted training priors; no refit, score computation, or selection performed.",
        "scope": "Base error prior before utterance-specific removal of all G_acc acceptable SUB phones and renormalization.",
    }
    (output / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# 候选错误图的训练先验表\n\n"
        "这些表只导出已经保存的 fold 训练先验，不重新训练、不计算性能、不选择参数。\n\n"
        "- `directed_phone_edges.csv`：每个 fold、每个 canonical phone 有 1 条 OK、38 条 SUB、1 条 DEL；不包含 CTC blank。\n"
        "- `insertion_phone_distribution.csv`：39 个 INS 音素及全局插入概率。\n"
        "- `insertion_rates.csv`：全局插入/停止概率及真实训练间隙计数。\n"
        "- `export_summary.json`：来源、训练说话人、已完成 fold 数，及当前导出是否仍为部分结果。\n\n"
        "这是基础错误先验，不是某句话的最终解码图。实际解码会把该句 G_acc 中的所有正常替代音从 SUB 候选中排除，"
        "再对剩余 SUB 候选重新归一化；OK 的权重也会分配给正常变体。`<DEL>` 表示删除事件，不是 CTC blank。\n\n"
        "`operation_probability` 是 OK/SUB/DEL（或插入）概率；`conditional_alternative_probability` 是发生该操作后"
        "实现为候选音的概率；两者乘积是 `joint_prior_probability`。表中概率尚未应用开发集选出的 edit_scale。\n\n"
        "`raw_train_count` 是该条有向事件在训练患者中的真实原始次数，不是目标音总次数，也不是平滑后的有效计数。"
        "只统计与原训练器相同的无质量排除、稳定且可诊断事件；开发和测试患者不参与。"
        "原训练器按说话人等权再平滑，因此不能简单把 raw_train_count 除以行总数复原表中的概率。"
        "INSERT 只计每个稳定、可诊断间隙的前 2 个音，与训练器一致；超过上限的间隙另记。"
        "如果冻结标签文件不可用，计数留空而非猜测。\n\n"
        "`fallback=global` 表示该音训练样本或说话人数不足，使用全局分布回退；`phone_specific` 表示音素级平滑先验。"
        "插入分布在首版始终是全局分布。\n\n"
        "复跑：在 dysarthria/code 下执行 `python scripts/export_dual_prior_tables.py`，或加 `--cohort primary5`。"
        "更多 fold 完成后重跑会更新导出；不会修改原 fit.json。\n",
        encoding="utf-8",
    )
    print(f"{directory.name}: exported {len(exported)} folds, {len(edges)} directed edges -> {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CODE_ROOT / "artifacts/evaluation/dual_graph/dual_graph_v1")
    parser.add_argument("--cohort", nargs="+", choices=["primary5", "sensitivity7"], default=["primary5", "sensitivity7"])
    args = parser.parse_args()
    for cohort in args.cohort:
        export_cohort(args.root / cohort)


if __name__ == "__main__":
    main()
