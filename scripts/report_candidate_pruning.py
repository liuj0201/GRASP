"""Write a factual Chinese experiment report from frozen pruning outputs."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/candidate_pruning_20260909"
NAMES = {"full_graph_da": "完整图 + DA", "acoustic_top1_da": "声学 top-1 剪枝 + DA",
         "acoustic_top2_da": "声学 top-2 剪枝 + DA", "learned_top1_da": "声学 top-1 + 训练混淆剪枝 + DA"}
SHORT = {"full_graph_da": "Full", "acoustic_top1_da": "Acoustic top-1",
         "acoustic_top2_da": "Acoustic top-2", "learned_top1_da": "Top-1 + learned"}
BENCH = {"full_graph_da": "full_compact", "acoustic_top1_da": "acoustic_top1",
         "acoustic_top2_da": "acoustic_top2", "learned_top1_da": "hybrid_top1_train4"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return f"{100 * value:.2f}"


def main():
    assert (OUT / "predictions_complete.json").exists()
    reports = {cohort: read(OUT / cohort / "metrics.json") for cohort in ("primary5", "sensitivity7")}
    pilot = read(OUT / "pilot_learned.json")
    methods = list(NAMES)
    indexed = {cohort: {r["method"]: r for r in report["results"]} for cohort, report in reports.items()}
    lines = [
        "# 有依据的候选图剪枝：实现与实验记录", "", "日期：2026-09-09。所有数值由本目录冻结预测及独立 CPU 计时生成。", "",
        "## 这次实现了什么", "",
        "本轮 CF 回到用户最初提出的含义：在图推断前减少替代和插入候选。历史 CF 的局部先验重加权没有用于这些新方案。声学模型、39 个 ARPABET 音素加 CTC blank 的输出系统、可接受发音图、删除定义和四项 Graph+DA 评分特征保持一致。", "",
        "候选来自两类可追溯证据：当前录音的原始音素帧概率，以及本折训练患者的转录推导替代标注。没有根据直觉编写 K→T 等混淆规则，也没有把已有文献中测试患者的具体发音经验搬进来。", "",
        "## 候选怎么选", "",
        "1. 根据参考文本已有的词典和可接受发音策略，得到每个位置的正确候选集合。所有正确候选始终保留；患者常见错误不会被改成正确发音。",
        "2. 冻结的声学模型仍为每一帧输出全部 39 个音素及 blank 的概率。声学剪枝在每帧非 blank 音素中取概率最高的 1 个或 2 个，并要求其原始概率至少为 0.01；把所有帧留下的音素合并。它不是只保留一个最终转录，也不需要先找音素边界。",
        "3. 学习版本从本折训练患者的稳定、可诊断替代标签统计有向边。各患者按其稳定且可诊断音素位置数归一化贡献，再按均衡计数排序，每个参考音素最多加入 4 个有实际观察支持的替代音。4 是预先固定的算法预算；未观察到的关系不补齐、不平滑造边。",
        "4. 每个位置的替代候选等于“声学候选 ∪ 训练邻居”，再去掉该位置的正确音素；插入候选只用声学集合。正确发音、独立的删除 epsilon 事件，以及 CTC 的 blank/repeat 规则保留。blank 不是删除音素。",
        "5. 删除其他分支，并移除无效节点及 CTC 状态，再对剩余图做求和与最大路径推断。保留分支沿用原弧权重，声学概率和剩余替代分布都不重新归一化；最终事件后验仍通过图总概率正常归一化。",
        "6. 从新图获得原有四项证据：求和后验 GOP、最大边际 GOP、SUB/OK 和 DEL/OK 的对数比；用相同的 DA 候选模型、开发患者选择和校准流程拟合评分。", "",
        "形式化表示：若 A(X) 是整段录音的声学候选，B_train(p) 是训练中观察到的 p 的替代邻居，OK_i 是第 i 个位置的正确集合，则 SUB_i = (A(X) ∪ ⋃_{p∈OK_i} B_train(p)) \\ OK_i。仅声学方案令 B_train 为空。", "",
        "这个版本的声学集合是整段录音的并集。因此长句可能留下更多候选，不能声称每个位置始终只有固定数量，也尚未加入新的上下文连读规则。", "",
        "剪枝后得到的是保留子图内的事件后验，是对完整图评分的近似；它不等于完整图的精确后验。声学集合保留的帧概率质量、保留的整图路径概率质量和真实错音保留率，是三个不同量。资源 pilot 的路径质量比只用于离线核查，选候选时没有先跑完整图。", "",
        "## 数据与评价协议", "",
        "- 沿用五患者主实验和七患者敏感性实验的患者划分及固定质量排除，共 1,626 段录音。两组有重叠，不能视为两次独立验证。",
        "- 训练样本的混淆表额外排除它自己的患者；开发和测试只用外层训练患者的表。相同候选图可共享无标签推断缓存，但每折的候选来源保持独立记录。",
        "- 新图规则在读取本轮测试指标前固定；全部外层预测先写出并记录 SHA256，随后才统一加入测试标签。历史外层结果此前已经被查看，所以本轮属于回顾性探索。",
        "- 所有方法在同样的 6,820 / 8,353 个可评价位置上计算指标。真实替代音不在候选内的位置仍参与检测评价，没有被剔除。",
        "- 训练和评价标签沿用现有 PHN 转录推导标签，不是本轮新增的临床人工错误标注。", "",
        "本轮针对二元音素错误检测、真实替代候选保留及图计算资源；没有重新训练具体替代/删除诊断头，也没有评估新的 LLM 反馈效果。", "",
        "## 检测结果", "", "AP、F1 和替代候选保留率单位均为百分比。AP/F1 为患者宏平均；替代候选保留率为稳定替代位置的合并比例。", "",
        "| 方法 | 五人 AP | 五人 F1 | 五人替代保留 | 七人 AP | 七人 F1 | 七人替代保留 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    table_rows = []
    for method in methods:
        a, b = indexed["primary5"][method], indexed["sensitivity7"][method]
        values = [a["macro"]["auprc"], a["macro"]["f1"], a["candidate_retention"]["retention_rate"],
                  b["macro"]["auprc"], b["macro"]["f1"], b["candidate_retention"]["retention_rate"]]
        lines.append("| " + NAMES[method] + " | " + " | ".join(map(pct, values)) + " |")
        table_rows.append([method, *values])
    with (OUT / "detection_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "AP5", "F1_5", "SUB_retention5", "AP7", "F1_7", "SUB_retention7"])
        writer.writerows(table_rows)
    lines += ["", "这里的完整图对照是原有 Graph+DA 的四特征版本，不是后来融合 PP-AF 等分数的 ensemble。二者不能用不同输入的数字混称同一个完整方案。", "",
              "候选保留不等于错误检出：真实替代音被剪掉后，系统仍可能通过删除或其他证据判断“有错”，但不能据此认为真实替代被正确解释。每位患者、漏掉的替代位置及配对统计见各队列 metrics.json 和 candidate_misses.jsonl.gz。", "",
              "## 如何理解这批结果", "",
              "声学 top-2 是本次比较中七人 AP 最高的剪枝配置，五人 AP 也略高于完整图；但 F1 分别从 26.39/32.21 变为 26.32/31.68，不能说所有检测指标都提高。七人 AP 的配对提升为 0.83 个百分点，患者 bootstrap 95% 区间为 [0.18, 1.72]，未经多重比较校正的 sign-flip p=0.046875；本轮比较了多个配置且属于回顾性探索，不据此作确认性显著提升结论。", "",
              "训练混淆表确实补回了部分纯声学 top-1 漏掉的真实替代：五人从 85/127 提到 102/127，七人从 90/140 提到 113/140。它把七人 AP 从 35.67 恢复到 36.38，与完整图基本相同；五人 AP 为 27.29，高于完整图的 26.69。频率在这里用来选边，未变成旧 CF 的先验重加权。", "",
              "当前最明显的不足是具体错音覆盖。学习版仍遗漏五人 25/127、七人 27/140 个稳定真实替代；这些位置全部继续参与检测指标。在七人组的 27 个遗漏位置中，23 个没有被检测为错误。因此这轮结果支持继续研究有依据的候选剪枝及其效率价值，但不足以将这个预算直接定为具体音素反馈的最终配置。", "",
              "这些候选遗漏不等于同等数量的新增漏检。冻结预测的事后配对核查显示，七人组这 27 个位置与完整图的二元判定完全相同，两者都漏检 23 个；五人组的 25 个位置新增 2 个漏检、恢复 1 个检测，净新增 1 个漏检。这里报告的是当前评分器的检测行为，同时保留“真实替代音已经不在图中”这一结构限制。详见 missing_candidate_paired_detection.json。", "",
              "下一阶段应在训练/开发患者上选择满足替代覆盖要求的声学预算和扩展规则，再冻结评价；本轮没有在看到这些测试遗漏后新增或调优第五种配置。", "",
              "## CPU 和内存", "",
              "在 24 条按患者及参考长度选定的录音上，单进程、两线程、Numba 预热后，每方法测两次，并轮换/反转顺序。下表计入候选选择、图编译和求和/最大路径推断；不含声学模型、缓存加载和校准器。", "",
              "学习版的计时使用七患者实验中该录音对应的外层训练表；没有用包含该测试患者的混淆表。五患者训练表的耗时未单独测量。", "",
              "计时样本的参考长度为 1–40 个音素，未覆盖母集中 30 条超过 40 个音素的长句，因此它是范围有限的资源 pilot。准确率比较仍包含既定母集中的这些长句。", "",
              "| 方法 | 平均 SUB 候选/位置 | 图耗时/完整压缩图 | 图耗时均值 (s) | 前向 history 均值 (MiB) |",
              "|---|---:|---:|---:|---:|"]
    for method in methods:
        row = pilot["summary"][BENCH[method]]
        lines.append(f"| {NAMES[method]} | {row['mean_substitution_candidates']['mean']:.2f} | "
            f"{100 * row['total_time_ratio_to_full_compact']:.1f}% | {row['mean_total_s']:.3f} | "
            f"{row['sum_forward_bytes']['mean'] / 2**20:.2f} |")
    lines += ["", "完整候选也使用相同的图压缩步骤作为资源对照，避免把通用代码优化算成候选选择的收益。另有原始未压缩完整图对照，见 pilot_learned.json。", "",
              "本轮最后几条录音上，各方法共同出现绝对耗时上升，因此采用同一轮配对比较，不混用两轮 pilot 的绝对耗时。所有计时均保留，没有挑选较快的重跑结果。", "",
              "学习版在这 24 条录音上平均保留 97.06% 的完整模型图概率质量；它与七人测试中 80.71% 的真实替代候选保留率不同。模型认为重要的路径，不一定覆盖人工转录推导出的真实错音，不能把两种比例统称为“保留的信息”。", "",
              "前向 history 是一个明确的动态规划数组，大小为 (T+1)×S×8 字节；不是进程峰值内存。当前编译阶段仍会生成临时未压缩数组。上述结果也不是整套网页工具或边缘设备的加速倍数：声学编码器仍计算原来的全音素概率。", "",
              "## 依据与可复现文件", "",
              "文献核查见 [literature_evidence_zh.md](literature_evidence_zh.md)。Mengistu & Rudzicz (2011) 确有有向替代观察，但其 TORGO 患者与本实验重叠，未将该表硬编码。其他研究的类别模式也未被扩展为未经核实的音素对。", "",
              "训练证据的稀疏性见 training_evidence_audit.json：50 个相互重叠的训练上下文中，45.03% 的“上下文×源音素”组合没有稳定替代观察；69.64% 的已选“上下文×方向边”仅有一次支持。示例训练组 M01+M04+M05 有 AH→IH 9 次/3 人、T→D 6 次/1 人。它们是可追溯的训练观察，不能转述成面向所有患者的替代概率。", "",
              "- configs/candidate_pruning_20260909.json：预设方案。",
              "- training_maps/：每个训练患者组合的候选及逐边原始证据。",
              "- fold_indices/：每折各患者使用哪个训练表、每段录音使用哪个图。",
              "- graph_jobs.jsonl.gz、graph_cache/：冻结候选及原始图结果。",
              "- primary5/、sensitivity7/：拟合模型、冻结预测、指标和遗漏位置。",
              "- full_graph_history_verification.json：新完整图与旧固定证据的逐值一致性。",
              "- final_integrity_audit.json：实际 12 折、50 个训练上下文和 53,571 条选择路由的患者隔离；完整图评分模型的选择、阈值、患者内排序和二分类决定与旧版一致。重拟合连续分数存在微小浮点差，未声称逐位一致。",
              "- validation.json：52 项聚焦测试通过，包括独立穷举的图推断检查。",
              "- pilot.json、pilot_learned.json：资源测量；mass99 仅为资源探索，不在本轮准确率比较中。", "",
              "```powershell",
              "python -m da_cf_gop.sparse_experiment prepare --output <new-directory>",
              "python -m da_cf_gop.sparse_experiment infer --output <new-directory> --workers 8",
              "python -m da_cf_gop.sparse_experiment predict --output <new-directory>",
              "python -m da_cf_gop.sparse_evaluation --output <new-directory>",
              "python scripts/benchmark_sparse_graph.py --include-learned --run-output <new-directory> --output <new-directory>/pilot_learned.json",
              "```", "",
              "正式论文和网页默认评分器未在这轮实验中替换。", ""]
    (OUT / "RESULTS_zh.md").write_text("\n".join(lines), encoding="utf-8")

    colors = ["#40516c", "#e49b28", "#2f89b5", "#008b76"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
    for method, color in zip(methods, colors):
        x = 100 * pilot["summary"][BENCH[method]]["total_time_ratio_to_full_compact"]
        y = 100 * indexed["sensitivity7"][method]["macro"]["auprc"]
        ax.scatter(x, y, color=color, s=65, label=SHORT[method], zorder=3)
    ax.set_title("Seven-patient cohort; graph CPU pilot on 24 recordings")
    ax.set_xlabel("Graph CPU time (% of compact full graph)")
    ax.set_ylabel("Speaker-macro AP (%)")
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(OUT / "accuracy_vs_graph_cpu.png", dpi=180)
    plt.close(fig)
    print(OUT / "RESULTS_zh.md")


if __name__ == "__main__":
    main()
