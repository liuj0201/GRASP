"""Summarize saved local calls without making any model calls."""
from pathlib import Path
import json
import statistics
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/evaluation/llm_three_tasks_20260921'

def quantile(values, q):
    values = sorted(values)
    position = (len(values)-1)*q
    lower = int(position)
    upper = min(lower+1, len(values)-1)
    return values[lower] + (values[upper]-values[lower]) * (position-lower)

def summarize(rows):
    n = len(rows)
    passed = sum(r['evaluation']['success'] for r in rows)
    checks = {}
    for key in sorted({k for r in rows for k in r['evaluation']['checks']}):
        values = [r['evaluation']['checks'][key] for r in rows if key in r['evaluation']['checks']]
        if all(isinstance(value, bool) for value in values):
            checks[key] = {'passed': sum(values), 'n': len(values)}
    grouped = {}
    for row in rows:
        meta = row['metadata']
        key = meta.get('document', meta.get('group_id', row['id']))
        grouped.setdefault(key, []).append(int(row['evaluation']['success']))
    counts = np.array([[sum(v), len(v)] for v in grouped.values()])
    rng = np.random.default_rng(20260921)
    draws = counts[rng.integers(0, len(counts), size=(2000, len(counts)))].sum(axis=1)
    interval = np.quantile(draws[:, 0] / draws[:, 1], [.025, .975]).tolist()
    strata = {}
    for row in rows:
        if row['task'] == 'phrase':
            key = 'answerable' if row['evaluation']['answerable'] else 'no_gold_span'
        elif row['task'] == 'summary':
            key = row['metadata']['condition']
        else:
            key = str(row['metadata']['target_count']) + '_target_words'
        strata.setdefault(key, {'success': 0, 'n': 0})
        strata[key]['success'] += int(row['evaluation']['success'])
        strata[key]['n'] += 1
    paired = []
    lookup = {r['id']: r for r in rows}
    for row in rows:
        parent = row['metadata'].get('paired_with')
        if parent and parent in lookup:
            paired.append(bool(row['evaluation']['success'] and lookup[parent]['evaluation']['success']))
    phrase_hard = None
    if rows[0]['task'] == 'phrase':
        phrase_hard = {'success': sum(all(r['evaluation']['checks'][key] for key in
            ('original_contiguous', 'target_occurrence_covered', 'length_2_to_6')) for r in rows), 'n': n}
    return {'n': n, 'success': passed, 'rate': passed/n,
            'exploratory_group_bootstrap_95ci': interval, 'resampling_groups': len(grouped),
            'strata': strata,
            'phrase_contiguous_target_length': phrase_hard,
            'paired_both_correct': {'success': sum(paired), 'n': len(paired)} if paired else None,
            'p50_seconds': statistics.median(r['latency_seconds'] for r in rows),
            'p95_seconds': quantile([r['latency_seconds'] for r in rows], .95),
            'checks': checks,
            'finish_reasons': dict(Counter(r['finish_reason'] for r in rows)),
            'baselines': {key: {'success': sum(r['baselines'][key]['evaluation']['success'] for r in rows), 'n': n}
                          for key in rows[0].get('baselines', {})}}

def main():
    summary = {}
    lines = ['# Qwen 1.5B 三功能本地可行性试验', '',
             '模型：Qwen2.5-1.5B-Instruct-Q4_K_M；llama.cpp，本机 CPU 两线程，temperature=0，单次生成，最多128输出token。',
             '无微调、无付费API、无云端裁判。结果来自保存的原始输出，不包含模板回退后的替代答案。', '',
             '## 结果', '', '| 批次 | 功能 | 原始输出通过 | P50 秒 | P95 秒 |', '|---|---|---:|---:|---:|']
    all_rows = {}
    for path in sorted(OUT.glob('*/outputs.jsonl'), key=lambda p: (p.parent.name != 'test_final', p.parent.name)):
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if not rows:
            continue
        label = path.parent.name
        all_rows[label] = rows
        summary[label] = {}
        for task in ('phrase', 'summary', 'sentence'):
            selected = [r for r in rows if r['task'] == task]
            if selected:
                result = summarize(selected)
                summary[label][task] = result
                lines.append(f"| {label} | {task} | {result['success']}/{result['n']} ({result['rate']:.1%}) | {result['p50_seconds']:.3f} | {result['p95_seconds']:.3f} |")
    final = summary.get('test_final', {})
    if sum(result['n'] for result in final.values()) == 300:
        phrase, aggregate, sentence = (final[name] for name in ('phrase', 'summary', 'sentence'))
        lines += ['', '## 本轮结论', '',
                  f"- 原句片段：{phrase['success']}/100 严格通过；有答案 {phrase['strata']['answerable']['success']}/80，无答案 {phrase['strata']['no_gold_span']['success']}/20。当前提取和弃答能力不足。",
                  f"- 共性统计转述：{aggregate['success']}/100 通过；这是代码归并后交给模型的受控转述，总结模板为100/100。本轮没有检验模型独立归类多个错误。",
                  f"- 新练习句：{sentence['success']}/100 满足目标词、长度、输出格式和不复制原句；单目标 {sentence['strata']['1_target_words']['success']}/50，双目标 {sentence['strata']['2_target_words']['success']}/50。自然度和教学价值未自动证实。",
                  '- 新句中的实际反例：sentence_test_097 输出 “The tree casts a shadow on the sun.”，硬约束通过但语义不合理。因此硬约束通过率不能称为自然度或实际可用率。',
                  '- 三项本地推理均已跑通，但当前结果没有建立相对模板或规则的优势。保存原始失败，不以模板回退结果替代模型结果。', '']
    lines += ['', '## 指标含义与边界', '',
              '- phrase：连续原文、指定目标、2–6词、命中既有句法树中的允许短语集合；无合法片段时应返回NONE。句法标注是结构依据，不是患者练习效果。',
              '- summary：根据已计算的单一目标统计写出指定形式的一句总结。通过表示受控表达与统计一致，不表示模型自主发现了音系规律。',
              '- sentence：dev_v1_sentence 的成功指标包含限定词表与受控语法；在查看测试输出前，dev_v2/dev_v3_sentence/test_final 改为短提示，主指标仅为目标词、5–10词、单句纯文本格式及不复制原句。原语法仅作辅助覆盖检查，超出该子集不等于英语错误。两版成功率不可当作同口径改善。',
              '- 开发批用于诊断，不是独立正式结果。提示修订若存在，分别保存批次；正式测试一次冻结运行。',
              '- 不将相同基础输入的变体计为新增患者。数据为已有句法语料/合成任务输入，本轮没有采集患者数据。',
              '- 固定窗口、总结模板与句子规则基线分别报告。本轮快速试验未加入完整句法解析器基线。', '']
    for label, rows in all_rows.items():
        lines += [f'## {label} 明细', '']
        for task, result in summary[label].items():
            lines += [f'### {task}', '', '```json', json.dumps(result, ensure_ascii=False, indent=2), '```', '']
            selected = [r for r in rows if r['task'] == task]
            examples = ([r for r in selected if not r['evaluation']['success']][:5]
                        + [r for r in selected if r['evaluation']['success']][:2])
            for row in examples:
                lines += [f"**{row['id']} — {'通过' if row['evaluation']['success'] else '未通过'}**", '',
                          '输入：`' + json.dumps(row['input'], ensure_ascii=False).replace('`','') + '`', '',
                          '```text', row['raw_text'], '```',
                          '检查：`' + json.dumps(row['evaluation'], ensure_ascii=False).replace('`','') + '`', '']
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'RESULTS_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
