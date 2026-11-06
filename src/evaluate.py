"""汇总基线与 BERT 的测试集结果，输出对比表和错误分析。

用法：
    python -m src.evaluate            # 打印报告并写入 reports/evaluation_report.md
    python -m src.evaluate --examples 8
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics as mt
from src.config import BASELINE_DIR, BERT_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_stdout_utf8

MODEL_LABELS = {"baseline": "TF-IDF + Logistic Regression", "bert": "chinese-roberta-wwm-ext 微调"}
METRIC_ROWS = [
    ("Accuracy", "accuracy"),
    ("Precision（正面类）", "precision"),
    ("Recall（正面类）", "recall"),
    ("F1（正面类）", "f1"),
    ("Macro-F1", "macro_f1"),
]

NEGATION = ("不", "没", "无", "未", "别", "难")
TRANSITION = ("但", "可是", "不过", "然而", "虽然", "却", "倒是")
POS_WORDS = ("不错", "很好", "满意", "干净", "舒服", "超值", "热情", "推荐", "实惠", "安静", "方便")
NEG_WORDS = ("差", "脏", "旧", "失望", "吵", "贵", "破", "投诉", "糟糕", "难吃", "不行")
SARCASM = ("呵呵", "真是", "居然", "竟然", "号称", "所谓的", "太棒了", "好极了", "厉害")
CATEGORIES = ("转折结构", "否定表达", "褒贬混合", "反讽语气", "文本过短", "其他")


def load_block(name: str) -> dict | None:
    directory = BASELINE_DIR if name == "baseline" else BERT_DIR
    m = mt.read_json(directory / "metrics.json")
    preds = mt.read_jsonl(directory / "predictions.jsonl")
    if not m or not preds:
        return None
    return {"name": name, "metrics": m, "predictions": preds}


def classify_error(rec: dict) -> str:
    text = rec["text"]
    if any(w in text for w in TRANSITION):
        return "转折结构"
    if any(w in text for w in NEGATION):
        return "否定表达"
    if any(w in text for w in POS_WORDS) and any(w in text for w in NEG_WORDS):
        return "褒贬混合"
    if any(w in text for w in SARCASM):
        return "反讽语气"
    if len(text) <= 10:
        return "文本过短"
    return "其他"


def comparison_table(blocks: list[dict]) -> str:
    header = "| 指标 | " + " | ".join(MODEL_LABELS[b["name"]] for b in blocks) + " |"
    sep = "| --- |" + " --- |" * len(blocks)
    rows = [header, sep]
    for title, key in METRIC_ROWS:
        rows.append(f"| {title} | " + " | ".join(f"{b['metrics']['test'][key]:.4f}" for b in blocks) + " |")
    for title, key in (("测试样本数", "rows"), ("负面样本数", "neg"), ("正面样本数", "pos")):
        values = []
        for b in blocks:
            cm = b["metrics"]["test"]["confusion_matrix"]["matrix"]
            total = sum(map(sum, cm))
            counts = b["metrics"]["test"]["true_label_counts"]
            values.append({"rows": total, "neg": counts["负面"], "pos": counts["正面"]}[key])
        rows.append(f"| {title} | " + " | ".join(str(v) for v in values) + " |")
    return "\n".join(rows)


def confusion_section(block: dict) -> str:
    cm = block["metrics"]["test"]["confusion_matrix"]
    tn, fp = cm["matrix"][0]
    fn, tp = cm["matrix"][1]
    return (
        f"**{MODEL_LABELS[block['name']]}**\n\n"
        f"| 真实 \\ 预测 | 负面 | 正面 |\n| --- | --- | --- |\n"
        f"| 负面 | {tn} (TN) | {fp} (FP) |\n| 正面 | {fn} (FN) | {tp} (TP) |\n\n"
        f"错误 {fp + fn} 条：假阳性 {fp}（负面判成正面），假阴性 {fn}（正面判成负面）。"
    )


def error_section(block: dict, limit: int) -> str:
    errors = [r for r in block["predictions"] if r["true"] != r["pred"]]
    lines = [f"### {MODEL_LABELS[block['name']]}：{len(errors)} 条错误", ""]
    counts = {c: 0 for c in CATEGORIES}
    for rec in errors:
        counts[classify_error(rec)] += 1
    lines.append("错误类型分布：" + "、".join(f"{k} {v}" for k, v in counts.items() if v))
    lines.append("")
    lines.append(
        "> 类型由关键词规则按优先级判定（转折 > 否定 > 褒贬混合 > 反讽 > 过短 > 其他），"
        "一条评论常同时具备多种特征，标签只反映第一个命中的特征。"
    )
    lines.append("")
    for rec in errors[:limit]:
        direction = "负面误判为正面" if rec["true"] == 0 else "正面误判为负面"
        snippet = rec["text"][:70].replace("\n", " ")
        prob = rec.get("prob_positive")
        prob_txt = f"，P(正面)={prob:.3f}" if isinstance(prob, float) else ""
        lines.append(
            f"- **{classify_error(rec)}｜{direction}{prob_txt}**：{snippet}"
            + ("…" if len(rec["text"]) > 70 else "")
        )
    return "\n".join(lines)


def agreement_section(blocks: list[dict]) -> str:
    if len(blocks) < 2:
        return ""
    a = {r["text"]: r for r in blocks[0]["predictions"]}
    b = {r["text"]: r for r in blocks[1]["predictions"]}
    common = set(a) & set(b)
    both_ok = sum(1 for t in common if a[t]["pred"] == a[t]["true"] and b[t]["pred"] == b[t]["true"])
    both_bad = sum(1 for t in common if a[t]["pred"] != a[t]["true"] and b[t]["pred"] != b[t]["true"])
    only_a = sum(1 for t in common if a[t]["pred"] == a[t]["true"] and b[t]["pred"] != b[t]["true"])
    only_b = sum(1 for t in common if a[t]["pred"] != a[t]["true"] and b[t]["pred"] == b[t]["true"])
    return (
        f"## 两个模型在同一测试集上的对照（{len(common)} 条）\n\n"
        f"- 都预测正确：{both_ok}\n- 都预测错误：{both_bad}\n"
        f"- 仅基线正确：{only_a}\n- 仅 BERT 正确：{only_b}\n"
    )


def dataset_section() -> str:
    meta = mt.read_json(PROCESSED_DIR / "meta.json")
    if not meta:
        return "## 数据集概况\n\n未找到 data/processed/meta.json，请先运行 `python -m src.data_loader`。\n"
    s = meta["splits"]
    return (
        "## 数据集概况\n\n"
        f"- 来源：{meta['dataset']}，文件 `{meta['source_file']}`，编码 {meta['encoding']}\n"
        f"- 列名：文本列 `{meta['detected_text_col']}`，标签列 `{meta['detected_label_col']}`；"
        f"标签映射 0=负面、1=正面\n"
        f"- 原始 {meta['clean_stats']['raw_rows']} 行，清洗后 {meta['clean_stats']['clean_rows']} 行"
        f"（空文本/缺失 {meta['clean_stats']['missing_text']}，重复 {meta['clean_stats']['duplicate_text']}）\n"
        f"- 划分（seed={meta['random_seed']}，分层 80/10/10）："
        f"train {s['train']['rows']}、val {s['val']['rows']}、test {s['test']['rows']}\n"
        f"- 测试集类别分布：负面 {s['test']['label_distribution']['0']}、正面 {s['test']['label_distribution']['1']}"
        f"（正面占 {s['test']['positive_ratio']:.1%}）\n"
    )


def build_report(blocks: list[dict], limit: int) -> str:
    parts = [
        "# 模型评估报告",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M}；所有指标均由实际运行结果读取，测试集未参与调参。\n",
        dataset_section(),
    ]
    if not blocks:
        parts.append("尚无可用模型结果，请先运行 `python -m src.train_baseline`。")
        return "\n".join(parts)

    parts.append("## 测试集指标对比\n\n" + comparison_table(blocks) + "\n")
    parts.append("## 混淆矩阵\n\n" + "\n".join(confusion_section(b) + "\n" for b in blocks))
    parts.append(agreement_section(blocks))
    parts.append("## 错误分析\n\n" + "\n".join(error_section(b, limit) + "\n" for b in blocks))
    for b in blocks:
        if b["name"] == "baseline" and b["metrics"].get("top_features"):
            tf = b["metrics"]["top_features"]
            parts.append(
                "## 基线模型可解释性（LR 权重最高的字符 n-gram）\n\n"
                f"- 偏正面：{'、'.join(tf['正面'][:10])}\n- 偏负面：{'、'.join(tf['负面'][:10])}\n"
            )
    return "\n".join(parts)


def main() -> None:
    ensure_stdout_utf8()
    parser = argparse.ArgumentParser(description="生成模型评估报告")
    parser.add_argument("--examples", type=int, default=6, help="每个模型展示的错误样本数")
    args = parser.parse_args()

    blocks = [b for b in (load_block(name) for name in ("baseline", "bert")) if b]
    report = build_report(blocks, args.examples)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "evaluation_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()
