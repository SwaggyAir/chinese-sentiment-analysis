"""统一的二分类指标计算与保存，供基线和 BERT 训练脚本复用。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import LABEL_NAMES, NEG_LABEL, POS_LABEL


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = [NEG_LABEL, POS_LABEL]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, labels=labels, pos_label=POS_LABEL)), 4),
        "recall": round(float(recall_score(y_true, y_pred, labels=labels, pos_label=POS_LABEL)), 4),
        "f1": round(float(f1_score(y_true, y_pred, labels=labels, pos_label=POS_LABEL)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro")), 4),
        "per_class": {
            LABEL_NAMES[label]: {
                "precision": round(
                    float(precision_score(y_true, y_pred, labels=labels, pos_label=label, zero_division=0)), 4
                ),
                "recall": round(
                    float(recall_score(y_true, y_pred, labels=labels, pos_label=label, zero_division=0)), 4
                ),
                "f1": round(
                    float(f1_score(y_true, y_pred, labels=labels, pos_label=label, zero_division=0)), 4
                ),
                "support": int((y_true == label).sum()),
            }
            for label in labels
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "note": "行=真实标签(0 负面, 1 正面)，列=预测标签(0 负面, 1 正面)",
        },
        "true_label_counts": {LABEL_NAMES[label]: int((y_true == label).sum()) for label in labels},
        "pred_label_counts": {LABEL_NAMES[label]: int((y_pred == label).sum()) for label in labels},
    }


def save_predictions(
    path: Path,
    texts: list[str],
    y_true,
    y_pred,
    prob_positive=None,
) -> None:
    """prob_positive 为每条样本的正面类概率（一维）。"""
    records = []
    for i, text in enumerate(texts):
        record = {"text": text, "true": int(y_true[i]), "pred": int(y_pred[i])}
        if prob_positive is not None:
            record["prob_positive"] = round(float(prob_positive[i]), 6)
        records.append(record)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
