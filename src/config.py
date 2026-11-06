"""项目级配置：路径、随机种子、标签映射与数据列候选名。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "ChnSentiCorp_htl_all.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_DIR = MODELS_DIR / "baseline"
BERT_DIR = MODELS_DIR / "bert"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)

# 对比模型默认底座；可用 --model 覆盖，例如 bert-base-chinese。
DEFAULT_BERT_MODEL = "hfl/chinese-roberta-wwm-ext"
DEFAULT_MAX_LENGTH = 256

TEXT_COL_CANDIDATES = ("review", "text", "sentence", "content")
LABEL_COL_CANDIDATES = ("label", "labels", "sentiment", "target", "class")

NEG_LABEL = 0
POS_LABEL = 1
LABEL_NAMES = {NEG_LABEL: "负面", POS_LABEL: "正面"}

# 原始数据里可能出现的标签写法，统一映射到 0=负面 / 1=正面。
RAW_LABEL_MAP = {
    "0": NEG_LABEL,
    "1": POS_LABEL,
    "0.0": NEG_LABEL,
    "1.0": POS_LABEL,
    "neg": NEG_LABEL,
    "negative": NEG_LABEL,
    "bad": NEG_LABEL,
    "0=负面": NEG_LABEL,
    "负面": NEG_LABEL,
    "差评": NEG_LABEL,
    "pos": POS_LABEL,
    "positive": POS_LABEL,
    "good": POS_LABEL,
    "1=正面": POS_LABEL,
    "正面": POS_LABEL,
    "好评": POS_LABEL,
}

SPLIT_FILES = {
    "train": PROCESSED_DIR / "train.csv",
    "val": PROCESSED_DIR / "val.csv",
    "test": PROCESSED_DIR / "test.csv",
}

# BERT 默认从 hf-mirror 拉取权重；已在环境变量中显式设置时保持用户配置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def ensure_stdout_utf8() -> None:
    """Windows 控制台默认 GBK，打印中文样例会抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
