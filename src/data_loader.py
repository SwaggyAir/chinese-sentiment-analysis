"""数据读取、检查、清洗与分层划分。

用法：
    python -m src.data_loader            # 生成 data/processed/{train,val,test}.csv 并打印概况
    python -m src.data_loader --inspect  # 只打印概况，不写文件
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):  # 支持 python src/data_loader.py 直接运行
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    LABEL_COL_CANDIDATES,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_CSV_PATH,
    RAW_LABEL_MAP,
    SPLIT_FILES,
    SPLIT_RATIOS,
    TEXT_COL_CANDIDATES,
    ensure_stdout_utf8,
)

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "big5", "latin-1")


def read_csv_any_encoding(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise RuntimeError(f"无法解析 {path}：{last_error}")


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """列名可能随整理版本变化，优先按候选名匹配，再按内容特征推断。"""
    lower = {c: str(c).strip().lower() for c in df.columns}
    text_col = next(
        (c for name in TEXT_COL_CANDIDATES for c, low in lower.items() if low == name), None
    )
    if text_col is None:
        text_col = next(
            (c for c in df.columns if df[c].dtype == object and df[c].astype(str).str.len().mean() > 10),
            None,
        )
    label_col = next(
        (c for name in LABEL_COL_CANDIDATES for c, low in lower.items() if low == name), None
    )
    if label_col is None:
        candidates = [
            c
            for c in df.columns
            if c != text_col and df[c].nunique(dropna=True) == 2
        ]
        label_col = candidates[0] if candidates else None
    if text_col is None or label_col is None:
        raise ValueError(f"未能识别文本列/标签列，现有列：{list(df.columns)}")
    return text_col, label_col


def normalize_label(value: object) -> int | None:
    key = str(value).strip().lower()
    if key in RAW_LABEL_MAP:
        return RAW_LABEL_MAP[key]
    try:
        number = float(key)
    except ValueError:
        return None
    return int(number) if number in (0.0, 1.0) else None


def load_raw(path: Path = RAW_CSV_PATH) -> tuple[pd.DataFrame, str, str, str]:
    """返回原始 DataFrame 以及编码、文本列名、标签列名。"""
    if not path.exists():
        raise FileNotFoundError(
            f"未找到数据文件 {path}，请先按 data/README.md 下载 ChnSentiCorp_htl_all.csv。"
        )
    df, encoding = read_csv_any_encoding(path)
    text_col, label_col = detect_columns(df)
    return df, encoding, text_col, label_col


def clean(df: pd.DataFrame, text_col: str, label_col: str) -> tuple[pd.DataFrame, dict]:
    """删除空文本与无法识别的标签，去除重复文本（保留首次出现）。"""
    stats: dict[str, int] = {"raw_rows": int(len(df))}
    work = df.copy()
    work["text"] = work[text_col].astype("string").str.strip()
    work["label"] = work[label_col].map(normalize_label)

    stats["missing_text"] = int(work["text"].isna().sum() + (work["text"] == "").sum())
    stats["unrecognized_label"] = int(work["label"].isna().sum())
    work = work.dropna(subset=["text", "label"])
    work = work[work["text"] != ""]

    before = len(work)
    work = work.drop_duplicates(subset=["text"], keep="first")
    stats["duplicate_text"] = int(before - len(work))
    work["label"] = work["label"].astype(int)
    stats["clean_rows"] = int(len(work))
    return work[["text", "label"]].reset_index(drop=True), stats


def stratified_split(df: pd.DataFrame, seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """分层随机划分 80/10/10；测试集只用于最终评估。"""
    _, val_ratio, test_ratio = SPLIT_RATIOS
    train, rest = train_test_split(
        df, test_size=val_ratio + test_ratio, stratify=df["label"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest["label"], random_state=seed
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def summarize(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().sort_index()
    return {
        "rows": int(len(df)),
        "label_distribution": {str(int(k)): int(v) for k, v in counts.items()},
        "positive_ratio": round(float(df["label"].mean()), 4),
        "text_len_mean": round(float(df["text"].str.len().mean()), 1),
        "text_len_median": float(df["text"].str.len().median()),
        "text_len_max": int(df["text"].str.len().max()),
    }


def prepare(path: Path = RAW_CSV_PATH, seed: int = RANDOM_SEED, write: bool = True) -> dict:
    raw, encoding, text_col, label_col = load_raw(path)
    cleaned, stats = clean(raw, text_col, label_col)
    splits = stratified_split(cleaned, seed)

    meta = {
        "source_file": str(path.name),
        "dataset": "ChnSentiCorp_htl_all（SophonPlus/ChineseNlpCorpus 整理版）",
        "encoding": encoding,
        "detected_text_col": text_col,
        "detected_label_col": label_col,
        "raw_label_values": sorted(str(v) for v in raw[label_col].dropna().unique())[:10],
        "label_mapping": {"0": "0 = 负面", "1": "1 = 正面"},
        "clean_rules": [
            "去除文本首尾空白",
            "删除空文本与缺失标签行",
            "删除无法识别的标签值",
            "完全相同的文本只保留第一次出现",
        ],
        "clean_stats": stats,
        "random_seed": seed,
        "split_sizes": {name: int(len(df)) for name, df in splits.items()},
        "splits": {name: summarize(df) for name, df in splits.items()},
        "clean_summary": summarize(cleaned),
    }

    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in splits.items():
            df.to_csv(SPLIT_FILES[name], index=False, encoding="utf-8-sig")
        (PROCESSED_DIR / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {"meta": meta, "raw": raw, "cleaned": cleaned, "splits": splits}


def load_splits() -> dict[str, pd.DataFrame]:
    missing = [str(p) for p in SPLIT_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少划分文件 {missing}，请先运行：python -m src.data_loader"
        )
    return {
        name: pd.read_csv(path, encoding="utf-8-sig")[["text", "label"]]
        for name, path in SPLIT_FILES.items()
    }


def print_report(result: dict) -> None:
    raw, cleaned, meta = result["raw"], result["cleaned"], result["meta"]
    print("=" * 64)
    print(f"数据文件：{meta['source_file']}  编码：{meta['encoding']}")
    print(f"原始行数：{meta['clean_stats']['raw_rows']}  列名：{list(raw.columns)}")
    print(f"识别到文本列：{meta['detected_text_col']}  标签列：{meta['detected_label_col']}")
    raw_counts = raw[meta["detected_label_col"]].value_counts(dropna=False)
    print(f"原始标签分布：{dict(raw_counts)}")
    print(f"缺失值数量：{dict(raw.isna().sum())}")
    print(
        "清洗：空文本/缺失 {} 条，标签无法识别 {} 条，重复文本 {} 条".format(
            meta["clean_stats"]["missing_text"],
            meta["clean_stats"]["unrecognized_label"],
            meta["clean_stats"]["duplicate_text"],
        )
    )
    print(f"清洗后样本：{meta['clean_stats']['clean_rows']} 条")
    print(f"清洗后标签分布（0=负面, 1=正面）：{meta['clean_summary']['label_distribution']}")
    print(
        f"文本长度：均值 {meta['clean_summary']['text_len_mean']}，"
        f"中位数 {meta['clean_summary']['text_len_median']}，最长 {meta['clean_summary']['text_len_max']}"
    )
    print(f"随机种子：{meta['random_seed']}  划分比例：80/10/10（分层）")
    for name in ("train", "val", "test"):
        s = meta["splits"][name]
        print(f"  {name:5s} {s['rows']:5d} 条  标签分布 {s['label_distribution']}")
    print("=" * 64)


def main() -> None:
    ensure_stdout_utf8()
    parser = argparse.ArgumentParser(description="准备 ChnSentiCorp 数据")
    parser.add_argument("--inspect", action="store_true", help="只打印概况，不写出划分文件")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    result = prepare(seed=args.seed, write=not args.inspect)
    print_report(result)
    if not args.inspect:
        print(f"已写出：{PROCESSED_DIR / 'train.csv'}、val.csv、test.csv、meta.json")


if __name__ == "__main__":
    main()
