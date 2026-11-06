"""基线模型：字符级 n-gram TF-IDF + Logistic Regression。

用法：
    python -m src.train_baseline
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics as mt
from src.config import BASELINE_DIR, LABEL_NAMES, RANDOM_SEED, ensure_stdout_utf8
from src.data_loader import load_splits

# 只在验证集上比较，测试集不参与调参。
GRID = {
    "ngram_range": [(1, 2), (1, 3)],
    "C": [1.0, 5.0],
    "class_weight": [None, "balanced"],
}
MAX_FEATURES = 100_000


def make_vectorizer(ngram_range: tuple[int, int]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,
        max_features=MAX_FEATURES,
    )


def make_classifier(C: float, class_weight: str | None) -> LogisticRegression:
    return LogisticRegression(
        C=C, class_weight=class_weight, max_iter=2000, random_state=RANDOM_SEED
    )


def top_features(vectorizer: TfidfVectorizer, model: LogisticRegression, k: int = 15) -> dict:
    names = np.asarray(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    order = np.argsort(coef)
    return {
        LABEL_NAMES[0]: [f"{names[i]}({coef[i]:+.2f})" for i in order[:k][::-1]],
        LABEL_NAMES[1]: [f"{names[i]}({coef[i]:+.2f})" for i in order[-k:][::-1]],
    }


def main() -> None:
    ensure_stdout_utf8()
    parser = argparse.ArgumentParser(description="训练 TF-IDF + Logistic Regression 基线")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    t0 = time.time()

    print(
        f"样本量：train {len(train)} / val {len(val)} / test {len(test)}"
        f"（test 中 负面 {int((test['label'] == 0).sum())}，正面 {int((test['label'] == 1).sum())}）"
    )

    print("\n在验证集上搜索超参数（测试集不参与调参）：")
    search: list[dict] = []
    val_cache: dict[tuple, np.ndarray] = {}
    for ngram in GRID["ngram_range"]:
        vectorizer = make_vectorizer(ngram)
        xtr = vectorizer.fit_transform(train["text"].tolist())
        xval = vectorizer.transform(val["text"].tolist())
        val_cache[ngram] = (vectorizer, xval)
        for C in GRID["C"]:
            for cw in GRID["class_weight"]:
                model = make_classifier(C, cw).fit(xtr, train["label"])
                m = mt.compute_metrics(
                    val["label"].to_numpy(), model.predict(xval)
                )
                search.append(
                    {
                        "ngram_range": list(ngram),
                        "C": C,
                        "class_weight": cw,
                        "val_accuracy": m["accuracy"],
                        "val_macro_f1": m["macro_f1"],
                    }
                )
                print(
                    f"  ngram{ngram} C={C} class_weight={cw}: "
                    f"val acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}"
                )

    best = max(search, key=lambda r: (r["val_macro_f1"], r["val_accuracy"]))
    best_ngram = tuple(best["ngram_range"])
    print(
        f"\n最优配置：ngram{best_ngram} C={best['C']} class_weight={best['class_weight']}"
        f"（val macro-F1={best['val_macro_f1']}）"
    )

    vectorizer = make_vectorizer(best_ngram)
    xtr = vectorizer.fit_transform(train["text"].tolist())
    xtest = vectorizer.transform(test["text"].tolist())
    model = make_classifier(best["C"], best["class_weight"]).fit(xtr, train["label"])

    y_true = test["label"].to_numpy()
    y_pred = model.predict(xtest)
    y_prob = model.predict_proba(xtest)
    test_metrics = mt.compute_metrics(y_true, y_pred)
    _, xval = val_cache[best_ngram]
    val_metrics = mt.compute_metrics(val["label"].to_numpy(), model.predict(xval))

    print("\n测试集结果：")
    print(f"  Accuracy  {test_metrics['accuracy']:.4f}")
    print(f"  Precision {test_metrics['precision']:.4f}（正面类）")
    print(f"  Recall    {test_metrics['recall']:.4f}（正面类）")
    print(f"  F1        {test_metrics['f1']:.4f}（正面类）  macro-F1 {test_metrics['macro_f1']:.4f}")
    tn, fp = test_metrics["confusion_matrix"]["matrix"][0]
    fn, tp = test_metrics["confusion_matrix"]["matrix"][1]
    print(f"  混淆矩阵 [[{tn}, {fp}], [{fn}, {tp}]]  行=真实(负面/正面)，列=预测(负面/正面)")
    print(f"  测试集正负样本：{test_metrics['true_label_counts']}")
    print(f"  特征数：{xtest.shape[1]}，训练耗时：{time.time() - t0:.1f}s")

    features = top_features(vectorizer, model)
    print(f"  正面权重最高字符 n-gram：{'、'.join(features[LABEL_NAMES[1]][:10])}")
    print(f"  负面权重最高字符 n-gram：{'、'.join(features[LABEL_NAMES[0]][:10])}")

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, BASELINE_DIR / "vectorizer.joblib")
    joblib.dump(model, BASELINE_DIR / "model.joblib")
    mt.write_json(
        BASELINE_DIR / "model_config.json",
        {
            "model": "TF-IDF(char_wb) + Logistic Regression",
            "vectorizer": {
                "analyzer": "char_wb",
                "ngram_range": list(best_ngram),
                "min_df": 2,
                "sublinear_tf": True,
                "max_features": MAX_FEATURES,
                "n_features": int(xtest.shape[1]),
            },
            "classifier": {"C": best["C"], "class_weight": best["class_weight"], "max_iter": 2000},
            "random_seed": args.seed,
            "trained_on": "train split only (80%)",
            "tuned_on": "val split (10%)",
        },
    )
    mt.write_json(
        BASELINE_DIR / "metrics.json",
        {
            "model_name": "baseline_tfidf_logreg",
            "train_seconds": round(time.time() - t0, 2),
            "val": val_metrics,
            "test": test_metrics,
            "hyperparameter_search": search,
            "best_config": best,
            "top_features": features,
        },
    )
    mt.save_predictions(
        BASELINE_DIR / "predictions.jsonl",
        test["text"].tolist(),
        y_true,
        y_pred,
        y_prob[:, 1],
    )
    print(f"\n模型与结果已保存到 {BASELINE_DIR}")


if __name__ == "__main__":
    main()
