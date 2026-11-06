"""统一预测接口：加载已训练模型，对单条中文评论输出情感标签与置信度。

Streamlit 界面和其它脚本都通过这里调用模型，BERT 不可用时只暴露基线模型。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

from src.config import BASELINE_DIR, BERT_DIR, LABEL_NAMES


@dataclass
class Prediction:
    model: str
    label: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "label": self.label,
            "label_name": self.label_name,
            "confidence": round(self.confidence, 4),
            "probabilities": self.probabilities,
            "meta": self.meta,
        }


class BasePredictor:
    name = ""
    display_name = ""

    def predict(self, text: str) -> Prediction:  # pragma: no cover - 接口定义
        raise NotImplementedError

    @staticmethod
    def _to_prediction(model_name: str, probs: list[float]) -> Prediction:
        label = int(max(range(len(probs)), key=lambda i: probs[i]))
        return Prediction(
            model=model_name,
            label=label,
            label_name=LABEL_NAMES[label],
            confidence=float(probs[label]),
            probabilities={LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
        )


class BaselinePredictor(BasePredictor):
    name = "baseline"
    display_name = "TF-IDF + Logistic Regression（基线）"

    def __init__(self) -> None:
        import joblib

        self.vectorizer = joblib.load(BASELINE_DIR / "vectorizer.joblib")
        self.model = joblib.load(BASELINE_DIR / "model.joblib")
        self.config = json.loads((BASELINE_DIR / "model_config.json").read_text("utf-8"))

    def predict(self, text: str) -> Prediction:
        probs = self.model.predict_proba(self.vectorizer.transform([text]))[0]
        return self._to_prediction(self.name, [float(p) for p in probs])


class BertPredictor(BasePredictor):
    name = "bert"
    display_name = "chinese-roberta-wwm-ext 微调（对比模型）"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.config = json.loads((BERT_DIR / "model_config.json").read_text("utf-8"))
        self.max_length = int(self.config.get("max_length", 256))
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_DIR.as_posix())
        self.model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR.as_posix())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def predict(self, text: str) -> Prediction:
        batch = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**batch).logits, dim=-1)[0]
        return self._to_prediction(self.name, [float(p) for p in probs.tolist()])


PREDICTORS = {"baseline": BaselinePredictor, "bert": BertPredictor}


def is_trained(name: str) -> bool:
    if name == "baseline":
        return all(
            (BASELINE_DIR / f).exists()
            for f in ("vectorizer.joblib", "model.joblib", "model_config.json")
        )
    if name == "bert":
        return (BERT_DIR / "model_config.json").exists() and (BERT_DIR / "config.json").exists()
    return False


def unavailable_reason(name: str) -> str:
    """返回模型当前不可用的原因，可用时返回空字符串。"""
    if not is_trained(name):
        return f"模型未训练（缺少 models/{name} 产物），请先运行对应训练脚本"
    if name == "bert":
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer  # noqa: F401
        except ImportError:
            return "当前 Python 环境没有 torch/transformers，请在 CUDA 环境（如 E:/conda_envs/pytorch_CV）中运行界面"
    return ""


def available_models() -> list[str]:
    return [name for name in PREDICTORS if not unavailable_reason(name)]


@lru_cache(maxsize=4)
def get_predictor(name: str) -> BasePredictor:
    if name not in PREDICTORS:
        raise ValueError(f"未知模型：{name}")
    if not is_trained(name):
        raise FileNotFoundError(
            f"{name} 模型不可用：{unavailable_reason(name) or '未训练'}"
        )
    return PREDICTORS[name]()


def predict(text: str, model: str = "baseline") -> Prediction:
    if not text or not text.strip():
        raise ValueError("请输入中文评论文本")
    return get_predictor(model).predict(text.strip())
