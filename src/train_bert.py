"""对比模型：中文预训练模型（默认 hfl/chinese-roberta-wwm-ext）微调二分类。

本脚本需要 torch + transformers。若依赖缺失、模型下载失败或训练中断，脚本会给出明确
提示并以非零状态退出，不影响 TF-IDF 基线模型与网页界面。

用法（在带 CUDA 的环境）：
    E:/conda_envs/pytorch_CV/python.exe -m src.train_bert
快速试跑（只跑 30 个训练 step 验证通路）：
    python -m src.train_bert --epochs 1 --max-steps 30
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics as mt
from src.config import BERT_DIR, DEFAULT_BERT_MODEL, MODELS_DIR, RANDOM_SEED, ensure_stdout_utf8


def resolve_model_path(name: str) -> str:
    """优先使用已手动下载到 models/pretrained/<名称> 的权重，避免依赖镜像连通性。"""
    candidate = Path(name)
    if candidate.exists():
        return str(candidate)
    local = MODELS_DIR / "pretrained" / name.rstrip("/").split("/")[-1]
    if local.exists() and (local / "config.json").exists():
        print(f"[本地权重] 使用 {local}")
        return str(local)
    return name


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微调中文预训练模型做情感二分类")
    parser.add_argument("--model", default=DEFAULT_BERT_MODEL, help="Hugging Face 模型名或本地目录")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-steps", type=int, default=0, help="累计到该训练步数后停止，用于冒烟测试")
    parser.add_argument("--limit-train", type=int, default=0, help=">0 时只用前 N 条训练样本")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(choice: str, torch_module) -> str:
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境 CUDA 不可用")
        return "cuda"
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def make_collate(tokenizer, max_length: int, torch_module):
    def collate(batch):
        encoded = tokenizer(
            [item[0] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch_module.tensor([item[1] for item in batch], dtype=torch_module.long)
        return encoded

    return collate


def make_loader(pairs, tokenizer, args, torch_module, shuffle: bool):
    generator = None
    if shuffle:
        generator = torch_module.Generator()
        generator.manual_seed(args.seed)
    return torch_module.utils.data.DataLoader(
        pairs,
        batch_size=args.batch_size if shuffle else args.eval_batch_size,
        shuffle=shuffle,
        collate_fn=make_collate(tokenizer, args.max_length, torch_module),
        generator=generator,
    )


def warmup_lambda(total_steps: int, warmup_steps: int):
    def fn(current: int) -> float:
        if current < warmup_steps:
            return current / max(1, warmup_steps)
        return max(0.0, (total_steps - current) / max(1, total_steps - warmup_steps))

    return fn


def evaluate(model, tokenizer, texts, labels, args, device, torch_module):
    """在整个集合上分批推理，返回指标、预测标签和正面类概率。"""
    import numpy as np

    model.eval()
    loader = make_loader(list(zip(texts, labels)), tokenizer, args, torch_module, shuffle=False)
    preds: list[int] = []
    probs: list[float] = []
    with torch_module.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            probabilities = torch_module.softmax(model(**batch).logits, dim=-1)
            preds.extend(probabilities.argmax(dim=-1).tolist())
            probs.extend(probabilities[:, 1].tolist())
    y_pred = np.asarray(preds)
    y_prob = np.asarray(probs)
    return mt.compute_metrics(np.asarray(labels), y_pred), y_pred, y_prob


def train_one_epoch(model, loader, optimizer, scheduler, scaler, args, device, torch_module):
    model.train()
    loss_sum, steps, t0 = 0.0, 0, time.time()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch_module.amp.autocast("cuda", enabled=(device == "cuda")):
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch_module.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        value = float(loss.item())
        loss_sum += value
        steps += 1
        if steps % 50 == 0:
            print(f"  step {steps}/{len(loader)}  loss={value:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
    return loss_sum / max(1, steps), steps, round(time.time() - t0, 1)


def main() -> int:
    ensure_stdout_utf8()
    args = parse_args()

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        print(f"[BERT 跳过] 当前环境缺少依赖：{exc}")
        print("基线模型不受影响；请在安装了 torch 与 transformers 的环境里运行本脚本。")
        return 1

    from src.data_loader import load_splits

    set_seed(args.seed)
    device = resolve_device(args.device, torch)
    if device == "cpu":
        print("[提示] 正在使用 CPU 训练，速度很慢。建议改用 GPU 环境，或加 --limit-train 500 先试跑。")
    else:
        print(f"[设备] {torch.cuda.get_device_name(0)} / torch {torch.__version__}")

    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    if args.limit_train > 0:
        train = train.head(args.limit_train)
    train_texts = train["text"].tolist()
    train_labels = [int(v) for v in train["label"]]

    model_path = resolve_model_path(args.model)
    print(
        f"\n加载预训练模型：{model_path}"
        f"（HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'huggingface 默认')}）"
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    except Exception as exc:
        print(f"[BERT 跳过] 预训练模型下载或加载失败：{type(exc).__name__}: {exc}")
        print("可按 data/README.md 的方式手动把权重放到 models/pretrained/<模型名> 后重试；"
              "项目其余功能不受影响，可继续使用基线模型：python -m src.train_baseline")
        return 1

    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    loader = make_loader(list(zip(train_texts, train_labels)), tokenizer, args, torch, shuffle=True)

    no_decay = ("bias", "LayerNorm.weight")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
    )
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_lambda(total_steps, int(total_steps * args.warmup_ratio))
    )

    print(
        f"训练配置：epochs={args.epochs} batch_size={args.batch_size} max_length={args.max_length} "
        f"lr={args.lr} seed={args.seed} 训练样本={len(train_texts)} 每 epoch {len(loader)} step"
    )

    history: list[dict] = []
    best: dict = {"macro_f1": -1.0, "accuracy": -1.0, "epoch": None, "state": None}
    done_steps = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss, steps, seconds = train_one_epoch(
            model, loader, optimizer, scheduler, scaler, args, device, torch
        )
        done_steps += steps
        val_metrics, _, _ = evaluate(
            model, tokenizer, val["text"].tolist(), [int(v) for v in val["label"]], args, device, torch
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(epoch_loss, 4),
                "seconds": seconds,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        print(
            f"epoch {epoch}：loss={epoch_loss:.4f}  val acc={val_metrics['accuracy']:.4f}"
            f"  macro-F1={val_metrics['macro_f1']:.4f}  用时 {seconds}s"
        )
        if (val_metrics["macro_f1"], val_metrics["accuracy"]) > (best["macro_f1"], best["accuracy"]):
            best = {
                "macro_f1": val_metrics["macro_f1"],
                "accuracy": val_metrics["accuracy"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        if args.max_steps and done_steps >= args.max_steps:
            print(f"[冒烟测试] 已达到 --max-steps={args.max_steps}，停止后续 epoch")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"\n按验证集 macro-F1 选用 epoch {best['epoch']} 的权重")

    val_list = [int(v) for v in val["label"]]
    test_list = [int(v) for v in test["label"]]
    test_metrics, y_pred, y_prob = evaluate(
        model, tokenizer, test["text"].tolist(), test_list, args, device, torch
    )
    val_metrics, _, _ = evaluate(
        model, tokenizer, val["text"].tolist(), val_list, args, device, torch
    )

    print("\n测试集结果（BERT 对比模型）：")
    print(f"  Accuracy  {test_metrics['accuracy']:.4f}")
    print(f"  Precision {test_metrics['precision']:.4f}（正面类）")
    print(f"  Recall    {test_metrics['recall']:.4f}（正面类）")
    print(f"  F1        {test_metrics['f1']:.4f}（正面类）  macro-F1 {test_metrics['macro_f1']:.4f}")
    tn, fp = test_metrics["confusion_matrix"]["matrix"][0]
    fn, tp = test_metrics["confusion_matrix"]["matrix"][1]
    print(f"  混淆矩阵 [[{tn}, {fp}], [{fn}, {tp}]]  行=真实(负面/正面)")
    print(f"  测试集正负样本：{test_metrics['true_label_counts']}")

    BERT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(BERT_DIR)
    tokenizer.save_pretrained(BERT_DIR)
    mt.write_json(
        BERT_DIR / "model_config.json",
        {
            "model": "中文预训练 Transformer 编码器 + 分类头微调",
            "base_model": args.model,
            "resolved_model_path": model_path,
            "num_labels": 2,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "random_seed": args.seed,
            "device": device,
            "trained_samples": len(train_texts),
            "completed_steps": done_steps,
            "best_epoch": best["epoch"],
            "trained_on": "train split only (80%)",
            "selected_on": "val split (10%) by macro-F1",
            "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        },
    )
    mt.write_json(
        BERT_DIR / "metrics.json",
        {
            "model_name": "chinese_roberta_wwm_finetuned",
            "base_model": args.model,
            "history": history,
            "best_epoch": best["epoch"],
            "val": val_metrics,
            "test": test_metrics,
        },
    )
    mt.save_predictions(
        BERT_DIR / "predictions.jsonl",
        test["text"].tolist(),
        test_list,
        y_pred,
        y_prob,
    )
    print(f"\n模型与结果已保存到 {BERT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
