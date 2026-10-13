"""Train every condition x fold x seed and dump per-tweet validation predictions.

No checkpoints are saved (save_strategy="no"). Instead the validation logits of the
BEST-macro-F1 epoch are captured during training, which gives both the run's score and
its per-tweet predictions (needed for the emoji-slice metrics and the per-tweet analyses).

Each run writes results/preds/<tuning>/<run>.json with one record per validation tweet:
    {val_index, has_emoji, true, pred, text}
where val_index is the row index into the original dataframe, so the SAME tweet can be
joined across conditions on (fold, seed, val_index). The fold split is identical across
all conditions (StratifiedKFold random_state=0; preprocessing never reorders rows),
which is what makes the cross-condition join valid.

Tuning modes:
    frozen — IndoBERT frozen, only the classification head trains, lr 1e-3
    lora   — frozen IndoBERT + rank-8 LoRA adapters on query/value, lr 2e-4 (default)
    full   — every weight trains, lr 2e-5

Usage:
    # one smoke run (verify before a sweep)
    python slice_eval.py --conditions b0 --folds 0 --seeds 42

    # the seven paper conditions, 5 folds x 3 seeds = 105 runs per tuning mode
    # (skips runs whose preds JSON already exists)
    python slice_eval.py --tuning frozen
    python slice_eval.py --tuning lora
    python slice_eval.py --tuning full

    # polarity-filtered tag-text (keep only positive / only negative emoji tags)
    python slice_eval.py --tuning lora --conditions t1_pos t1_neg
"""
import argparse
import json
import os
import random
import re

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import Trainer, TrainingArguments

from src.dataset import get_fold, _load_raw, N_SPLITS
from src.models.indobert import build_model, get_tokenizer, tokenize_dataset, TUNING_MODES
from src.results_io import RESULTS_DIR, preds_dir

PAPER_CONDITIONS = [
    "b0", "t1", "b1",
    "f1_master_ours", "f1_master_ang", "f1_esr_ours", "f1_esr_ang",
]
ALL_CONDITIONS = PAPER_CONDITIONS + ["t1_pos", "t1_neg"]
DEFAULT_LR = {"frozen": 1e-3, "lora": 2e-4, "full": 2e-5}
TAG_RE = re.compile(r"<[^>]+>")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def val_meta(dataset_name: str, fold: int):
    """Recover, for each validation tweet (in val_ds order), its original df index,
    raw-text emoji flag, and label. Mirrors get_fold's split exactly."""
    df, _ = _load_raw(dataset_name)              # df["text"] is RAW (pre-preprocess)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    _, val_idx = list(skf.split(df, df["label"]))[fold]
    val_df = df.iloc[val_idx].reset_index(drop=True)
    raw_text = val_df["text"].astype(str).tolist()
    has_emoji = [bool(TAG_RE.search(t)) for t in raw_text]
    labels = val_df["label"].astype(int).tolist()
    return list(val_idx), raw_text, has_emoji, labels


def make_capture():
    """compute_metrics that also stashes the logits/labels of the best-macro-F1 epoch."""
    store = {"best_f1": -1.0, "logits": None, "labels": None}

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        mf1 = f1_score(labels, preds, average="macro")
        if mf1 > store["best_f1"]:
            store["best_f1"] = mf1
            store["logits"] = np.asarray(logits)
            store["labels"] = np.asarray(labels)
        return {"accuracy": acc, "macro_f1": mf1}

    return compute_metrics, store


def run_one(condition: str, fold: int, seed: int, dataset: str, tuning: str,
            epochs: int, batch_size: int, lr: float):
    run_name = f"{condition}_{dataset}_fold{fold}_seed{seed}"
    out_dir = preds_dir(tuning)
    out_path = os.path.join(out_dir, f"{run_name}.json")
    if os.path.exists(out_path):
        print(f"  skip (exists): {tuning}/{run_name}")
        return

    print(f"\n=== Slice eval [{tuning}, lr={lr}]: {run_name} ===")
    set_seed(seed)

    train_ds, val_ds, label2id = get_fold(dataset, condition, fold)
    num_labels = len(label2id)

    val_index, raw_text, has_emoji, meta_labels = val_meta(dataset, fold)
    assert len(meta_labels) == len(val_ds), "val_meta length mismatch with val_ds"

    tokenizer = get_tokenizer()
    train_ds = tokenize_dataset(train_ds, tokenizer)
    val_ds = tokenize_dataset(val_ds, tokenizer)

    if condition.startswith("f1_"):
        from src.models.f1_fusion import F1FusionModel
        from src.preprocess import FEATURE_DIM
        cols = ["input_ids", "attention_mask", "label", "emoji_feature"]
        train_ds.set_format("torch", columns=cols)
        val_ds.set_format("torch", columns=cols)
        model = F1FusionModel(num_labels, feature_dim=FEATURE_DIM[condition], tuning=tuning)
    else:
        cols = ["input_ids", "attention_mask", "label"]
        train_ds.set_format("torch", columns=cols)
        val_ds.set_format("torch", columns=cols)
        model = build_model(num_labels, tuning=tuning)

    compute_metrics, store = make_capture()
    training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "_trainer_tmp"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        seed=seed,
        report_to="none",
        logging_steps=20,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    train_out = trainer.train()

    logits = store["logits"]
    cap_labels = store["labels"].astype(int).tolist()
    # The captured eval order equals val_ds order equals val_meta order.
    assert cap_labels == meta_labels, "captured label order != val_meta order"

    preds = np.argmax(logits, axis=-1).astype(int).tolist()
    acc = accuracy_score(meta_labels, preds)
    mf1 = f1_score(meta_labels, preds, average="macro")

    records = []
    for i in range(len(preds)):
        records.append({
            "val_index": int(val_index[i]),
            "has_emoji": bool(has_emoji[i]),
            "true": int(meta_labels[i]),
            "pred": int(preds[i]),
            "text": raw_text[i],
        })

    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "run": run_name,
        "condition": condition,
        "fold": fold,
        "seed": seed,
        "dataset": dataset,
        "tuning": tuning,
        "learning_rate": lr,
        "train_runtime_s": train_out.metrics.get("train_runtime"),
        "label2id": label2id,
        "eval_accuracy": acc,
        "eval_macro_f1": mf1,
        "best_epoch_macro_f1": store["best_f1"],
        "n_val": len(preds),
        "n_emoji": int(sum(has_emoji)),
        "records": records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  saved {out_path}  acc={acc:.4f} f1={mf1:.4f}  "
          f"(n_val={len(preds)}, n_emoji={sum(has_emoji)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=PAPER_CONDITIONS, choices=ALL_CONDITIONS)
    p.add_argument("--tuning", default="lora", choices=TUNING_MODES)
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1])
    p.add_argument("--dataset", default="pilkada")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-3 for frozen, 2e-4 for lora, 2e-5 for full")
    args = p.parse_args()
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.tuning]

    total = len(args.conditions) * len(args.folds) * len(args.seeds)
    print(f"Planned runs: {total}  [{args.tuning}, lr={lr}]  -> {preds_dir(args.tuning)}")
    for cond in args.conditions:
        for fold in args.folds:
            for seed in args.seeds:
                run_one(cond, fold, seed, args.dataset, args.tuning,
                        args.epochs, args.batch_size, lr)
    print(f"\nDone. Aggregate with: python slice_report.py --tuning {args.tuning}")


if __name__ == "__main__":
    main()
