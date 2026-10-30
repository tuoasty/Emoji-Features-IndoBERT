"""Train every condition x fold x seed and dump per-tweet test-fold predictions.

Checkpoint selection uses an inner validation (dev) split, never the test fold: each
training fold gives up DEV_FRAC of its tweets (src/dataset.py) as a dev set. After every
epoch both dev and test are evaluated; the epoch with the best dev macro-F1 (ties: lower
dev loss, then the earlier epoch) is the run's checkpoint, and its test-fold predictions
are the ones reported. The last epoch's test predictions are kept too (pred_last), as a
no-selection robustness check. No checkpoints are saved (save_strategy="no"); the test
logits of every epoch are captured in memory and the selected one is written out.

Each run writes results/preds/<tag>/<run>.json with one record per test-fold tweet:
    {val_index, has_emoji, true, pred, pred_last, text}
where val_index is the row index into the original dataframe, so the SAME tweet can be
joined across conditions on (fold, seed, val_index). The fold split and the dev split
are identical across all conditions and backbones (StratifiedKFold random_state=0,
dev split DEV_SEED; preprocessing never reorders rows), which is what makes the
cross-condition join valid. <tag> is the tuning mode for IndoBERT (lora, full, frozen)
and <backbone>_<tuning> otherwise (indobertweet_lora).

Tuning modes:
    frozen — encoder frozen, only the classification head trains, lr 1e-3
    lora   — frozen encoder + rank-8 LoRA adapters on query/value, lr 2e-4 (default)
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

    # IndoBERTweet backbone
    python slice_eval.py --backbone indobertweet --conditions b0 t1 f1_master_ours
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

from src.dataset import get_fold, _load_raw, N_SPLITS, DEV_FRAC, DEV_SEED
from src.models.indobert import (BACKBONES, build_model, get_tokenizer, tokenize_dataset,
                                 TUNING_MODES)
from src.results_io import RESULTS_DIR, preds_dir, run_tag

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
    """Recover, for each test-fold tweet (in test_ds order), its original df index,
    raw-text emoji flag, and label. Mirrors get_fold's outer split exactly."""
    df, _ = _load_raw(dataset_name)              # df["text"] is RAW (pre-preprocess)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    _, val_idx = list(skf.split(df, df["label"]))[fold]
    val_df = df.iloc[val_idx].reset_index(drop=True)
    raw_text = val_df["text"].astype(str).tolist()
    has_emoji = [bool(TAG_RE.search(t)) for t in raw_text]
    labels = val_df["label"].astype(int).tolist()
    return list(val_idx), raw_text, has_emoji, labels


def make_capture(n_test: int):
    """compute_metrics that also stashes the test-fold logits of every epoch.
    Dev and test differ in size, which is how a call is told apart."""
    store = {"test_logits": [], "test_labels": None}

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        if len(labels) == n_test:
            store["test_logits"].append(np.asarray(logits))
            store["test_labels"] = np.asarray(labels)
        return {"accuracy": accuracy_score(labels, preds),
                "macro_f1": f1_score(labels, preds, average="macro")}

    return compute_metrics, store


def dev_curve(log_history):
    """Per-epoch dev macro-F1 and loss, in epoch order, from the Trainer's log."""
    rows = [h for h in log_history if "eval_dev_macro_f1" in h]
    rows.sort(key=lambda h: h["epoch"])
    return [h["eval_dev_macro_f1"] for h in rows], [h["eval_dev_loss"] for h in rows]


def select_epoch(dev_f1, dev_loss):
    """Best dev macro-F1; ties go to the lower dev loss, then the earlier epoch."""
    return min(range(len(dev_f1)), key=lambda i: (-dev_f1[i], dev_loss[i], i))


def run_one(condition: str, fold: int, seed: int, dataset: str, tuning: str,
            backbone: str, epochs: int, batch_size: int, lr: float):
    run_name = f"{condition}_{dataset}_fold{fold}_seed{seed}"
    tag = run_tag(tuning, backbone)
    out_dir = preds_dir(tag)
    out_path = os.path.join(out_dir, f"{run_name}.json")
    if os.path.exists(out_path):
        print(f"  skip (exists): {tag}/{run_name}")
        return

    print(f"\n=== Slice eval [{tag}, lr={lr}]: {run_name} ===")
    set_seed(seed)

    train_ds, dev_ds, test_ds, label2id = get_fold(dataset, condition, fold, backbone=backbone)
    num_labels = len(label2id)
    assert len(dev_ds) != len(test_ds), "dev and test must differ in size (capture keys on it)"

    val_index, raw_text, has_emoji, meta_labels = val_meta(dataset, fold)
    assert len(meta_labels) == len(test_ds), "val_meta length mismatch with test_ds"

    tokenizer = get_tokenizer(backbone)
    train_ds = tokenize_dataset(train_ds, tokenizer)
    dev_ds = tokenize_dataset(dev_ds, tokenizer)
    test_ds = tokenize_dataset(test_ds, tokenizer)

    if condition.startswith("f1_"):
        from src.models.f1_fusion import F1FusionModel
        from src.preprocess import FEATURE_DIM
        cols = ["input_ids", "attention_mask", "label", "emoji_feature"]
        model = F1FusionModel(num_labels, feature_dim=FEATURE_DIM[condition], tuning=tuning,
                              backbone=backbone)
    else:
        cols = ["input_ids", "attention_mask", "label"]
        model = build_model(num_labels, tuning=tuning, backbone=backbone)
    for ds in (train_ds, dev_ds, test_ds):
        ds.set_format("torch", columns=cols)

    compute_metrics, store = make_capture(len(test_ds))
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
        eval_dataset={"dev": dev_ds, "test": test_ds},
        compute_metrics=compute_metrics,
    )
    train_out = trainer.train()

    dev_f1, dev_loss = dev_curve(trainer.state.log_history)
    test_logits = store["test_logits"]
    assert len(dev_f1) == len(test_logits) == epochs, \
        f"expected {epochs} dev/test evals, got {len(dev_f1)}/{len(test_logits)}"
    cap_labels = store["test_labels"].astype(int).tolist()
    # The captured eval order equals test_ds order equals val_meta order.
    assert cap_labels == meta_labels, "captured label order != val_meta order"

    best = select_epoch(dev_f1, dev_loss)
    preds = np.argmax(test_logits[best], axis=-1).astype(int).tolist()
    preds_last = np.argmax(test_logits[-1], axis=-1).astype(int).tolist()
    test_f1_by_epoch = [float(f1_score(meta_labels, np.argmax(l, axis=-1), average="macro"))
                        for l in test_logits]
    acc = accuracy_score(meta_labels, preds)
    mf1 = f1_score(meta_labels, preds, average="macro")

    records = []
    for i in range(len(preds)):
        records.append({
            "val_index": int(val_index[i]),
            "has_emoji": bool(has_emoji[i]),
            "true": int(meta_labels[i]),
            "pred": int(preds[i]),
            "pred_last": int(preds_last[i]),
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
        "backbone": backbone,
        "model": BACKBONES[backbone],
        "learning_rate": lr,
        "train_runtime_s": train_out.metrics.get("train_runtime"),
        "label2id": label2id,
        "selection": f"best dev macro-F1 (dev = {DEV_FRAC:.0%} of train fold, seed {DEV_SEED})",
        "n_train": len(train_ds),
        "n_dev": len(dev_ds),
        "best_epoch": best + 1,
        "dev_macro_f1_by_epoch": [float(x) for x in dev_f1],
        "dev_loss_by_epoch": [float(x) for x in dev_loss],
        # diagnostic only (never used for selection): how the old test-fold pick compares
        "test_macro_f1_by_epoch": test_f1_by_epoch,
        "eval_accuracy": acc,
        "eval_macro_f1": mf1,
        "n_val": len(preds),
        "n_emoji": int(sum(has_emoji)),
        "records": records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  saved {out_path}  epoch={best + 1}  acc={acc:.4f} f1={mf1:.4f}  "
          f"(n_train={len(train_ds)}, n_dev={len(dev_ds)}, n_test={len(preds)}, "
          f"n_emoji={sum(has_emoji)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=PAPER_CONDITIONS, choices=ALL_CONDITIONS)
    p.add_argument("--tuning", default="lora", choices=TUNING_MODES)
    p.add_argument("--backbone", default="indobert", choices=sorted(BACKBONES))
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1])
    p.add_argument("--dataset", default="pilkada")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-3 for frozen, 2e-4 for lora, 2e-5 for full")
    args = p.parse_args()
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.tuning]
    tag = run_tag(args.tuning, args.backbone)

    total = len(args.conditions) * len(args.folds) * len(args.seeds)
    print(f"Planned runs: {total}  [{tag}, lr={lr}]  -> {preds_dir(tag)}")
    for cond in args.conditions:
        for fold in args.folds:
            for seed in args.seeds:
                run_one(cond, fold, seed, args.dataset, args.tuning, args.backbone,
                        args.epochs, args.batch_size, lr)
    print(f"\nDone. Aggregate with: python slice_report.py --tuning {tag}")


if __name__ == "__main__":
    main()
