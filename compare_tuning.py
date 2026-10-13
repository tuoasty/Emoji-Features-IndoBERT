"""Frozen vs LoRA vs full fine-tuning: one table over every condition each mode has run.

Same aggregation as slice_report.py: per fold, average the seeds; then mean +/- std
(population std) across the 5 folds. Accuracy is reported on the whole corpus, the
emoji slice and the non-emoji slice; macro-F1 on the whole corpus.

The tuning mode carried into the per-tweet analyses is fixed in advance: the mode with
the highest B0 whole-corpus mean accuracy, unless its lead over LoRA is under one std,
in which case LoRA is kept. This script prints that verdict.

Usage:
    python compare_tuning.py
"""
import collections
import os
import statistics as st

from sklearn.metrics import f1_score

from slice_report import mcnemar_exact_p
from src.models.indobert import TUNING_MODES
from src.results_io import load_runs, report_dir

CONDITION_ORDER = ["b0", "b1", "t1", "f1_esr_ang", "f1_esr_ours",
                   "f1_master_ang", "f1_master_ours", "t1_pos", "t1_neg"]


def fold_stat(runs, condition, metric, only_emoji=None):
    """{fold: seed-averaged metric} for one condition."""
    by_fold = collections.defaultdict(list)
    for (cond, fold, _seed), recs in runs.items():
        if cond != condition:
            continue
        rs = list(recs.values())
        if only_emoji is not None:
            rs = [r for r in rs if r["has_emoji"] == only_emoji]
        if not rs:
            continue
        y, p = [r["true"] for r in rs], [r["pred"] for r in rs]
        if metric == "acc":
            by_fold[fold].append(sum(a == b for a, b in zip(y, p)) / len(y))
        else:
            by_fold[fold].append(f1_score(y, p, average="macro"))
    return {f: st.mean(v) for f, v in by_fold.items()}


def majority_correct(runs, condition):
    """{(fold, val_index): seed-majority correct} over ALL tweets (not only emoji)."""
    votes = collections.defaultdict(list)
    for (cond, fold, _seed), recs in runs.items():
        if cond != condition:
            continue
        for i, r in recs.items():
            votes[(fold, i)].append(r["pred"] == r["true"])
    return {k: sum(v) > len(v) / 2 for k, v in votes.items()}


def tuning_mcnemar(runs, conditions=("b0", "t1")):
    """Per-tweet exact McNemar, LoRA vs each other tuning mode, on all tweets."""
    out = ["", "=== Per-tweet exact McNemar vs LoRA, all tweets (seed-majority, paired on (fold, val_index)) ==="]
    for cond in conditions:
        if not any(k[0] == cond for k in runs.get("lora", {})):
            continue
        ref = majority_correct(runs["lora"], cond)
        for t in TUNING_MODES:
            if t == "lora" or not any(k[0] == cond for k in runs[t]):
                continue
            other = majority_correct(runs[t], cond)
            keys = sorted(set(ref) & set(other))
            lora_only = sum(1 for k in keys if ref[k] and not other[k])
            other_only = sum(1 for k in keys if other[k] and not ref[k])
            p = mcnemar_exact_p(lora_only, other_only)
            out.append(f"  {cond:4s} lora vs {t:6s}: n={len(keys)}  lora-only-right={lora_only}  "
                       f"{t}-only-right={other_only}  exact p={p:.4f}")
    return out


def mean_std(fm):
    vals = list(fm.values())
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0), len(vals)


def main():
    runs = {t: load_runs(t) for t in TUNING_MODES}
    lines = ["=== Frozen vs LoRA vs full fine-tuning (mean +/- std over 5 folds, seeds averaged) ==="]
    cols = [("whole acc", "acc", None), ("emoji acc", "acc", True),
            ("non-emoji acc", "acc", False), ("whole macro-F1", "f1", None)]
    lines.append(f"  {'condition':16s} {'tuning':6s} {'runs':>4s} "
                 + " ".join(f"{name:>16s}" for name, _, _ in cols))
    b0 = {}
    for cond in CONDITION_ORDER:
        for t in TUNING_MODES:
            n_runs = sum(1 for k in runs[t] if k[0] == cond)
            if not n_runs:
                continue
            cells = []
            for _, metric, flag in cols:
                m, s, _ = mean_std(fold_stat(runs[t], cond, metric, flag))
                cells.append(f"{m*100:7.2f} +/- {s*100:4.2f}")
            lines.append(f"  {cond:16s} {t:6s} {n_runs:>4d} " + " ".join(f"{c:>16s}" for c in cells))
            if cond == "b0":
                b0[t] = mean_std(fold_stat(runs[t], "b0", "acc"))
        lines.append("")

    if "lora" in b0 and len(b0) > 1:
        lines.append("  B0 whole acc: " + ", ".join(
            f"{t} {m*100:.2f} +/- {s*100:.2f}" for t, (m, s, _) in b0.items()))
        best = max(b0, key=lambda t: b0[t][0])
        gap = b0[best][0] - b0["lora"][0]
        threshold = max(b0[best][1], b0["lora"][1])
        if best == "lora":
            verdict = "LoRA is best -> keep LoRA"
        elif gap < threshold:
            verdict = (f"{best} leads by {gap*100:.2f}pp, under one std "
                       f"({threshold*100:.2f}pp) -> keep LoRA")
        else:
            verdict = f"{best} leads by {gap*100:.2f}pp (> one std) -> carry forward {best}"
        lines.append(f"  -> {verdict}")
        for t in b0:
            n = sum(1 for k in runs[t] if k[0] == "b0")
            if n < 15:
                lines.append(f"  (provisional: only {n}/15 {t} B0 runs present)")

    lines += tuning_mcnemar(runs)
    text = "\n".join(lines)
    print(text)
    out_dir = os.path.dirname(report_dir("lora"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "compare_tuning.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n-> {os.path.join(out_dir, 'compare_tuning.txt')}")


if __name__ == "__main__":
    main()
