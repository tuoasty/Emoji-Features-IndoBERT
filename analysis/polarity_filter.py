"""Polarity filter: T1 vs T1-pos / T1-neg against B0 on the emoji slice.

t1_pos keeps only the emoji tags the master lexicon labels positive, t1_neg only the
negative ones. For each of t1 / t1_pos / t1_neg this reports, against B0:
  - emoji-slice accuracy, mean +/- std over folds (same rule as slice_report.py)
  - helped / hurt per tweet (seed-majority correctness, paired on (fold, val_index))
  - exact McNemar p (kept for the record; the paper reports helped / hurt only)

Writes results/report/<tuning>/mcnemar_polarity_filter.txt. Unlike running
slice_report.py with --t1/--f1 overrides, this never touches failure_cases.json.

Usage:
    python analysis/polarity_filter.py --tuning lora
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from slice_report import correctness_by_tweet, fold_means, mcnemar_exact_p  # noqa: E402
from src.models.indobert import TUNING_MODES  # noqa: E402
from src.results_io import load_runs, preds_dir, report_dir  # noqa: E402

CONDITIONS = ("t1", "t1_pos", "t1_neg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", default="lora", choices=TUNING_MODES)
    args = ap.parse_args()

    runs = load_runs(args.tuning)
    if not runs:
        print(f"No prediction files in {preds_dir(args.tuning)}. Run slice_eval.py first.")
        return
    present = [c for c in ("b0",) + CONDITIONS if any(k[0] == c for k in runs)]
    if "b0" not in present:
        print("B0 predictions are required as the reference condition.")
        return

    corr = {c: correctness_by_tweet(runs, c)[0] for c in present}
    lines = [f"=== Polarity filter [{args.tuning}]: vs B0 on the EMOJI slice ===",
             "  acc = mean +/- std over folds (seeds averaged within fold); helped/hurt and",
             "  McNemar = per tweet, seed-majority correctness, paired on (fold, val_index)",
             f"    {'condition':10s} {'emoji acc':>16s} {'n':>4s} {'helped':>7s} {'hurt':>5s} "
             f"{'net':>5s} {'McNemar p':>10s}"]
    for c in present:
        vals = list(fold_means(runs, c, True).values())
        acc = f"{st.mean(vals)*100:6.2f} +/- {st.pstdev(vals)*100:4.2f}"
        if c == "b0":
            lines.append(f"    {c:10s} {acc:>16s} {len(corr[c]):>4d}   (reference)")
            continue
        keys = sorted(set(corr["b0"]) & set(corr[c]))
        helped = sum(1 for k in keys if not corr["b0"][k] and corr[c][k])
        hurt = sum(1 for k in keys if corr["b0"][k] and not corr[c][k])
        p = mcnemar_exact_p(hurt, helped)
        lines.append(f"    {c:10s} {acc:>16s} {len(keys):>4d} {helped:>7d} {hurt:>5d} "
                     f"{helped - hurt:>+5d} {p:>10.4f}")
    lines.append("\n  CAVEAT: t1_pos / t1_neg sort tags by the master lexicon, which agrees closely")
    lines.append("  with gold (circularity). Diagnostic of the mechanism, not a method.")

    text = "\n".join(lines)
    print(text)
    out_dir = report_dir(args.tuning)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mcnemar_polarity_filter.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
