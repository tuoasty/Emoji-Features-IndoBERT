"""Polarity mix: is the harm to negative-gold tweets caused by POSITIVE emoji?

Each emoji-bearing tweet is placed in a cell by
    (which master-lexicon polarities its emoji tags carry) x (gold label)
with mix in {negative only, positive only, both}. For every condition present among
b0 / t1 / t1_pos / t1_neg, per-tweet correctness is the majority vote across seeds
(same rule as slice_report.py), and each condition is compared with B0 per cell:
helped = B0 wrong -> X right, hurt = B0 right -> X wrong.

If emoji simply add positivity, the B0 -> T1 harm on negative-gold tweets sits in the
cells where a positive emoji is present, t1_pos reproduces it, and t1_neg does not.

Usage:
    python analysis/polarity_mix.py --tuning lora
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data.reverse_map import load_tag_to_sentiment  # noqa: E402
from src.models.indobert import TUNING_MODES  # noqa: E402
from src.preprocess import _normalize_tag_key  # noqa: E402
from src.results_io import load_runs, preds_dir, report_dir  # noqa: E402

CONDITIONS = ("b0", "t1", "t1_pos", "t1_neg")
MIXES = ("negative only", "positive only", "both")
GOLD_NAME = {0: "negative", 1: "positive"}
TAG_RE = re.compile(r"<[^>]+>")


def emoji_mix(text: str, tag_to_sentiment: dict) -> str:
    pols = {tag_to_sentiment.get(_normalize_tag_key(t)) for t in TAG_RE.findall(text)}
    has_pos, has_neg = "positive" in pols, "negative" in pols
    if has_pos and has_neg:
        return "both"
    if has_pos:
        return "positive only"
    if has_neg:
        return "negative only"
    return "unmapped"


def per_tweet(runs, condition):
    """(fold, val_index) -> (majority correct, gold, text), emoji tweets only."""
    votes = collections.defaultdict(list)
    meta = {}
    for (cond, fold, _seed), recs in runs.items():
        if cond != condition:
            continue
        for i, r in recs.items():
            if r["has_emoji"]:
                votes[(fold, i)].append(r["pred"] == r["true"])
                meta[(fold, i)] = (r["true"], r["text"])
    return {k: (sum(v) > len(v) / 2, *meta[k]) for k, v in votes.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", default="lora", choices=TUNING_MODES)
    args = ap.parse_args()

    runs = load_runs(args.tuning)
    if not runs:
        print(f"No prediction files in {preds_dir(args.tuning)}. Run slice_eval.py first.")
        return
    present = [c for c in CONDITIONS if any(k[0] == c for k in runs)]
    if "b0" not in present:
        print("B0 predictions are required as the reference condition.")
        return

    tag_to_sentiment = load_tag_to_sentiment()
    pt = {c: per_tweet(runs, c) for c in present}
    cell = {k: (emoji_mix(text, tag_to_sentiment), gold)
            for k, (_, gold, text) in pt["b0"].items()}

    lines = [f"=== Polarity mix [{args.tuning}]: harm by emoji polarity mix x gold "
             f"(emoji slice, seed-majority) ===",
             f"conditions present: {present}"]
    unmapped = sum(1 for m, _ in cell.values() if m == "unmapped")
    if unmapped:
        lines.append(f"  WARNING: {unmapped} emoji tweets have no lexicon-mapped tag")

    out = {}
    header = f"    {'emoji mix':14s} {'gold':9s} {'n':>4s} " + " ".join(f"{c:>8s}" for c in present)
    lines.append("\n  Accuracy (%) per cell:")
    lines.append(header)
    for mix in MIXES:
        for g in (0, 1):
            keys = [k for k, v in cell.items() if v == (mix, g)]
            if not keys:
                continue
            accs = {c: sum(pt[c][k][0] for k in keys if k in pt[c]) / len(keys) for c in present}
            lines.append(f"    {mix:14s} {GOLD_NAME[g]:9s} {len(keys):>4d} "
                         + " ".join(f"{accs[c]*100:>8.2f}" for c in present))
            out[f"{mix}|{GOLD_NAME[g]}"] = {"n": len(keys), "acc": accs}

    for c in present:
        if c == "b0":
            continue
        lines.append(f"\n  B0 -> {c}: helped / hurt / net per cell")
        lines.append(f"    {'emoji mix':14s} {'gold':9s} {'n':>4s} {'helped':>7s} {'hurt':>6s} {'net':>5s}")
        for mix in MIXES:
            for g in (0, 1):
                keys = [k for k, v in cell.items() if v == (mix, g) and k in pt[c]]
                if not keys:
                    continue
                helped = sum(1 for k in keys if not pt["b0"][k][0] and pt[c][k][0])
                hurt = sum(1 for k in keys if pt["b0"][k][0] and not pt[c][k][0])
                lines.append(f"    {mix:14s} {GOLD_NAME[g]:9s} {len(keys):>4d} "
                             f"{helped:>7d} {hurt:>6d} {helped - hurt:>+5d}")
                out[f"{mix}|{GOLD_NAME[g]}"].setdefault("vs_b0", {})[c] = {
                    "helped": helped, "hurt": hurt, "net": helped - hurt}
    lines.append("\n  CAVEAT: cells are small (some n < 20). Descriptive only.")

    text = "\n".join(lines)
    print(text)
    out_dir = report_dir(args.tuning)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "polarity_mix.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(os.path.join(out_dir, "polarity_mix.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n-> {os.path.join(out_dir, 'polarity_mix.txt')}")


if __name__ == "__main__":
    main()
