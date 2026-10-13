"""Lexicon feature vs tag-text: per-tweet McNemar tests among T1 and the four fusion
conditions on the emoji slice, plus how far the two presence features differ.

Paper names: T1+L = the f1_* conditions; "continuous" = *_ours, "presence" = *_ang.
Reports, on the emoji slice:
  - T1 against each fusion condition (which tweets each gets right that the other
    gets wrong; seed-majority correctness, paired on (fold, val_index))
  - presence against continuous encoding, within each lexicon
  - exact McNemar p for every contrast (same test as slice_report.py)
and, over all 900 tweets, on how many tweets the master and ESR presence features
differ (the reason both presence conditions are kept).

Writes results/report/<tuning>/mcnemar_fusion.txt.

Usage:
    python analysis/fusion_mcnemar.py --tuning lora
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from slice_report import correctness_by_tweet, mcnemar_exact_p  # noqa: E402
from src.dataset import F1_CONDITIONS, _load_raw  # noqa: E402
from src.models.indobert import TUNING_MODES  # noqa: E402
from src.preprocess import extract_emoji_feature  # noqa: E402
from src.results_io import load_runs, preds_dir, report_dir  # noqa: E402

PAPER_NAME = {
    "t1": "T1",
    "f1_master_ours": "T1+L (master, cont.)",
    "f1_master_ang": "T1+L (master, pres.)",
    "f1_esr_ours": "T1+L (ESR, cont.)",
    "f1_esr_ang": "T1+L (ESR, pres.)",
}

# (a, b): "a only" counts tweets a gets right and b gets wrong.
CONTRASTS = [("t1", c) for c in F1_CONDITIONS] + [
    ("f1_master_ang", "f1_master_ours"),
    ("f1_esr_ang", "f1_esr_ours"),
]


def presence_feature_diff():
    """Tweets (of all 900) on which the master and ESR presence features differ."""
    from data.esr_map import load_emoji_to_esr
    from data.reverse_map import load_tag_to_emoji, load_tag_to_sentiment
    tag_to_emoji, tag_to_sentiment = load_tag_to_emoji(), load_tag_to_sentiment()
    emoji_to_esr = load_emoji_to_esr()
    df, _ = _load_raw("pilkada")
    texts = df["text"].astype(str)
    master = [extract_emoji_feature(t, "f1_master_ang", tag_to_sentiment=tag_to_sentiment)
              for t in texts]
    esr = [extract_emoji_feature(t, "f1_esr_ang", tag_to_emoji=tag_to_emoji,
                                 emoji_to_esr=emoji_to_esr) for t in texts]
    return sum(1 for m, e in zip(master, esr) if m != e), len(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", default="lora", choices=TUNING_MODES)
    args = ap.parse_args()

    runs = load_runs(args.tuning)
    if not runs:
        print(f"No prediction files in {preds_dir(args.tuning)}. Run slice_eval.py first.")
        return
    needed = {c for pair in CONTRASTS for c in pair}
    missing = sorted(c for c in needed if not any(k[0] == c for k in runs))
    if missing:
        print(f"Missing predictions for: {', '.join(missing)}")
        return

    corr = {c: correctness_by_tweet(runs, c)[0] for c in needed}
    lines = [f"=== Lexicon feature vs tag-text [{args.tuning}]: per-tweet McNemar, EMOJI slice ===",
             "  seed-majority correctness, paired on (fold, val_index); exact McNemar p",
             "  'A only' = tweets A gets right and B gets wrong",
             f"    {'A':22s} {'B':22s} {'n':>4s} {'A only':>7s} {'B only':>7s} {'McNemar p':>10s}"]
    for a, b in CONTRASTS:
        keys = sorted(set(corr[a]) & set(corr[b]))
        a_only = sum(1 for k in keys if corr[a][k] and not corr[b][k])
        b_only = sum(1 for k in keys if corr[b][k] and not corr[a][k])
        p = mcnemar_exact_p(a_only, b_only)
        lines.append(f"    {PAPER_NAME[a]:22s} {PAPER_NAME[b]:22s} {len(keys):>4d} "
                     f"{a_only:>7d} {b_only:>7d} {p:>10.4f}")

    n_diff, n_all = presence_feature_diff()
    lines.append(f"\n  Presence features, master vs ESR: differ on {n_diff} of {n_all} tweets")
    lines.append("  (all tweets; the feature does not depend on training).")

    text = "\n".join(lines)
    print(text)
    out_dir = report_dir(args.tuning)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mcnemar_fusion.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
