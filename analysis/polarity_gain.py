"""Polarity gain: model gain by GOLD polarity on the emoji slice, and direction of change.

Splits the 165 emoji-bearing tweets by gold label and computes the B0 -> T1 accuracy
gain separately for positive-gold and negative-gold tweets: do emoji help both
polarities, or help one and hurt the other?

Mirrors slice_report.py exactly:
    - reads results/preds/<tuning>/*.json
    - per-tweet correctness AND per-tweet predicted label = majority vote across seeds,
      paired on (fold, val_index), emoji tweets only (has_emoji == True)
    - gold: negative=0, positive=1

Reports (descriptive + one exact-McNemar p per subgroup; subgroups are small):
    1. Per-condition slice accuracy split by gold polarity.
    2. B0 -> T1 gain by polarity (headline) + B0->B1, B0->F1, T1->F1 for context.
    3. Direction-of-change: b0_pred -> t1_pred flips, by gold polarity and flip direction.
    4. Helped / hurt correctness transitions (B0->T1) split by gold polarity.

Usage:
    python analysis/polarity_gain.py --tuning lora
    python analysis/polarity_gain.py --tuning full --f1 f1_esr_ours
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.indobert import TUNING_MODES  # noqa: E402
from src.results_io import load_runs, preds_dir, report_dir  # noqa: E402

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

GOLD_NAME = {0: "negative", 1: "positive"}


def per_tweet(runs, condition):
    """(fold, val_index) -> dict(gold, pred, correct, text) via seed-majority vote.
    Emoji tweets only. pred = majority predicted label; correct = majority correctness."""
    pred_votes = collections.defaultdict(list)
    corr_votes = collections.defaultdict(list)
    gold, text = {}, {}
    for (cond, fold, seed), recs in runs.items():
        if cond != condition:
            continue
        for i, r in recs.items():
            if not r["has_emoji"]:
                continue
            k = (fold, i)
            pred_votes[k].append(r["pred"])
            corr_votes[k].append(r["pred"] == r["true"])
            gold[k] = r["true"]
            text[k] = r["text"]
    out = {}
    for k in pred_votes:
        n = len(pred_votes[k])
        out[k] = {
            "gold": gold[k],
            "pred": 1 if sum(pred_votes[k]) > n / 2 else 0,   # binary majority label
            "correct": sum(corr_votes[k]) > n / 2,
            "text": text[k],
        }
    return out


def mcnemar_exact_p(n01, n10):
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    if scipy_stats is not None:
        return float(scipy_stats.binomtest(k, n, 0.5).pvalue)
    from math import comb
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def subgroup_acc(pt, gold_val):
    keys = [k for k, v in pt.items() if v["gold"] == gold_val]
    if not keys:
        return None
    n_corr = sum(pt[k]["correct"] for k in keys)
    return n_corr, len(keys), n_corr / len(keys)


def gain_table(lines, pt_a, name_a, pt_b, name_b):
    """B->A? No: gain = acc(B) - acc(A) for A->B contrast, split by gold, with McNemar."""
    keys = sorted(set(pt_a) & set(pt_b))
    lines.append(f"\n  {name_a} -> {name_b}  gain (acc {name_b} - acc {name_a}), by gold polarity:")
    lines.append(f"    {'gold':10s} {'n':>4s} {name_a+' acc':>12s} {name_b+' acc':>12s} "
                 f"{'gain(pp)':>9s} {'disc(a/b)':>10s} {'McNemar p':>10s}")
    result = {}
    for g in (0, 1):
        gk = [k for k in keys if pt_a[k]["gold"] == g]
        if not gk:
            continue
        ca = sum(pt_a[k]["correct"] for k in gk)
        cb = sum(pt_b[k]["correct"] for k in gk)
        acc_a, acc_b = ca / len(gk), cb / len(gk)
        # discordant on correctness: a-only-right / b-only-right
        n_a = sum(1 for k in gk if pt_a[k]["correct"] and not pt_b[k]["correct"])
        n_b = sum(1 for k in gk if not pt_a[k]["correct"] and pt_b[k]["correct"])
        p = mcnemar_exact_p(n_a, n_b)
        lines.append(f"    {GOLD_NAME[g]:10s} {len(gk):>4d} {acc_a*100:>11.2f} {acc_b*100:>11.2f} "
                     f"{(acc_b-acc_a)*100:>+9.2f} {f'{n_a}/{n_b}':>10s} {p:>10.4f}")
        result[GOLD_NAME[g]] = {"n": len(gk), "acc_a": acc_a, "acc_b": acc_b,
                                "gain_pp": (acc_b - acc_a) * 100,
                                "disc_a": n_a, "disc_b": n_b, "mcnemar_p": p}
    # whole slice row
    ca = sum(pt_a[k]["correct"] for k in keys)
    cb = sum(pt_b[k]["correct"] for k in keys)
    n_a = sum(1 for k in keys if pt_a[k]["correct"] and not pt_b[k]["correct"])
    n_b = sum(1 for k in keys if not pt_a[k]["correct"] and pt_b[k]["correct"])
    lines.append(f"    {'ALL':10s} {len(keys):>4d} {ca/len(keys)*100:>11.2f} {cb/len(keys)*100:>11.2f} "
                 f"{(cb-ca)/len(keys)*100:>+9.2f} {f'{n_a}/{n_b}':>10s} {mcnemar_exact_p(n_a,n_b):>10.4f}")
    result["ALL"] = {"n": len(keys), "gain_pp": (cb - ca) / len(keys) * 100,
                     "disc_a": n_a, "disc_b": n_b, "mcnemar_p": mcnemar_exact_p(n_a, n_b)}
    return result


def direction_of_change(lines, pt_b0, pt_t1):
    """b0_pred -> t1_pred label flips on the slice, by gold polarity and direction."""
    keys = sorted(set(pt_b0) & set(pt_t1))
    lines.append("\n  DIRECTION-OF-CHANGE: b0_pred -> t1_pred label flips (adding emoji tags):")
    lines.append(f"    {'gold':10s} {'n':>4s} {'no-flip':>8s} {'neg->pos':>9s} {'pos->neg':>9s}")
    dc = {}
    for g in (0, 1):
        gk = [k for k in keys if pt_b0[k]["gold"] == g]
        noflip = neg2pos = pos2neg = 0
        for k in gk:
            a, b = pt_b0[k]["pred"], pt_t1[k]["pred"]
            if a == b:
                noflip += 1
            elif a == 0 and b == 1:
                neg2pos += 1
            else:
                pos2neg += 1
        lines.append(f"    {GOLD_NAME[g]:10s} {len(gk):>4d} {noflip:>8d} {neg2pos:>9d} {pos2neg:>9d}")
        dc[GOLD_NAME[g]] = {"n": len(gk), "no_flip": noflip, "neg->pos": neg2pos, "pos->neg": pos2neg}
    lines.append("    (a flip TOWARD the gold sign = correction; adding emoji nudges the label)")
    return dc


def helped_hurt_by_polarity(lines, pt_b0, pt_t1):
    keys = sorted(set(pt_b0) & set(pt_t1))
    lines.append("\n  HELPED / HURT correctness transitions B0 -> T1, by gold polarity:")
    lines.append(f"    {'gold':10s} {'n':>4s} {'helped':>7s} {'hurt':>6s} {'net':>5s}")
    hh = {}
    for g in (0, 1):
        gk = [k for k in keys if pt_b0[k]["gold"] == g]
        helped = sum(1 for k in gk if not pt_b0[k]["correct"] and pt_t1[k]["correct"])
        hurt = sum(1 for k in gk if pt_b0[k]["correct"] and not pt_t1[k]["correct"])
        lines.append(f"    {GOLD_NAME[g]:10s} {len(gk):>4d} {helped:>7d} {hurt:>6d} {helped-hurt:>+5d}")
        hh[GOLD_NAME[g]] = {"n": len(gk), "helped": helped, "hurt": hurt, "net": helped - hurt}
    return hh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", default="lora", choices=TUNING_MODES)
    ap.add_argument("--b0", default="b0")
    ap.add_argument("--b1", default="b1")
    ap.add_argument("--t1", default="t1")
    ap.add_argument("--f1", default="f1_master_ours")
    args = ap.parse_args()

    runs = load_runs(args.tuning)
    if not runs:
        print(f"No prediction files in {preds_dir(args.tuning)}. Run slice_eval.py first.")
        return

    pt = {name: per_tweet(runs, cond)
          for name, cond in [("B0", args.b0), ("B1", args.b1),
                             ("T1", args.t1), ("F1", args.f1)]}

    lines = []
    lines.append(f"=== Polarity gain [{args.tuning}]: gain by GOLD polarity + direction of change (emoji slice) ===")
    lines.append(f"F1 headline variant = {args.f1}")

    # 1. per-condition slice accuracy by polarity
    lines.append("\n  Per-condition emoji-slice accuracy, split by gold polarity "
                 "(per-tweet majority vote):")
    lines.append(f"    {'cond':6s} {'neg acc (n)':>18s} {'pos acc (n)':>18s} {'all acc (n)':>18s}")
    for name in ("B0", "B1", "T1", "F1"):
        p = pt[name]
        neg = subgroup_acc(p, 0)
        pos = subgroup_acc(p, 1)
        allc = sum(v["correct"] for v in p.values())
        alln = len(p)
        def f(x):
            return f"{x[2]*100:6.2f} ({x[0]}/{x[1]})" if x else "--"
        lines.append(f"    {name:6s} {f(neg):>18s} {f(pos):>18s} "
                     f"{f((allc, alln, allc/alln)):>18s}")

    # 2. gain tables (headline = B0 -> T1)
    lines.append("\n  --- GAIN BY POLARITY ---")
    res_b0_t1 = gain_table(lines, pt["B0"], "B0", pt["T1"], "T1")   # headline
    gain_table(lines, pt["B0"], "B0", pt["B1"], "B1")
    gain_table(lines, pt["B0"], "B0", pt["F1"], "F1")
    gain_table(lines, pt["T1"], "T1", pt["F1"], "F1")

    # 3. direction of change
    dc = direction_of_change(lines, pt["B0"], pt["T1"])

    # 4. helped/hurt by polarity
    hh = helped_hurt_by_polarity(lines, pt["B0"], pt["T1"])

    lines.append("\n  CAVEAT: subgroup n ~75/90; cells are small. Read as descriptive + effect size.")

    text = "\n".join(lines)
    print(text)
    out_dir = report_dir(args.tuning)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "polarity_gain.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(os.path.join(out_dir, "polarity_gain.json"), "w", encoding="utf-8") as fh:
        json.dump({"f1_variant": args.f1, "b0_t1_gain": res_b0_t1,
                   "direction_of_change": dc, "helped_hurt": hh}, fh,
                  ensure_ascii=False, indent=2)
    print(f"\n-> {os.path.join(out_dir, 'polarity_gain.txt')}")
    print(f"-> {os.path.join(out_dir, 'polarity_gain.json')}")


if __name__ == "__main__":
    main()
