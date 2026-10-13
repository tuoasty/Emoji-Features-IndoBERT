"""Aggregate slice_eval.py outputs: emoji-slice metrics, significance, failure cases.

Reads results/preds/<tuning>/*.json (written by slice_eval.py) and produces:
  1. Per-condition accuracy on the emoji slice vs non-emoji slice vs whole corpus
     (seed-averaged within fold, then mean +/- std across folds).
  2. A sanity check that non-emoji predictions drift across conditions no more than
     retraining noise (conditions differ only on emoji tweets; if not, preprocessing leaks).
  3. Significance on the EMOJI slice, all pairwise contrasts over B0/B1/T1/F1:
       - PRIMARY: per-tweet exact McNemar (binomial) on majority-vote correctness,
         paired on (fold, val_index).
       - SECONDARY: paired t-test + Cohen's d across the 5 fold-means (n=5,
         underpowered; kept as the effect-size view).
  4. Failure-case tracking on emoji tweets, joined across conditions on (fold, val_index):
        - emoji HELPED:  B0 wrong -> T1/F1 right
        - emoji HURT:    B0 right -> T1/F1 wrong
        - persistent:    B0 wrong AND F1 wrong
     Correctness is a majority vote across available seeds. Tweet texts are dumped to
     results/report/<tuning>/failure_cases.json for the qualitative error taxonomy.

Usage:
    python slice_report.py --tuning lora
    python slice_report.py --tuning full --f1 f1_master_ang   # different headline F1 variant
"""
import argparse
import collections
import json
import os
import statistics as st

from src.models.indobert import TUNING_MODES
from src.results_io import load_runs, preds_dir, report_dir

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


def _acc(records, only_emoji=None):
    rs = records
    if only_emoji is True:
        rs = [r for r in records if r["has_emoji"]]
    elif only_emoji is False:
        rs = [r for r in records if not r["has_emoji"]]
    if not rs:
        return None
    return sum(r["pred"] == r["true"] for r in rs) / len(rs)


def fold_means(runs, condition, slice_flag):
    """Per-fold accuracy on a slice, averaging seeds within each fold."""
    by_fold = collections.defaultdict(list)
    for (cond, fold, seed), recs in runs.items():
        if cond != condition:
            continue
        a = _acc(list(recs.values()), only_emoji=slice_flag)
        if a is not None:
            by_fold[fold].append(a)
    return {f: st.mean(v) for f, v in sorted(by_fold.items())}


def summarize(runs, conditions):
    print("\n=== ACCURACY by slice (mean +/- std over folds, seeds averaged within fold) ===")
    print(f"{'condition':16s} {'whole':>16s} {'emoji-slice':>16s} {'non-emoji':>16s}")
    table = {}
    for c in conditions:
        row = {}
        for name, flag in [("whole", None), ("emoji", True), ("nonemoji", False)]:
            fm = fold_means(runs, c, flag)
            if fm:
                vals = list(fm.values())
                row[name] = (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0, fm)
        table[c] = row
        def fmt(k):
            if k not in row:
                return f"{'--':>16s}"
            m, s, _ = row[k]
            return f"{m*100:7.2f} +/- {s*100:4.2f}"
        print(f"{c:16s} {fmt('whole'):>16s} {fmt('emoji'):>16s} {fmt('nonemoji'):>16s}")
    return table


def _nonemoji_flip_rate(runs, key_a, key_b):
    """Fraction of shared non-emoji val tweets whose pred differs between two runs."""
    a, b = runs[key_a], runs[key_b]
    idx = [i for i in a if not a[i]["has_emoji"] and i in b]
    if not idx:
        return None
    diff = sum(1 for i in idx if a[i]["pred"] != b[i]["pred"])
    return diff, len(idx)


def _pooled_rate(pairs):
    d = sum(x[0] for x in pairs if x)
    n = sum(x[1] for x in pairs if x)
    return (d, n, d / n) if n else (0, 0, 0.0)


def sanity_nonemoji_drift(runs, conditions):
    """Non-emoji predictions are NOT expected to be bit-identical across conditions:
    each condition is retrained independently (save_strategy="no"), so the learned
    weights — and hence even the non-emoji predictions — drift with training
    stochasticity. The correct null is therefore the *within-condition, between-seed*
    flip rate. The slice story holds iff the cross-condition non-emoji drift is no
    larger than that retraining-noise baseline. (Pre-2026-07 the buggy b1 was
    bit-identical to b0 and served as a 0-flip identity proof of the join machinery;
    with b1 fixed that control is gone — rerun a duplicate b0 under a different
    run name if the identity proof is ever needed again.)
    """
    print("\n=== SANITY: non-emoji drift vs retraining-noise baseline ===")
    seeds = sorted({s for (_, _, s) in runs})
    folds = sorted({f for (_, f, _) in runs})
    import itertools

    # Baseline: within each condition, flip rate between distinct seeds, same fold.
    print("  within-condition between-seed flip (pure retrain noise, the null):")
    baselines = {}
    for c in conditions:
        pairs = [_nonemoji_flip_rate(runs, (c, f, s1), (c, f, s2))
                 for f in folds for s1, s2 in itertools.combinations(seeds, 2)
                 if (c, f, s1) in runs and (c, f, s2) in runs]
        d, n, r = _pooled_rate(pairs)
        baselines[c] = r
        if n:
            print(f"    {c:16s}: {d:4d}/{n:4d} = {r*100:5.1f}%")
    # Null = worst-case retraining noise, with a 25% tolerance band: the baselines
    # are themselves noisy point estimates, and a genuine preprocessing leak would
    # force non-emoji drift far above the seed-noise floor (tens of %, not ~1pp over).
    null_rate = max(baselines.values()) if baselines else 1.0
    threshold = null_rate * 1.25

    # Cross-condition, same seed/fold, against the first condition (b0).
    base = conditions[0]
    print(f"  cross-condition same-seed flip vs {base} "
          f"(null={null_rate*100:.1f}%, threshold={threshold*100:.1f}%):")
    ok = True
    for c in conditions[1:]:
        pairs = [_nonemoji_flip_rate(runs, (base, f, s), (c, f, s))
                 for f in folds for s in seeds
                 if (base, f, s) in runs and (c, f, s) in runs]
        d, n, r = _pooled_rate(pairs)
        if not n:
            continue
        verdict = "OK (within retrain noise)" if r <= threshold else "DRIFT > threshold"
        if r > threshold:
            ok = False
        print(f"    {base} vs {c:16s}: {d:4d}/{n:4d} = {r*100:5.1f}%   [{verdict}]")
    if ok:
        print(f"  OK: cross-condition non-emoji drift is within retraining noise; "
              f"the emoji effect is confined to the slice, as claimed.")
    else:
        print("  WARNING: drift exceeds threshold - investigate a preprocessing leak.")


def paired_stats(a_fm, b_fm):
    """Paired test across shared folds. Returns (mean_diff, t, p, cohen_d, n)."""
    folds = sorted(set(a_fm) & set(b_fm))
    diffs = [b_fm[f] - a_fm[f] for f in folds]
    if len(diffs) < 2:
        return None
    md = st.mean(diffs)
    sd = st.pstdev(diffs) if len(diffs) > 1 else 0.0
    sd_sample = st.stdev(diffs)
    cohen_d = md / sd_sample if sd_sample else float("inf")
    if scipy_stats is not None:
        t, p = scipy_stats.ttest_rel([b_fm[f] for f in folds], [a_fm[f] for f in folds])
    else:
        se = sd_sample / (len(diffs) ** 0.5)
        t = md / se if se else float("inf")
        p = float("nan")
    return md, float(t), float(p), cohen_d, len(diffs)


def _pairs(named_conditions):
    import itertools
    return list(itertools.combinations(named_conditions, 2))


def significance(runs, named_conditions):
    print("\n=== SECONDARY: paired t-test on EMOJI-slice fold-means (n=5, underpowered) ===")
    if scipy_stats is None:
        print("  (scipy not found -- p-values omitted, t-stat is manual)")
    for (an, a), (bn, b) in _pairs(named_conditions):
        a_fm = fold_means(runs, a, True)
        b_fm = fold_means(runs, b, True)
        res = paired_stats(a_fm, b_fm)
        if res is None:
            print(f"  {an} vs {bn}: insufficient folds")
            continue
        md, t, p, d, n = res
        sig = "SIG" if (p == p and p < 0.05) else ("ns" if p == p else "?")
        print(f"  {an} vs {bn}: mean diff = {md*100:+.2f}pp  t={t:+.2f}  "
              f"p={p:.4f}  cohen_d={d:+.2f}  (n={n} folds)  [{sig}]")


def mcnemar_exact_p(n01, n10):
    """Two-sided exact McNemar p-value: binomial test of min(n01,n10) out of the
    n01+n10 discordant pairs against p=0.5. Exact form chosen because discordant
    counts here are small (~10-20); the chi-square approximation would be invalid."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    if scipy_stats is not None:
        return float(scipy_stats.binomtest(k, n, 0.5).pvalue)
    from math import comb
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar_significance(runs, named_conditions):
    """PRIMARY test (§III.F): per-tweet exact McNemar on the emoji slice.

    Correctness per tweet = majority vote across seeds (same rule as failure_cases),
    paired across conditions on (fold, val_index).
    """
    print("\n=== PRIMARY: exact McNemar per tweet, EMOJI slice (majority vote over seeds) ===")
    corr = {}
    for name, cond in named_conditions:
        c, _ = correctness_by_tweet(runs, cond)
        if c:
            corr[name] = c
    for (an, _), (bn, _) in _pairs(named_conditions):
        if an not in corr or bn not in corr:
            print(f"  {an} vs {bn}: missing predictions, skipped")
            continue
        a, b = corr[an], corr[bn]
        keys = sorted(set(a) & set(b))
        n01 = sum(1 for k in keys if a[k] and not b[k])   # A right, B wrong
        n10 = sum(1 for k in keys if not a[k] and b[k])   # A wrong, B right
        p = mcnemar_exact_p(n01, n10)
        sig = "SIG" if p < 0.05 else "ns"
        print(f"  {an} vs {bn}: n={len(keys)}  discordant={n01 + n10} "
              f"({an}-only-right={n01}, {bn}-only-right={n10})  "
              f"net {bn}-{an}={n10 - n01:+d} tweets  exact p={p:.4f}  [{sig}]")


def correctness_by_tweet(runs, condition):
    """Majority-vote correctness per (fold, val_index) across seeds, emoji tweets only."""
    votes = collections.defaultdict(list)  # (fold, val_index) -> [correct bools]
    text = {}
    for (cond, fold, seed), recs in runs.items():
        if cond != condition:
            continue
        for i, r in recs.items():
            if not r["has_emoji"]:
                continue
            votes[(fold, i)].append(r["pred"] == r["true"])
            text[(fold, i)] = r["text"]
    out = {}
    for k, v in votes.items():
        out[k] = sum(v) > len(v) / 2  # majority correct
    return out, text


def failure_cases(runs, b0, t1, f1, out_dir, out_name="failure_cases.json"):
    print("\n=== FAILURE / IMPACT CASES on emoji tweets (majority vote across seeds) ===")
    cb0, txt = correctness_by_tweet(runs, b0)
    ct1, t2 = correctness_by_tweet(runs, t1)
    cf1, t3 = correctness_by_tweet(runs, f1)
    txt.update(t2); txt.update(t3)
    keys = sorted(set(cb0) & set(ct1) & set(cf1))

    cats = {
        "helped_by_t1": [],   # B0 wrong -> T1 right
        "helped_by_f1": [],   # B0 wrong -> F1 right
        "hurt_by_t1":   [],   # B0 right -> T1 wrong
        "hurt_by_f1":   [],   # B0 right -> F1 wrong
        "persistent":   [],   # B0 wrong AND F1 wrong
        "f1_over_t1":   [],   # T1 wrong -> F1 right (lexicon adds over tag text)
        "f1_under_t1":  [],   # T1 right -> F1 wrong (lexicon breaks what tag text got)
    }
    for k in keys:
        b, t, f = cb0[k], ct1[k], cf1[k]
        rec = {"fold": k[0], "val_index": k[1], "text": txt[k],
               "b0": b, "t1": t, "f1": f}
        if not b and t: cats["helped_by_t1"].append(rec)
        if not b and f: cats["helped_by_f1"].append(rec)
        if b and not t: cats["hurt_by_t1"].append(rec)
        if b and not f: cats["hurt_by_f1"].append(rec)
        if not b and not f: cats["persistent"].append(rec)
        if not t and f: cats["f1_over_t1"].append(rec)
        if t and not f: cats["f1_under_t1"].append(rec)

    n_emoji = len(keys)
    print(f"  emoji tweets compared: {n_emoji}  (F1 headline = {f1})")
    for name, items in cats.items():
        print(f"  {name:14s}: {len(items):3d}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"f1_variant": f1, "n_emoji_tweets": n_emoji, "categories": cats},
                  fh, ensure_ascii=False, indent=2)
    rel = os.path.relpath(out_path, os.path.dirname(os.path.abspath(__file__)))
    print(f"  full case lists + tweet texts -> {rel.replace(os.sep, '/')}")
    print("    (inspect 'hurt_by_f1' and 'persistent' for the failure taxonomy:")
    print("     sarcasm / emoji-contradicts-text / mixed-polarity / uncovered-emoji)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tuning", default="lora", choices=TUNING_MODES)
    p.add_argument("--b0", default="b0")
    p.add_argument("--b1", default="b1")
    p.add_argument("--t1", default="t1")
    p.add_argument("--f1", default="f1_master_ours", help="headline F1 variant")
    args = p.parse_args()

    runs = load_runs(args.tuning)
    if not runs:
        print(f"No prediction files in {preds_dir(args.tuning)}. Run slice_eval.py first.")
        return
    conds_present = sorted({c for (c, _, _) in runs})
    print(f"[{args.tuning}] loaded predictions for conditions: {conds_present}")

    # A slot overridden with a non-default condition is labeled with that condition,
    # so e.g. --t1 t1_pos prints "T1=t1_pos" instead of passing as plain T1.
    defaults = {"B0": "b0", "B1": "b1", "T1": "t1", "F1": "f1_master_ours"}
    named = [(slot if defaults.get(slot, cond) == cond else f"{slot}={cond}", cond)
             for slot, cond in [("B0", args.b0), ("B1", args.b1), ("T1", args.t1), ("F1", args.f1)]]
    summarize(runs, conds_present)
    sanity_nonemoji_drift(runs, [c for _, c in named if c in conds_present])
    mcnemar_significance(runs, named)
    significance(runs, named)
    # failure_cases.json feeds the error taxonomy (b0 / t1 / f1_master_ours). Only the
    # default B0/T1 pairing writes case lists; another --f1 variant gets its own file.
    if (args.b0, args.t1) == ("b0", "t1"):
        name = "failure_cases.json" if args.f1 == "f1_master_ours" else f"failure_cases_{args.f1}.json"
        failure_cases(runs, args.b0, args.t1, args.f1, report_dir(args.tuning), name)
    else:
        print("\n(failure_cases not written: B0/T1 overridden; the case lists are for b0 vs t1)")


if __name__ == "__main__":
    main()
