"""Revision statistics for the ICIMCIS 2026 paper (revision_actions.md DAT-1 .. DAT-6).

All numbers come from existing predictions; nothing is retrained. Same rules as
slice_report.py / polarity_gain.py: emoji slice = has_emoji (165 tweets), per-tweet
correctness and predicted label = majority vote over seeds, paired on (fold, val_index).
F1 = f1_master_ours (the single "T1+L" of the paper).

  DAT-1  Between-halves polarity test: Fisher exact on the discordant (helped/hurt)
         tweets split by gold half; direction-of-flip binomial; T1-pos on negative-gold
         tweets carrying a master-positive emoji; Bonferroni / Holm over candidate families.
  DAT-2  Predicted-positive rate, per-half recall and macro-F1 on the emoji slice.
  DAT-3  Paired bootstrap 95% CI (over emoji tweets) of accuracy differences; paired
         fold-level differences (mean, sample SD); fold-mean vs per-tweet gap.
  DAT-4  Per-seed McNemar (each seed's own predictions, no majority vote).
  DAT-5  Duplicate / retweet check over the 900 raw tweets, and whether copies cross folds.
  DAT-6  Breakdown of the tweets on which the master and ESR presence features differ.

Writes results/report/<tuning>/revision_stats.txt.

Usage:
    python analysis/revision_stats.py --tuning lora
    python analysis/revision_stats.py --tuning full
"""
import argparse
import collections
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.polarity_gain import per_tweet  # noqa: E402
from slice_report import fold_means, mcnemar_exact_p  # noqa: E402
from src.dataset import N_SPLITS, _load_raw  # noqa: E402
from src.models.indobert import TUNING_MODES  # noqa: E402
from src.preprocess import _normalize_tag_key, _tags, extract_emoji_feature  # noqa: E402
from src.results_io import load_runs, preds_dir, report_dir  # noqa: E402

from scipy import stats as scipy_stats  # noqa: E402

NAME = {
    "b0": "B0", "b1": "B1", "t1": "T1",
    "f1_master_ours": "T1+L (master, cont.)", "f1_master_ang": "T1+L (master, pres.)",
    "f1_esr_ours": "T1+L (ESR, cont.)", "f1_esr_ang": "T1+L (ESR, pres.)",
    "t1_pos": "T1-pos", "t1_neg": "T1-neg",
}
F1 = "f1_master_ours"
F1_ALL = ["f1_esr_ang", "f1_esr_ours", "f1_master_ang", "f1_master_ours"]
RQ1_PAIRS = [("b0", "b1"), ("b0", "t1"), ("b0", F1), ("b1", "t1"), ("b1", F1), ("t1", F1)]
N_BOOT = 10_000
ALPHA = 0.05


def binom_p(k, n):
    return 1.0 if n == 0 else float(scipy_stats.binomtest(k, n, 0.5).pvalue)


def transitions(pa, pb, keys):
    """helped = a wrong -> b right; hurt = a right -> b wrong."""
    helped = sum(1 for k in keys if not pa[k]["correct"] and pb[k]["correct"])
    hurt = sum(1 for k in keys if pa[k]["correct"] and not pb[k]["correct"])
    return helped, hurt


def holm(pvals):
    """Holm step-down at ALPHA. Returns list of (p, threshold, rejected) in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [None] * m
    still = True
    for rank, i in enumerate(order):
        thr = ALPHA / (m - rank)
        rej = still and pvals[i] <= thr
        if not rej:
            still = False
        out[i] = (pvals[i], thr, rej)
    return out


# ---------------------------------------------------------------- DAT-1
def dat1(pt, have, lines):
    lines.append("=== DAT-1: polarity asymmetry (emoji slice, seed-majority) ===")
    lines.append("  Fisher exact on discordant tweets: rows = gold half, cols = helped/hurt vs B0")
    lines.append(f"    {'contrast':26s} {'pos h/h':>8s} {'neg h/h':>8s} {'Fisher p':>9s}  "
                 f"{'McNemar pos':>11s} {'McNemar neg':>11s}")
    rq3 = {}
    for b in [c for c in ["t1", F1, "b1", "t1_pos", "t1_neg"] if c in have]:
        res = {}
        for g in (1, 0):
            keys = [k for k in pt["b0"] if pt["b0"][k]["gold"] == g]
            res[g] = transitions(pt["b0"], pt[b], keys)
        table = [[res[1][0], res[1][1]], [res[0][0], res[0][1]]]
        fp = float(scipy_stats.fisher_exact(table).pvalue)
        mp = {g: mcnemar_exact_p(res[g][0], res[g][1]) for g in (1, 0)}
        rq3[b] = (mp[1], mp[0], fp)
        lines.append(f"    {'B0 -> ' + NAME[b]:26s} {res[1][0]:>3d}/{res[1][1]:<4d} "
                     f"{res[0][0]:>3d}/{res[0][1]:<4d} {fp:>9.4f}  {mp[1]:>11.4f} {mp[0]:>11.4f}")

    lines.append("\n  Direction of changed predictions vs B0 (two-sided binomial, H0: 50/50)")
    for b in [c for c in ["t1", F1, "b1", "t1_pos", "t1_neg"] if c in have]:
        keys = list(pt["b0"])
        to_pos = sum(1 for k in keys if pt["b0"][k]["pred"] == 0 and pt[b][k]["pred"] == 1)
        to_neg = sum(1 for k in keys if pt["b0"][k]["pred"] == 1 and pt[b][k]["pred"] == 0)
        n = to_pos + to_neg
        lines.append(f"    {'B0 -> ' + NAME[b]:26s} toward positive {to_pos:>2d} of {n:>2d}  "
                     f"p = {binom_p(to_pos, n):.4f}")

    if "t1_pos" in have:
        from data.reverse_map import load_tag_to_sentiment
        t2s = load_tag_to_sentiment()
        keys = [k for k in pt["b0"] if pt["b0"][k]["gold"] == 0 and any(
            t2s.get(_normalize_tag_key(t)) == "positive" for t in _tags(pt["b0"][k]["text"]))]
        helped, hurt = transitions(pt["b0"], pt["t1_pos"], keys)
        lines.append(f"\n  T1-pos on negative-gold tweets with a master-positive emoji: n={len(keys)}  "
                     f"broken={hurt} corrected={helped}  McNemar/binomial p = "
                     f"{mcnemar_exact_p(helped, hurt):.4f}")

    # Correction families.
    rq1 = {}
    for a, b in RQ1_PAIRS:
        keys = sorted(set(pt[a]) & set(pt[b]))
        a_only = sum(1 for k in keys if pt[a][k]["correct"] and not pt[b][k]["correct"])
        b_only = sum(1 for k in keys if pt[b][k]["correct"] and not pt[a][k]["correct"])
        rq1[f"{NAME[a]} vs {NAME[b]}"] = mcnemar_exact_p(a_only, b_only)
    t1f1 = {}
    for g, gname in ((1, "pos"), (0, "neg")):
        keys = [k for k in pt["t1"] if pt["t1"][k]["gold"] == g]
        h, u = transitions(pt["t1"], pt[F1], keys)
        t1f1[gname] = mcnemar_exact_p(h, u)

    fams = {
        "RQ1: six McNemar contrasts, all emoji tweets": rq1,
        "RQ3 (2): positive halves, B0->T1 and B0->T1+L": {
            "B0->T1 pos": rq3["t1"][0], "B0->T1+L pos": rq3[F1][0]},
        "RQ3 (4): B0->T1 and B0->T1+L, both halves": {
            "B0->T1 pos": rq3["t1"][0], "B0->T1 neg": rq3["t1"][1],
            "B0->T1+L pos": rq3[F1][0], "B0->T1+L neg": rq3[F1][1]},
        "RQ3 (8): B0->T1, B0->B1, B0->T1+L, T1->T1+L, both halves": {
            "B0->T1 pos": rq3["t1"][0], "B0->T1 neg": rq3["t1"][1],
            "B0->B1 pos": rq3["b1"][0], "B0->B1 neg": rq3["b1"][1],
            "B0->T1+L pos": rq3[F1][0], "B0->T1+L neg": rq3[F1][1],
            "T1->T1+L pos": t1f1["pos"], "T1->T1+L neg": t1f1["neg"]},
        "Between-halves (2): Fisher, B0->T1 and B0->T1+L": {
            "Fisher B0->T1": rq3["t1"][2], "Fisher B0->T1+L": rq3[F1][2]},
    }
    lines.append("\n  Multiple-comparison families (alpha = 0.05)")
    for fam, d in fams.items():
        names, ps = list(d), list(d.values())
        m = len(ps)
        h = holm(ps)
        lines.append(f"    {fam}  [m={m}, Bonferroni threshold {ALPHA / m:.4f}]")
        for nm, p, (_, thr, rej) in zip(names, ps, h):
            bonf = "survives" if p <= ALPHA / m else "fails"
            lines.append(f"      {nm:22s} p = {p:.4f}   Bonferroni {bonf:8s}  "
                         f"Holm (thr {thr:.4f}) {'survives' if rej else 'fails'}")
    lines.append("")


# ---------------------------------------------------------------- DAT-2
def dat2(pt, have, lines):
    lines.append("=== DAT-2: predicted-positive rate, recall by half, macro-F1 (emoji slice, "
                 "seed-majority label) ===")
    any_c = next(iter(pt.values()))
    n = len(any_c)
    gold_pos = sum(1 for v in any_c.values() if v["gold"] == 1)
    lines.append(f"  gold-positive share: {gold_pos}/{n} = {gold_pos / n * 100:.2f}%")
    lines.append(f"    {'condition':22s} {'pred-pos rate':>14s} {'pos recall':>11s} "
                 f"{'neg recall':>11s} {'macro-F1':>9s} {'accuracy':>9s}")
    for c in [c for c in ["b0", "b1", "t1"] + F1_ALL + ["t1_pos", "t1_neg"] if c in have]:
        d = pt[c]
        tp = sum(1 for v in d.values() if v["gold"] == 1 and v["pred"] == 1)
        tn = sum(1 for v in d.values() if v["gold"] == 0 and v["pred"] == 0)
        fp = sum(1 for v in d.values() if v["gold"] == 0 and v["pred"] == 1)
        fn = sum(1 for v in d.values() if v["gold"] == 1 and v["pred"] == 0)
        f1p = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        f1n = 2 * tn / (2 * tn + fn + fp) if tn else 0.0
        lines.append(f"    {NAME[c]:22s} {(tp + fp) / n * 100:>13.2f}% "
                     f"{tp / (tp + fn) * 100:>10.2f}% {tn / (tn + fp) * 100:>10.2f}% "
                     f"{(f1p + f1n) / 2 * 100:>9.2f} {(tp + tn) / n * 100:>9.2f}")
    lines.append("")


# ---------------------------------------------------------------- DAT-3
def boot_ci(pa, pb, keys, rng):
    a = np.array([pa[k]["correct"] for k in keys], dtype=float)
    b = np.array([pb[k]["correct"] for k in keys], dtype=float)
    d = b - a
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return d.mean() * 100, lo * 100, hi * 100


def dat3(runs, pt, have, lines):
    lines.append(f"=== DAT-3: paired bootstrap 95% CI of accuracy difference (pp), emoji "
                 f"tweets, seed-majority, {N_BOOT} resamples, rng seed 0 ===")
    rng = np.random.default_rng(0)
    contrasts = [("b0", "t1"), ("b0", "b1"), ("b0", F1)] + [("t1", c) for c in F1_ALL]
    lines.append(f"    {'contrast':34s} {'subset':6s} {'n':>4s} {'diff':>7s} {'95% CI':>18s}")
    for a, b in contrasts:
        if a not in have or b not in have:
            continue
        subsets = [("all", None)]
        if (a, b) in [("b0", "t1"), ("b0", F1), ("b0", "b1")]:
            subsets += [("pos", 1), ("neg", 0)]
        for sname, g in subsets:
            keys = sorted(k for k in pt[a] if g is None or pt[a][k]["gold"] == g)
            m, lo, hi = boot_ci(pt[a], pt[b], keys, rng)
            lines.append(f"    {NAME[a] + ' -> ' + NAME[b]:34s} {sname:6s} {len(keys):>4d} "
                         f"{m:>+7.2f} [{lo:>+6.2f}, {hi:>+6.2f}]")

    lines.append("\n  Paired fold-level differences, emoji slice (seed-averaged per fold; n=5 folds)")
    lines.append(f"    {'contrast':34s} {'fold diffs (pp)':>36s} {'mean':>7s} {'sample SD':>10s}")
    for a, b in contrasts:
        if a not in have or b not in have:
            continue
        fa, fb = fold_means(runs, a, True), fold_means(runs, b, True)
        diffs = [(fb[f] - fa[f]) * 100 for f in sorted(fa)]
        lines.append(f"    {NAME[a] + ' -> ' + NAME[b]:34s} "
                     f"{' '.join(f'{x:+6.2f}' for x in diffs):>36s} "
                     f"{np.mean(diffs):>+7.2f} {np.std(diffs, ddof=1):>10.2f}")

    lines.append("\n  Fold-mean vs per-tweet (seed-majority) emoji-slice difference")
    for a, b in [("b0", "t1"), ("b0", F1)]:
        fa, fb = fold_means(runs, a, True), fold_means(runs, b, True)
        fm = (np.mean(list(fb.values())) - np.mean(list(fa.values()))) * 100
        keys = sorted(pt[a])
        pa = np.mean([pt[a][k]["correct"] for k in keys]) * 100
        pb = np.mean([pt[b][k]["correct"] for k in keys]) * 100
        lines.append(f"    {NAME[a] + ' -> ' + NAME[b]:34s} fold-mean {fm:+.2f}  per-tweet "
                     f"{pb - pa:+.2f}  ({NAME[a]} {pa:.2f} -> {NAME[b]} {pb:.2f})")
    lines.append("")


# ---------------------------------------------------------------- DAT-4
def seed_correct(runs, cond, seed):
    out = {}
    for (c, fold, s), recs in runs.items():
        if c != cond or s != seed:
            continue
        for i, r in recs.items():
            if r["has_emoji"]:
                out[(fold, i)] = {"correct": r["pred"] == r["true"], "gold": r["true"]}
    return out


def dat4(runs, have, lines):
    lines.append("=== DAT-4: per-seed exact McNemar, emoji slice (each seed's own predictions) ===")
    seeds = sorted({s for (_, _, s) in runs})
    contrasts = [(a, b, None) for a, b in RQ1_PAIRS]
    contrasts += [("t1", c, None) for c in F1_ALL if c != F1]
    contrasts += [("f1_master_ang", "f1_master_ours", None), ("f1_esr_ang", "f1_esr_ours", None)]
    contrasts += [("b0", "t1", 1), ("b0", "t1", 0), ("b0", F1, 1), ("b0", F1, 0)]
    hdr = "".join(f"  seed {s:<3d} b/a   p     " for s in seeds)
    lines.append(f"    {'contrast (a -> b)':42s}{hdr}")
    for a, b, g in contrasts:
        if a not in have or b not in have:
            continue
        label = f"{NAME[a]} -> {NAME[b]}" + ("" if g is None else f" [{'pos' if g else 'neg'}]")
        cells = []
        for s in seeds:
            ca, cb = seed_correct(runs, a, s), seed_correct(runs, b, s)
            keys = [k for k in ca if k in cb and (g is None or ca[k]["gold"] == g)]
            a_only = sum(1 for k in keys if ca[k]["correct"] and not cb[k]["correct"])
            b_only = sum(1 for k in keys if cb[k]["correct"] and not ca[k]["correct"])
            cells.append(f"  {b_only:>3d}/{a_only:<3d} {mcnemar_exact_p(a_only, b_only):.4f}  ")
        lines.append(f"    {label:42s}{''.join(cells)}")
    lines.append("  (b/a = tweets only b gets right / only a gets right)")

    lines.append("\n  Per-seed between-halves Fisher (helped/hurt vs B0 by gold half)")
    for b in ("t1", F1):
        cells = []
        for s in seeds:
            c0, cb = seed_correct(runs, "b0", s), seed_correct(runs, b, s)
            t = []
            for g in (1, 0):
                keys = [k for k in c0 if k in cb and c0[k]["gold"] == g]
                t.append([sum(1 for k in keys if not c0[k]["correct"] and cb[k]["correct"]),
                          sum(1 for k in keys if c0[k]["correct"] and not cb[k]["correct"])])
            p = float(scipy_stats.fisher_exact(t).pvalue)
            cells.append(f"  seed {s}: pos {t[0][0]}/{t[0][1]} neg {t[1][0]}/{t[1][1]} p={p:.4f}")
        lines.append(f"    B0 -> {NAME[b]:22s}{''.join(cells)}")
    lines.append("")


# ---------------------------------------------------------------- DAT-5
def _norm(t):
    t = t.lower()
    t = re.sub(r"^rt\s+@\w+\s*:?\s*", "", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^\w<>\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def dat5(lines):
    from sklearn.model_selection import StratifiedKFold
    lines.append("=== DAT-5: duplicate / retweet check, 900 raw tweets ===")
    df, _ = _load_raw("pilkada")
    df = df.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    fold_of = {}
    for f, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        for i in val_idx:
            fold_of[int(i)] = f
    texts = df["text"].astype(str).tolist()
    n_rt = sum(1 for t in texts if re.match(r"^\s*rt\s+@", t, re.I))
    lines.append(f"  tweets starting with 'RT @': {n_rt}")
    for level, key in (("exact text", lambda t: t.strip()), ("normalized", _norm)):
        groups = collections.defaultdict(list)
        for i, t in enumerate(texts):
            groups[key(t)].append(i)
        dups = [g for g in groups.values() if len(g) > 1]
        extra = sum(len(g) - 1 for g in dups)
        cross = [g for g in dups if len({fold_of[i] for i in g}) > 1]
        mixed = [g for g in dups if len({int(df.loc[i, 'label']) for i in g}) > 1]
        emoji = [g for g in dups if any(re.search(r"<[^>]+>", texts[i]) for i in g)]
        lines.append(f"  {level:10s}: {len(dups)} duplicate groups, {extra} redundant tweets; "
                     f"{len(cross)} groups span >1 fold; {len(mixed)} have conflicting labels; "
                     f"{len(emoji)} contain emoji")
        for g in dups[:15]:
            lines.append(f"      rows {g} folds {[fold_of[i] for i in g]} labels "
                         f"{[int(df.loc[i, 'label']) for i in g]}: {texts[g[0]][:90]!r}")
    lines.append("")


# ---------------------------------------------------------------- DAT-6
def dat6(lines):
    from data.esr_map import load_emoji_to_esr
    from data.reverse_map import load_tag_to_emoji, load_tag_to_sentiment
    lines.append("=== DAT-6: master vs ESR presence features, where they differ (900 tweets) ===")
    t2e, t2s, e2r = load_tag_to_emoji(), load_tag_to_sentiment(), load_emoji_to_esr()
    df, _ = _load_raw("pilkada")
    cls_name = {0: "pos", 1: "neg", 2: "none"}

    def esr_cls(tag):
        ch = t2e.get(_normalize_tag_key(tag))
        tr = e2r.get(ch) if ch else None
        if tr is None:
            return "uncovered"
        return {0: "neg", 1: "neut", 2: "pos"}[max(range(3), key=lambda i: tr[i])]

    directions = collections.Counter()
    by_emoji = collections.defaultdict(collections.Counter)
    n = 0
    for t in df["text"].astype(str):
        m = extract_emoji_feature(t, "f1_master_ang", tag_to_sentiment=t2s)
        e = extract_emoji_feature(t, "f1_esr_ang", tag_to_emoji=t2e, emoji_to_esr=e2r)
        if m == e:
            continue
        n += 1
        d = f"master {cls_name[m.index(1.0)]} -> ESR {cls_name[e.index(1.0)]}"
        directions[d] += 1
        for tag in sorted({_normalize_tag_key(x) for x in _tags(t)}):
            ms = t2s.get(tag, "missing")
            es = esr_cls(tag)
            if ms[:3] != es[:3]:
                by_emoji[d][f"{tag} (master {ms[:3]}, ESR {es})"] += 1
    lines.append(f"  differing tweets: {n}")
    for d, c in directions.most_common():
        lines.append(f"    {d}: {c} tweets; emoji whose master and ESR classes disagree "
                     f"(tweets containing each):")
        for name, k in by_emoji[d].most_common():
            lines.append(f"        {k:>3d}  {name}")

    # Master-lexicon agreement with gold (cited in RQ2 and Limitations).
    only = {"negative": collections.Counter(), "positive": collections.Counter()}
    for t, y in zip(df["text"].astype(str), df["label"]):
        s = {t2s.get(_normalize_tag_key(x)) for x in _tags(t)}
        if len(s) == 1 and next(iter(s)) in only:
            only[next(iter(s))][int(y)] += 1
    for pol, c in only.items():
        lines.append(f"  tweets with only master-{pol} emoji: {sum(c.values())}  "
                     f"gold-negative {c[0]}, gold-positive {c[1]}")
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", default="lora", choices=TUNING_MODES)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    runs = load_runs(args.tuning)
    if not runs:
        print(f"No prediction files in {preds_dir(args.tuning)}.")
        return
    have = {k[0] for k in runs}
    pt = {c: per_tweet(runs, c) for c in have}

    lines = [f"Revision statistics [{args.tuning}] (analysis/revision_stats.py)",
             f"F1 / T1+L = {F1}; conditions loaded: {sorted(have)}", ""]
    dat1(pt, have, lines)
    dat2(pt, have, lines)
    dat3(runs, pt, have, lines)
    dat4(runs, have, lines)
    dat5(lines)
    dat6(lines)

    text = "\n".join(lines)
    print(text)
    out_dir = report_dir(args.tuning)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "revision_stats.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
