"""Second-round revision statistics for the ICIMCIS 2026 paper (REVISION_PLAN.md B2).

Runs on the dev-selected predictions (slice_eval.py with the inner dev split). Nothing is
retrained. Same per-tweet rules as slice_report.py / revision_stats.py: emoji tweets =
has_emoji (165), per-tweet predicted label = majority vote over seeds (binary, 3 seeds, so
majority correctness = correctness of the majority label), paired on (fold, val_index).
Fold-level numbers average seeds within a fold, then mean +/- population SD over the 5
folds (as in slice_report.py).

  0  Run integrity: runs per condition, split sizes, selection bookkeeping, best-epoch
     distribution, test folds identical across conditions, Tracebacks in the sweep log.
  a  Metrics (R2.4): accuracy, macro-F1, per-class precision / recall, all and emoji
     tweets; fold mean +/- SD and seed-majority, plus seed-majority confusion matrices.
  b  Paired bootstrap 95% CIs over tweets (R2.3) for the principal contrasts, accuracy and
     macro-F1 differences, with exact McNemar p. All 21 pairwise McNemar tests on emoji
     tweets, uncorrected (exploratory).
  c  Primary family (R2.3): the contrasts defined by the RQs, RQ1 B0 vs T1 and RQ2 T1 vs
     T1+L (master, cont.), both on emoji tweets, Holm-corrected. Everything else is
     exploratory and uncorrected.
  d  Power (R1.1): exact McNemar power and minimum detectable effect at the observed
     discordant rate; tweets needed for the observed effect (Connor 1987 approximation).
  e  Selection bias (response letter): test macro-F1 of the v1-style pick (best test
     epoch), the dev-selected epoch and the last epoch, per condition.
  f  Robustness: the primary family and the dilution numbers again with pred_last.

Writes results/report/<tuning>/revision2_stats.txt. Works on partial sweeps: incomplete
conditions are flagged and contrasts use the tweets both conditions have.

Usage:
    python analysis/revision2_stats.py --tuning lora
    python analysis/revision2_stats.py --tuning indobertweet_lora
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.dataset import N_SPLITS  # noqa: E402
from src.results_io import RESULTS_DIR, RUN_TAGS, preds_dir, report_dir  # noqa: E402

NAME = {
    "b0": "B0", "b1": "B1", "t1": "T1",
    "f1_master_ours": "T1+L (master, cont.)", "f1_master_ang": "T1+L (master, pres.)",
    "f1_esr_ours": "T1+L (ESR, cont.)", "f1_esr_ang": "T1+L (ESR, pres.)",
    "t1_pos": "T1-pos", "t1_neg": "T1-neg",
}
F1 = "f1_master_ours"
F1_ALL = ["f1_master_ours", "f1_master_ang", "f1_esr_ours", "f1_esr_ang"]
MAIN = ["b0", "b1", "t1"] + F1_ALL
SEEDS = (0, 1, 42)
N_BOOT = 10_000
ALPHA = 0.05
POWER = 0.80
SUBSETS = {"all": None, "emoji": True, "non-emoji": False}

# (a, b, subset, gold) -- gold None = both halves. Principal contrasts for the CIs.
PRIMARY = [("b0", "t1", "emoji", None), ("t1", F1, "emoji", None)]
PRINCIPAL = (
    [("b0", "t1", s, None) for s in ("all", "emoji", "non-emoji")]
    + [("b0", "b1", "emoji", None), ("b0", F1, "emoji", None)]
    + [("t1", c, "emoji", None) for c in F1_ALL]
    + [("t1", F1, "all", None)]
    + [(a, b, "emoji", g) for a, b in (("b0", "t1"), ("b0", "b1"), ("b0", F1), ("t1", F1))
       for g in (1, 0)]
)


# ---------------------------------------------------------------- loading
def load_meta(tuning):
    """meta[(condition, fold, seed)] = full JSON (records keyed by val_index)."""
    meta = {}
    for path in glob.glob(os.path.join(preds_dir(tuning), "*.json")):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        d["records"] = {r["val_index"]: r for r in d["records"]}
        meta[(d["condition"], d["fold"], d["seed"])] = d
    return meta


def in_subset(r, subset):
    flag = SUBSETS[subset]
    return flag is None or r["has_emoji"] == flag


def majority(meta, cond, subset, field="pred"):
    """(fold, val_index) -> (gold, majority pred). Ties (partial sweeps only) -> negative."""
    votes, gold = collections.defaultdict(list), {}
    for (c, fold, _), d in meta.items():
        if c != cond:
            continue
        for i, r in d["records"].items():
            if in_subset(r, subset):
                votes[(fold, i)].append(r[field])
                gold[(fold, i)] = r["true"]
    return {k: (gold[k], int(sum(v) > len(v) / 2)) for k, v in votes.items()}


def counts(y, p):
    """Binary confusion counts with positive = 1: tp, fp, fn, tn."""
    y, p = np.asarray(y), np.asarray(p)
    return (int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()),
            int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum()))


def metrics(y, p):
    tp, fp, fn, tn = counts(y, p)
    div = lambda a, b: a / b if b else 0.0  # noqa: E731
    pp, rp = div(tp, tp + fp), div(tp, tp + fn)
    pn, rn = div(tn, tn + fn), div(tn, tn + fp)
    f1p, f1n = div(2 * tp, 2 * tp + fp + fn), div(2 * tn, 2 * tn + fn + fp)
    return {"acc": (tp + tn) / len(y), "mf1": (f1p + f1n) / 2,
            "P_neg": pn, "R_neg": rn, "P_pos": pp, "R_pos": rp}


METRIC_KEYS = ["acc", "mf1", "P_neg", "R_neg", "P_pos", "R_pos"]


def fold_metrics(meta, cond, subset, field="pred"):
    """Metric -> (mean, population SD) over folds, seeds averaged within each fold."""
    per_fold = collections.defaultdict(list)
    for (c, fold, _), d in meta.items():
        if c != cond:
            continue
        rs = [r for r in d["records"].values() if in_subset(r, subset)]
        per_fold[fold].append(metrics([r["true"] for r in rs], [r[field] for r in rs]))
    out = {}
    for m in METRIC_KEYS:
        fv = [np.mean([x[m] for x in v]) for v in per_fold.values()]
        out[m] = (float(np.mean(fv)), float(np.std(fv)))
    return out


def paired(ma, mb, gold=None):
    keys = sorted(k for k in set(ma) & set(mb) if gold is None or ma[k][0] == gold)
    y = np.array([ma[k][0] for k in keys])
    return keys, y, np.array([ma[k][1] for k in keys]), np.array([mb[k][1] for k in keys])


def mcnemar_exact_p(n01, n10):
    n = n01 + n10
    return 1.0 if n == 0 else float(scipy_stats.binomtest(min(n01, n10), n, 0.5).pvalue)


def holm(pvals):
    """Holm step-down at ALPHA -> list of (adjusted p, rejected) in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out, running = [None] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = (running, running <= ALPHA)
    return out


def complete(meta, cond):
    return sum(1 for (c, _, _) in meta if c == cond) == N_SPLITS * len(SEEDS)


def label(meta, cond):
    n = sum(1 for (c, _, _) in meta if c == cond)
    full = N_SPLITS * len(SEEDS)
    return NAME.get(cond, cond) + ("" if n == full else f" [partial {n}/{full}]")


# ---------------------------------------------------------------- 0
def sec0(meta, tuning, lines):
    lines.append("=== 0: run integrity ===")
    conds = sorted({c for (c, _, _) in meta})
    full = N_SPLITS * len(SEEDS)
    for c in conds:
        ks = [k for k in meta if k[0] == c]
        missing = [(f, s) for f in range(N_SPLITS) for s in SEEDS if (c, f, s) not in meta]
        lines.append(f"  {NAME.get(c, c):22s} {len(ks):>3d}/{full} runs"
                     + (f"   missing (fold, seed): {missing}" if missing else ""))
    sizes = collections.Counter((d["n_train"], d["n_dev"], d["n_val"]) for d in meta.values())
    lines.append(f"  (n_train, n_dev, n_test) across runs: {dict(sizes)}")
    other = collections.Counter((d["tuning"], d["backbone"], d["learning_rate"])
                                for d in meta.values())
    lines.append(f"  (tuning, backbone, lr): {dict(other)}")
    sel = collections.Counter(d.get("selection", "MISSING") for d in meta.values())
    lines.append(f"  selection: {dict(sel)}")

    bad = []
    for k, d in meta.items():
        by_ep = d["test_macro_f1_by_epoch"]
        rs = list(d["records"].values())
        y = [r["true"] for r in rs]
        if abs(d["eval_macro_f1"] - by_ep[d["best_epoch"] - 1]) > 1e-9:
            bad.append((k, "eval_macro_f1 != test F1 at best_epoch"))
        if abs(metrics(y, [r["pred_last"] for r in rs])["mf1"] - by_ep[-1]) > 1e-9:
            bad.append((k, "pred_last != last epoch"))
        if abs(metrics(y, [r["pred"] for r in rs])["mf1"] - d["eval_macro_f1"]) > 1e-9:
            bad.append((k, "pred != eval_macro_f1"))
    lines.append(f"  bookkeeping mismatches: {len(bad)}" + "".join(f"\n    {b}" for b in bad))

    # Test folds: same tweets and gold labels for every run of a fold.
    ref, diff = {}, []
    for (c, f, s), d in meta.items():
        sig = tuple(sorted((i, r["true"], r["has_emoji"]) for i, r in d["records"].items()))
        if f in ref and ref[f][1] != sig:
            diff.append(((c, f, s), ref[f][0]))
        ref.setdefault(f, ((c, f, s), sig))
    n_emoji = sum(sum(1 for x in sig if x[2]) for _, sig in ref.values())
    lines.append(f"  test folds identical across runs: {'yes' if not diff else 'NO ' + str(diff)}"
                 f"; folds seen {sorted(ref)}; emoji tweets over those folds: {n_emoji}")

    lines.append("  best_epoch distribution (epoch: runs):")
    for c in conds:
        h = collections.Counter(d["best_epoch"] for k, d in meta.items() if k[0] == c)
        lines.append(f"    {NAME.get(c, c):22s} " + "  ".join(f"{e}:{h[e]}" for e in sorted(h)))

    log = os.path.join(RESULTS_DIR, "logs", "revision_sweep.log")
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        lines.append(f"  sweep log: {txt.count('Traceback')} Traceback(s), "
                     f"{txt.count('Error')} 'Error' string(s)")
    lines.append("")


# ---------------------------------------------------------------- a
def seca(meta, have, lines):
    lines.append("=== a: metrics (R2.4); positive class = 1 ===")
    for subset in ("all", "emoji"):
        lines.append(f"  [{subset} tweets] fold mean +/- SD (%, seeds averaged within fold)")
        lines.append(f"    {'condition':34s}" + "".join(f"{m:>15s}" for m in METRIC_KEYS))
        for c in have:
            fm = fold_metrics(meta, c, subset)
            lines.append(f"    {label(meta, c):34s}" + "".join(
                f"{fm[m][0] * 100:>8.2f}+/-{fm[m][1] * 100:<4.2f}" for m in METRIC_KEYS))
        lines.append(f"  [{subset} tweets] seed-majority per tweet (%), confusion matrix "
                     f"[[TN FP] [FN TP]] (rows = gold neg/pos)")
        lines.append(f"    {'condition':34s}" + "".join(f"{m:>8s}" for m in METRIC_KEYS)
                     + "   confusion")
        for c in have:
            mj = majority(meta, c, subset)
            y = [v[0] for v in mj.values()]
            p = [v[1] for v in mj.values()]
            m = metrics(y, p)
            tp, fp, fn, tn = counts(y, p)
            lines.append(f"    {label(meta, c):34s}" + "".join(
                f"{m[k] * 100:>8.2f}" for k in METRIC_KEYS)
                + f"   [[{tn:>3d} {fp:>3d}] [{fn:>3d} {tp:>3d}]]  n={len(y)}")
        lines.append("")


# ---------------------------------------------------------------- b
def boot(y, pa, pb, rng):
    """Bootstrap distributions of the accuracy and macro-F1 differences (b - a)."""
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    Y = y[idx]

    def stat(p):
        P = p[idx]
        acc = (P == Y).mean(axis=1)
        tp = ((Y == 1) & (P == 1)).sum(axis=1)
        tn = ((Y == 0) & (P == 0)).sum(axis=1)
        fp = ((Y == 0) & (P == 1)).sum(axis=1)
        fn = ((Y == 1) & (P == 0)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            f1p = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
            f1n = np.where(tn > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
        return acc, (f1p + f1n) / 2

    (aa, fa), (ab, fb) = stat(pa), stat(pb)
    return ab - aa, fb - fa


def contrast(meta, a, b, subset, gold, field="pred"):
    ma, mb = majority(meta, a, subset, field), majority(meta, b, subset, field)
    keys, y, pa, pb = paired(ma, mb, gold)
    ca, cb = pa == y, pb == y
    a_only, b_only = int((ca & ~cb).sum()), int((cb & ~ca).sum())
    d_acc, d_f1 = boot(y, pa, pb, np.random.default_rng(0))  # same CI wherever printed
    halves = gold is None or len(set(y)) > 1  # macro-F1 is undefined within one gold half
    return {
        "n": len(keys), "acc_a": ca.mean() * 100, "acc_b": cb.mean() * 100,
        "d": (cb.mean() - ca.mean()) * 100,
        "ci": tuple(np.percentile(d_acc, [2.5, 97.5]) * 100),
        "d_f1": (metrics(y, pb)["mf1"] - metrics(y, pa)["mf1"]) * 100 if halves else None,
        "ci_f1": tuple(np.percentile(d_f1, [2.5, 97.5]) * 100) if halves else None,
        "b_only": b_only, "a_only": a_only, "p": mcnemar_exact_p(a_only, b_only),
    }


def cname(a, b, subset, gold):
    g = "" if gold is None else (" pos-gold" if gold == 1 else " neg-gold")
    return f"{NAME[a]} -> {NAME[b]}", f"{subset}{g}"


def fmt_contrast(meta, a, b, subset, gold, r):
    left, sub = cname(a, b, subset, gold)
    f1 = ("" if r["d_f1"] is None else
          f"  dF1 {r['d_f1']:>+6.2f} [{r['ci_f1'][0]:>+6.2f}, {r['ci_f1'][1]:>+6.2f}]")
    flag = "" if complete(meta, a) and complete(meta, b) else "  (partial)"
    return (f"    {left:32s} {sub:16s} n={r['n']:>3d}  {r['acc_a']:6.2f} -> {r['acc_b']:6.2f}  "
            f"dAcc {r['d']:>+6.2f} [{r['ci'][0]:>+6.2f}, {r['ci'][1]:>+6.2f}]{f1}  "
            f"b/a {r['b_only']:>2d}/{r['a_only']:<2d} p={r['p']:.4f}{flag}")


def secb(meta, have, lines):
    lines.append(f"=== b: paired bootstrap 95% CIs over tweets (seed-majority, {N_BOOT} "
                 f"resamples, rng seed 0 per contrast), pp; b/a = tweets only b / only a gets right; "
                 f"exact McNemar p, uncorrected ===")
    for a, b, subset, gold in PRINCIPAL:
        if a in have and b in have:
            lines.append(fmt_contrast(meta, a, b, subset, gold,
                                      contrast(meta, a, b, subset, gold)))
    lines.append("  (macro-F1 difference omitted within a single gold half: one class only)")

    lines.append("\n  All pairwise exact McNemar tests, emoji tweets (EXPLORATORY, uncorrected)")
    conds = [c for c in MAIN if c in have]
    for i, a in enumerate(conds):
        for b in conds[i + 1:]:
            _, y, pa, pb = paired(majority(meta, a, "emoji"), majority(meta, b, "emoji"))
            ca, cb = pa == y, pb == y
            a_only, b_only = int((ca & ~cb).sum()), int((cb & ~ca).sum())
            lines.append(f"    {NAME[a] + ' vs ' + NAME[b]:46s} b/a {b_only:>2d}/{a_only:<2d} "
                         f"p={mcnemar_exact_p(a_only, b_only):.4f}")
    lines.append("")


# ---------------------------------------------------------------- c
def primary_family(meta, have, field="pred"):
    res = []
    for a, b, subset, gold in PRIMARY:
        if a in have and b in have:
            res.append(((a, b, subset, gold), contrast(meta, a, b, subset, gold, field)))
    adj = holm([r["p"] for _, r in res])
    return res, adj


def secc(meta, have, lines, field="pred", title="c: primary family (R2.3)"):
    lines.append(f"=== {title} ===")
    lines.append("  Primary contrasts defined by the RQs: RQ1 B0 vs T1, RQ2 T1 vs T1+L (master, "
                 "cont.), emoji tweets. Holm over the family; all other tests exploratory.")
    res, adj = primary_family(meta, have, field)
    for ((a, b, subset, gold), r), (padj, rej) in zip(res, adj):
        lines.append(fmt_contrast(meta, a, b, subset, gold, r))
        lines.append(f"      Holm-adjusted p = {padj:.4f}  -> "
                     f"{'significant' if rej else 'not significant'} at {ALPHA}")
    lines.append("")
    return res


# ---------------------------------------------------------------- d
def reject_table(n, alpha):
    """reject[nd] = boolean array over b = 0..nd: exact two-sided McNemar rejects."""
    out = []
    for nd in range(n + 1):
        b = np.arange(nd + 1)
        if nd == 0:
            out.append(np.zeros(1, dtype=bool))
            continue
        lo = scipy_stats.binom.cdf(np.minimum(b, nd - b), nd, 0.5)
        out.append(np.minimum(1.0, 2 * lo) <= alpha)
    return out


def exact_power(n, p_disc, delta, rej):
    """Power of the exact McNemar test: discordant count ~ Bin(n, p_disc); among them,
    'b better' ~ Bin(nd, (p_disc + delta) / (2 p_disc))."""
    if delta > p_disc:
        return float("nan")
    pi = (p_disc + delta) / (2 * p_disc)
    pnd = scipy_stats.binom.pmf(np.arange(n + 1), n, p_disc)
    return float(sum(pnd[nd] * scipy_stats.binom.pmf(np.arange(nd + 1), nd, pi)[rej[nd]].sum()
                     for nd in range(n + 1) if pnd[nd] > 1e-12))


def mde(n, p_disc, rej):
    """Smallest accuracy difference with power >= POWER; None if unreachable even when
    every discordant tweet favours one side (delta = p_disc)."""
    for d in np.arange(0.001, p_disc + 1e-9, 0.001):
        if exact_power(n, p_disc, d, rej) >= POWER:
            return float(d)
    return None


def connor_n(p_disc, delta, alpha):
    za, zb = scipy_stats.norm.ppf(1 - alpha / 2), scipy_stats.norm.ppf(POWER)
    return (za * np.sqrt(p_disc) + zb * np.sqrt(p_disc - delta ** 2)) ** 2 / delta ** 2


def fmt_mde(n, p_disc, rej):
    m = mde(n, p_disc, rej)
    if m is None:
        return (f"unreachable (max power {exact_power(n, p_disc, p_disc, rej):.2f}, "
                f"all discordant one way)")
    return f"{m * 100:.1f}pp (~{m * n:.0f} net tweets)"


def secd(meta, have, primary, lines):
    lines.append(f"=== d: power of the exact McNemar test (R1.1), two-sided, target power "
                 f"{POWER:.0%} ===")
    lines.append("  Effect = accuracy difference; the discordant rate is held fixed (observed, "
                 "then a grid of assumed rates).")
    for (a, b, subset, gold), r in primary:
        n = r["n"]
        p_disc = (r["a_only"] + r["b_only"]) / n
        delta = abs(r["d"]) / 100
        left, sub = cname(a, b, subset, gold)
        lines.append(f"  {left} ({sub}): n={n}, discordant {r['a_only'] + r['b_only']} "
                     f"({p_disc * 100:.1f}%), observed |dAcc| = {delta * 100:.2f}pp")
        for alpha, why in ((ALPHA, "uncorrected"), (ALPHA / 2, "Holm, smaller p of 2")):
            rej = reject_table(n, alpha)
            if delta > 0:
                pw = exact_power(n, p_disc, delta, rej)
                need = f"~{connor_n(p_disc, delta, alpha):.0f} tweets"
            else:
                pw, need = alpha, "n/a (observed effect 0)"
            lines.append(f"    alpha={alpha:.3f} ({why}): power at observed effect {pw:.2f}; "
                         f"MDE at observed discordance {fmt_mde(n, p_disc, rej)}; "
                         f"needed for observed effect at {POWER:.0%}: {need}")
    if primary:
        n = primary[0][1]["n"]
        rej = reject_table(n, ALPHA)
        lines.append(f"  MDE at n={n}, alpha={ALPHA}, by assumed discordant rate:")
        for p_disc in (0.05, 0.10, 0.15, 0.20):
            lines.append(f"    discordant {p_disc * 100:>4.0f}%: {fmt_mde(n, p_disc, rej)}")
    lines.append("")


# ---------------------------------------------------------------- e
def sece(meta, lines):
    lines.append("=== e: epoch-selection bias, test macro-F1 over ALL test tweets (%), mean over "
                 "runs ===")
    lines.append("  v1-style = best test epoch (the v1 rule); dev = dev-selected epoch "
                 "(reported); last = epoch 5. Same 648-tweet training sets throughout; the v1 "
                 "runs themselves trained on 720.")
    lines.append(f"    {'condition':22s} {'runs':>4s} {'v1-style':>9s} {'dev':>7s} {'last':>7s} "
                 f"{'v1-dev':>7s} {'max v1-dev':>10s} {'dev=v1 pick':>12s}")
    allv = []
    for c in sorted({k[0] for k in meta}):
        ds = [d for k, d in meta.items() if k[0] == c]
        v1 = [max(d["test_macro_f1_by_epoch"]) for d in ds]
        dv = [d["test_macro_f1_by_epoch"][d["best_epoch"] - 1] for d in ds]
        la = [d["test_macro_f1_by_epoch"][-1] for d in ds]
        gap = [x - y for x, y in zip(v1, dv)]
        allv += gap
        same = sum(1 for g in gap if g < 1e-12)
        lines.append(f"    {NAME.get(c, c):22s} {len(ds):>4d} {np.mean(v1) * 100:>9.2f} "
                     f"{np.mean(dv) * 100:>7.2f} {np.mean(la) * 100:>7.2f} "
                     f"{np.mean(gap) * 100:>+7.2f} {max(gap) * 100:>+10.2f} "
                     f"{same:>5d}/{len(ds):<6d}")
    lines.append(f"  over all runs: mean v1-dev gap {np.mean(allv) * 100:+.2f}pp")
    lines.append("")


# ---------------------------------------------------------------- f
def secf(meta, have, lines):
    lines.append("=== f: robustness, last-epoch predictions (pred_last, no selection) ===")
    secc(meta, have, lines, field="pred_last", title="f.1: primary family with pred_last")
    lines.append("  f.2: accuracy (%) by subset, fold mean +/- SD: selected (pred) vs last "
                 "(pred_last)")
    lines.append(f"    {'condition':34s}" + "".join(f"{s + ' sel':>16s}{s + ' last':>16s}"
                                                    for s in SUBSETS))
    for c in have:
        cells = ""
        for s in SUBSETS:
            for field in ("pred", "pred_last"):
                m, sd = fold_metrics(meta, c, s, field)["acc"]
                cells += f"{m * 100:>10.2f}+/-{sd * 100:<3.1f}"
        lines.append(f"    {label(meta, c):34s}{cells}")
    lines.append("  f.3: dilution (B0 -> T1 gain, fold means, pp)")
    if "b0" in have and "t1" in have:
        for field in ("pred", "pred_last"):
            g = {s: (fold_metrics(meta, "t1", s, field)["acc"][0]
                     - fold_metrics(meta, "b0", s, field)["acc"][0]) * 100 for s in SUBSETS}
            ratio = g["emoji"] / g["all"] if abs(g["all"]) > 1e-9 else float("nan")
            lines.append(f"    {field:10s} all {g['all']:+.2f}  emoji {g['emoji']:+.2f}  "
                         f"non-emoji {g['non-emoji']:+.2f}  emoji/all {ratio:.1f}x")
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", default="lora", choices=RUN_TAGS)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    meta = load_meta(args.tuning)
    if not meta:
        print(f"No prediction files in {preds_dir(args.tuning)}.")
        return
    present = {k[0] for k in meta}
    have = [c for c in MAIN + ["t1_pos", "t1_neg"] if c in present]

    lines = [f"Revision-2 statistics [{args.tuning}] (analysis/revision2_stats.py)",
             f"T1+L = {F1}; conditions loaded: {have}", ""]
    sec0(meta, args.tuning, lines)
    seca(meta, have, lines)
    secb(meta, have, lines)
    primary = secc(meta, have, lines)
    secd(meta, have, primary, lines)
    sece(meta, lines)
    secf(meta, have, lines)

    text = "\n".join(lines)
    print(text)
    out_dir = report_dir(args.tuning)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "revision2_stats.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
