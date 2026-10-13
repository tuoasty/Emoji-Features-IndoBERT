"""Per-emoji predictiveness (paper Sec. III-D) of the gold label + polarity asymmetry.

Formalizes the tongue-emoji observation at dataset scale. For each emoji
TYPE, how strongly does its presence in a tweet predict the gold polarity, and does
the emoji's LEXICON valence (ESR / master sign) agree with its DEPLOYED stance (gold)?

Method:
  - Built on results/report/eda/eda_emoji_table.csv (corpus_eda.py) — tweet-level gold_pos /
    gold_neg per emoji tag — so it is byte-consistent with the EDA. Corpus = 450 pos /
    450 neg (base rate exactly 0.5), from eda_stats.json.
  - PMI toward positive:  PMI+ = log2( P(pos | tag) / 0.5 ).  >0 => skews positive,
    <0 => skews negative, magnitude = strength. (Base 0.5 because the corpus is balanced.)
  - Association test: Fisher's EXACT test on the 2x2 [tag present/absent] x [pos/neg]
    over all 900 tweets — exact, not chi-square, because per-emoji cells are tiny
    (same stance as the exact McNemar in slice_report.py). chi-square reported too but
    flagged invalid for small cells.
  - MIN_TWEETS threshold: emoji seen in < MIN_TWEETS tweets are listed but excluded
    from the highlighted / summary analysis.

Payoff:
  - VALENCE vs STANCE table: emoji where ESR sign (valence) disagrees with gold skew
    (stance) — the mocking cluster, whose valence a lexicon feature imports (see
    fusion_mcnemar.py and the error taxonomy). Turns the qualitative tongue reading into a table.
  - Polarity asymmetry: do lexicon-POSITIVE emoji predict pos-gold as strongly as
    lexicon-NEGATIVE emoji predict neg-gold?

Run:  python analysis/emoji_predictiveness.py
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(HERE, "results", "report", "eda")
TABLE = os.path.join(ANA, "eda_emoji_table.csv")
STATS = os.path.join(ANA, "eda_stats.json")

MIN_TWEETS = 5   # min tweets for the highlighted / summary analysis

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


def fisher_p(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]. Manual hypergeometric fallback."""
    if scipy_stats is not None:
        return float(scipy_stats.fisher_exact([[a, b], [c, d]])[1])
    # fallback: sum hypergeometric probs <= observed (two-sided)
    from math import comb
    n = a + b + c + d
    row1, col1 = a + b, a + c
    def hp(x):
        return (comb(col1, x) * comb(n - col1, row1 - x)) / comb(n, row1)
    p_obs = hp(a)
    lo, hi = max(0, row1 - (n - col1)), min(row1, col1)
    return min(1.0, sum(hp(x) for x in range(lo, hi + 1) if hp(x) <= p_obs + 1e-12))


def chi2_p(a, b, c, d):
    if scipy_stats is None:
        return float("nan")
    return float(scipy_stats.chi2_contingency([[a, b], [c, d]], correction=True)[1])


def pmi_pos(gp, gn):
    """log2( P(pos|tag) / 0.5 ). +inf/-inf when a cell is 0."""
    n = gp + gn
    p = gp / n
    if p == 0:
        return float("-inf")
    if p == 1:
        return float("inf")
    return math.log2(p / 0.5)


def load():
    with open(STATS, encoding="utf-8") as f:
        stats = json.load(f)
    tot_pos = stats["gold_all"]["positive"]
    tot_neg = stats["gold_all"]["negative"]
    rows = []
    with open(TABLE, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r["tokens"] = int(r["tokens"]); r["tweets"] = int(r["tweets"])
            r["gold_pos"] = int(r["gold_pos"]); r["gold_neg"] = int(r["gold_neg"])
            rows.append(r)
    return rows, tot_pos, tot_neg


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # emoji in output; Windows consoles default to cp1252
    rows, TP, TN = load()
    N = TP + TN
    out = []
    def p(s=""):
        out.append(s); print(s)

    p("=== Per-emoji predictiveness + polarity asymmetry ===")
    p(f"Corpus 2x2 base: {TP} pos / {TN} neg (base rate {TP/N:.3f}); {len(rows)} emoji types.")
    p(f"MIN_TWEETS for highlighted analysis = {MIN_TWEETS}. PMI+ vs base 0.5; Fisher exact 2x2.")

    # enrich
    for r in rows:
        gp, gn = r["gold_pos"], r["gold_neg"]
        r["p_pos"] = gp / (gp + gn)
        r["pmi_pos"] = pmi_pos(gp, gn)
        r["gold_sign"] = "POS" if r["p_pos"] > 0.5 else ("NEG" if r["p_pos"] < 0.5 else "TIE")
        a, b = gp, gn                    # tag present:   pos, neg
        c_, d_ = TP - gp, TN - gn        # tag absent:    pos, neg
        r["fisher_p"] = fisher_p(a, b, c_, d_)
        r["chi2_p"] = chi2_p(a, b, c_, d_)
        # valence (ESR) vs stance (gold) gap
        r["val_stance_gap"] = (r["esr_sign"] in ("POS", "NEG")
                               and r["esr_sign"] != r["gold_sign"] and r["gold_sign"] != "TIE")

    used = [r for r in rows if r["tweets"] >= MIN_TWEETS]
    used.sort(key=lambda r: -r["tweets"])

    def pmis(x):
        return "+inf" if x == float("inf") else ("-inf" if x == float("-inf") else f"{x:+.2f}")

    p(f"\n--- PER-EMOJI PREDICTIVENESS (tweets >= {MIN_TWEETS}, {len(used)} types) ---")
    p(f"{'emoji':<3}{'tag':<34}{'tw':>3}{'g+':>4}{'g-':>4}{'P(pos)':>8}{'PMI+':>7}"
      f"{'gold':>5}{'mstr':>5}{'esr':>5}{'Fish p':>8}  gap")
    for r in used:
        p(f"{r['emoji']:<3}{r['tag'][:33]:<34}{r['tweets']:>3}{r['gold_pos']:>4}{r['gold_neg']:>4}"
          f"{r['p_pos']:>8.2f}{pmis(r['pmi_pos']):>7}{r['gold_sign']:>5}"
          f"{r['master_sign'][:4]:>5}{r['esr_sign'][:4]:>5}{r['fisher_p']:>8.3f}"
          f"  {'<<' if r['val_stance_gap'] else ''}")

    # --- significant predictors (descriptive; multiple-comparison caveat) ---
    sig = [r for r in used if r["fisher_p"] < 0.05]
    p(f"\n--- Fisher p<0.05 among the {len(used)} tested (DESCRIPTIVE — "
      f"{len(used)} tests, no correction; treat as flags) ---")
    for r in sorted(sig, key=lambda r: r["fisher_p"]):
        p(f"  {r['emoji']} {r['tag'][:30]:<31} gold={r['gold_sign']} "
          f"P(pos)={r['p_pos']:.2f} PMI+={pmis(r['pmi_pos'])} Fisher p={r['fisher_p']:.3f}")

    # --- VALENCE vs STANCE gap (the tongue-investigation payoff, at corpus scale) ---
    gaps = [r for r in used if r["val_stance_gap"]]
    p(f"\n--- VALENCE (ESR) vs STANCE (gold) GAP: ESR sign != gold skew "
      f"({len(gaps)} of {len(used)}) ---")
    p("  (ESR reads emoji VALENCE; gold reflects deployed STANCE. Gap = the mocking/")
    p("   mocking cluster; a lexicon feature imports valence, not stance.)")
    for r in sorted(gaps, key=lambda r: r["pmi_pos"]):
        p(f"  {r['emoji']} {r['tag'][:30]:<31} ESR={r['esr_sign']:<4} master={r['master_sign'][:4]:<4}"
          f" -> gold {r['gold_sign']} (P(pos)={r['p_pos']:.2f}, n={r['tweets']})")

    # --- POLARITY ASYMMETRY  ---
    p("\n--- POLARITY ASYMMETRY: do POSITIVE-lexicon emoji predict pos as strongly as")
    p("    NEGATIVE-lexicon emoji predict neg?  (grouped by ESR sign, tweets>=MIN) ---")
    for lex in ("ESR", "master"):
        key = "esr_sign" if lex == "ESR" else "master_sign"
        def norm(v):
            v = v.upper()
            return "POS" if v.startswith("POS") else ("NEG" if v.startswith("NEG") else v)
        posset = [r for r in used if norm(r[key]) == "POS"]
        negset = [r for r in used if norm(r[key]) == "NEG"]
        def summarize(group, want):
            if not group:
                return "n/a"
            # accuracy of the lexicon sign as a gold predictor (tweet-weighted)
            hit = sum(r["gold_pos"] if want == "POS" else r["gold_neg"] for r in group)
            tot = sum(r["tweets"] for r in group)
            mean_ppos = sum(r["p_pos"] * r["tweets"] for r in group) / tot
            return (f"{len(group)} types, {tot} tw; lexicon-sign matches gold "
                    f"{hit}/{tot}={hit/tot*100:.0f}%; mean P(pos)={mean_ppos:.2f}")
        p(f"  [{lex}] POS-lexicon emoji -> predict POSITIVE: {summarize(posset,'POS')}")
        p(f"  [{lex}] NEG-lexicon emoji -> predict NEGATIVE: {summarize(negset,'NEG')}")

    p("\n--- READ ---")
    p("  * Positive-lexicon emoji predict pos-gold far more reliably than negative-lexicon")
    p("    emoji predict neg-gold IFF the numbers above show it; the asymmetry")
    p("    is stated from the % match, not assumed.")
    p("  * The valence-stance GAP rows are positive-VALENCE emoji deployed with negative")
    p("    STANCE, a candidate mechanism for the positive push (polarity_gain.py).")
    p(f"  CAVEAT: per-emoji n tiny (mostly {MIN_TWEETS}-11 tweets); {len(used)} Fisher tests,")
    p("    no multiple-comparison correction. Descriptive + effect size; do not over-test.")

    # write
    txt = "\n".join(out)
    with open(os.path.join(ANA, "emoji_predictiveness.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    fields = ["emoji", "tag", "tokens", "tweets", "gold_pos", "gold_neg", "p_pos",
              "pmi_pos", "gold_sign", "master_sign", "esr_sign", "fisher_p", "chi2_p",
              "val_stance_gap"]
    with open(os.path.join(ANA, "emoji_predictiveness.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r[k] if not isinstance(r.get(k), float)
                            else round(r[k], 4)) for k in fields})
    print(f"\n-> {os.path.join(ANA, 'emoji_predictiveness.txt')}")
    print(f"-> {os.path.join(ANA, 'emoji_predictiveness.csv')}  (all {len(rows)} types)")


if __name__ == "__main__":
    main()
