"""EDA figures (the paper's Fig. 1 is eda_fig2_per_emoji_gold.pdf). Reads results/report/eda/eda_stats.json + eda_emoji_table.csv.

Fig 1 -- type-vs-token polarity inversion (100% stacked horizontal bars).
Fig 2 -- per-emoji gold-sentiment split, top emoji (100% stacked horizontal bars).

Diverging sentiment palette (validated CVD-safe, dataviz skill):
  positive #2166AC (blue) | negative #B2182B (red) | neutral/uncovered #999999 (gray).
Static PNG for the paper/presentation: form-first, direct value labels, legend,
recessive axes; HTML-only hover/dark-mode steps N/A.

Run:  python analysis/corpus_eda_figures.py
Out:  results/report/eda/eda_fig1_type_vs_token.png, eda_fig2_per_emoji_gold.png
"""
import os
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN = os.path.join(HERE, "results", "report", "eda")

POS, NEG, NEU = "#2166AC", "#B2182B", "#999999"
INK, MUTED = "#1a1a1a", "#666666"

# ESR-sign overlay marker. Shape/ink only -- deliberately NOT the red/blue bar
# palette, so lexicon identity (an attribute of the emoji) never reuses the gold
# series hues. Filled = ESR-positive, gray = neutral, open ring = uncovered.
ESR_STYLE = {
    "POS":       dict(marker="o", s=34, facecolors=INK,       edgecolors=INK,       linewidths=0.0),
    "NEU":       dict(marker="o", s=34, facecolors="#c7c7c7",  edgecolors="#8a8a8a", linewidths=1.0),
    "uncovered": dict(marker="o", s=34, facecolors="white",    edgecolors=INK,       linewidths=1.1),
    "NEG":       dict(marker="v", s=38, facecolors=INK,        edgecolors=INK,       linewidths=0.0),
}

plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": INK,
})


def _stack_h(ax, y, segments, total, bar_h=0.62, label_fs=9):
    """Draw one 100%-stacked horizontal bar; segments=[(value,color,show_label)]."""
    left = 0.0
    for val, color, show in segments:
        if val <= 0:
            continue
        frac = val / total
        ax.barh(y, frac, left=left, height=bar_h, color=color,
                edgecolor="white", linewidth=1.2, zorder=3)
        if show and frac > 0.06:
            ax.text(left + frac / 2, y, f"{val}", ha="center", va="center",
                    color="white", fontsize=label_fs, fontweight="bold", zorder=4)
        left += frac


def fig1(stats):
    rows = [
        ("Lexicon\n(165 emoji types)", stats["lex_master"]["positive"],
         stats["lex_master"]["negative"], 0),
        ("In use — master\n(331 tokens)", stats["tok_master"]["positive"],
         stats["tok_master"]["negative"], 0),
        ("In use — ESR\n(331 tokens)", stats["tok_esr"].get("POS", 0),
         stats["tok_esr"].get("NEG", 0),
         stats["tok_esr"].get("NEU", 0) + stats["tok_esr"].get("uncovered", 0)),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ys = list(range(len(rows)))[::-1]  # first row on top
    for y, (label, pos, neg, neu) in zip(ys, rows):
        total = pos + neg + neu
        _stack_h(ax, y, [(pos, POS, True), (neg, NEG, True), (neu, NEU, True)], total)
        ax.text(1.015, y, f"{neg/total*100:.0f}% neg", va="center", ha="left",
                fontsize=8.5, color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.axvline(0.5, color="#dddddd", lw=0.8, zorder=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (POS, NEG, NEU)]
    ax.legend(handles, ["positive emoji", "negative emoji", "neutral / uncovered"],
              loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3,
              frameon=False, fontsize=8.5, handlelength=1.1, handleheight=1.1)
    ax.set_title("Emoji polarity flips from lexicon to usage: negative emoji are the\n"
                 "majority of types (53%) but a minority of tokens (33% master, 6% ESR)",
                 fontsize=10.5, fontweight="bold", loc="left", pad=10)
    fig.tight_layout()
    out = os.path.join(AN, "eda_fig1_type_vs_token.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def _short(name):
    n = name.strip().title()
    fixes = {"Face With Tears Of Joy": "Tears of joy",
             "Face With Stuck-Out Tounges": "Tongue out",
             "Face With Stuck-Out Tongue & Closed Eyes": "Tongue + eyes",
             "Face With Stuck-Out Tongue & Closed E": "Tongue + eyes",
             "Smiling Face With Smiling Eyes": "Smiling eyes",
             "Smiling Face With Sunglasses": "Sunglasses",
             "Smiling Face With Open Mouth & Smiling Eyes": "Open-mouth smile",
             "Smiling Face With Open Mouth & Closed Eyes": "Open-mouth (shut)",
             "Grining Face With Smile Eye": "Grinning face",
             "Face Blowing A Kiss": "Blowing a kiss",
             "Zipper-Mouth Face": "Zipper-mouth",
             "Folded Hands": "Folded hands",
             "Victory Hand": "Victory hand",
             "Thumbs Up": "Thumbs up",
             "Ok Hand": "OK hand",
             "See-No-Evil Monkey": "See-no-evil",
             "Rolling On The Floor Laughing": "ROFL"}
    return fixes.get(n, n)


def fig2(min_tweets=5):
    """Per-emoji gold split (diverging bar) with an ESR-sign marker overlay.

    Single-column IEEE figure, page-budget tuned: ~3.2in tall so it costs about a
    third of a column rather than three fifths. Bars encode gold sentiment (blue
    positive / red negative) as a share of the emoji's tweets; the marker left of
    each name encodes the ESR lexicon sign, and a trailing star marks a
    Fisher-significant gold skew (p<0.05 over all 900 tweets). The story:
    ESR-positive emoji (filled markers) that nonetheless sit in the gold-negative
    cluster -- valence is not stance. Carrying the Fisher marks here lets the
    figure stand in for the separate mocking-emoji table (its PMI+ column was a
    monotone transform of the bar share, so it added no information).
    No on-plot title; interpretation lives in the LaTeX caption.
    """
    with open(os.path.join(AN, "eda_emoji_table.csv"), encoding="utf-8-sig") as f:
        table = list(csv.DictReader(f))
    # Fisher exact p per tag, from emoji_predictiveness.py (same eda_emoji_table source,
    # so the join is exact); used only to star the significant rows.
    with open(os.path.join(AN, "emoji_predictiveness.csv"), encoding="utf-8-sig") as f:
        fisher = {r["tag"]: float(r["fisher_p"]) for r in csv.DictReader(f)}
    items = []
    for r in table:
        gp, gn = int(r["gold_pos"]), int(r["gold_neg"])
        tw = gp + gn
        if tw < min_tweets:
            continue
        star = "*" if fisher.get(r["tag"], 1.0) < 0.05 else ""
        items.append((_short(r["tag"]) + star, gp, gn, tw, r["esr_sign"]))
    # most-positive share on top; tie-break by n (larger first)
    items.sort(key=lambda x: (x[1] / x[3], x[3]))

    fig, ax = plt.subplots(figsize=(3.5, 0.150 * len(items) + 0.42))
    for y, (name, gp, gn, tw, es) in enumerate(items):
        _stack_h(ax, y, [(gp, POS, True), (gn, NEG, True)], tw, bar_h=0.70, label_fs=5.6)
        ax.text(1.072, y, f"{tw}", va="center", ha="right", fontsize=5.5, color=MUTED)

    # ESR-sign marker in its own column just left of the bars (blended transform:
    # x in axes fraction, y in data) -- shape/ink, never the bar hues.
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for y, (name, gp, gn, tw, es) in enumerate(items):
        st = ESR_STYLE.get(es, ESR_STYLE["uncovered"])
        ax.scatter([-0.052], [y], transform=trans, clip_on=False, zorder=6, **st)

    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[0] for it in items], fontsize=6.2)
    ax.tick_params(axis="y", pad=15)  # room for the marker column + the sig star
    ax.set_ylim(-0.62, len(items) - 0.38)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.tick_params(axis="x", labelsize=6.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0.5, color="#dddddd", lw=0.8, zorder=1)
    # No x-axis label: the 0/50%/100% ticks plus the LaTeX caption carry it, and
    # the label cost ~0.2in of page height at a 6-page limit.

    # ONE legend row only -- the second row (the ESR-marker key) cost ~0.5in of
    # page height, so the marker encoding moves to the LaTeX caption instead.
    gold_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (POS, NEG)]
    ax.legend(gold_handles, ["gold positive", "gold negative"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=6.3, handlelength=1.0, handleheight=1.0,
              columnspacing=1.4, handletextpad=0.4, borderaxespad=0.0)

    fig.tight_layout(pad=0.25)
    for ext in ("png", "pdf"):
        out = os.path.join(AN, f"eda_fig2_per_emoji_gold.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote eda_fig2_per_emoji_gold.{png,pdf}", f"({len(items)} emoji, min_tweets={min_tweets})")


def main():
    with open(os.path.join(AN, "eda_stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    fig1(stats)
    fig2()


if __name__ == "__main__":
    main()
