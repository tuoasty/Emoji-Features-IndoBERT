"""Corpus EDA (emoji use vs. label, paper Sec. III-C) -- descriptive, no training needed.

Mirrors src/dataset.py's corpus exactly (df[["text","label"]].dropna(); labels
sorted -> negative=0, positive=1) and the <TAG> emoji definition used by
slice_eval.py (raw text contains r"<[^>]+>" => emoji-bearing).

Produces (all TYPE-level = per distinct emoji, TOKEN-level = per occurrence):
  1. emoji frequency table  -> results/report/eda/eda_emoji_table.csv
  2. each emoji's pos/neg GOLD split (tweet-level)  [in the same table]
  3. usage polarity: TOKEN counts by ESR sign AND master sign
       (the real "are negative emoji rarer in USAGE" question; the 87neg/78pos
        lexicon figure is TYPE-level over the whole lexicon, not usage)
  4. emoji-per-tweet distribution
  5. tweet length (words, tags stripped): emoji-bearing vs non-emoji tweets
Prints a summary and writes eda_summary.txt + eda_emoji_table.csv.

Run:  python analysis/corpus_eda.py
"""
import os
import re
import sys
import csv
import json
from collections import Counter, defaultdict

import pandas as pd
import emoji as emoji_lib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "data"))
import reverse_map  # noqa: E402
import esr_map      # noqa: E402

DATASET_CSV = os.path.join(HERE, "data", "dataset_tweet_sentiment_pilkada_DKI_2017.csv")
OUT_DIR = os.path.join(HERE, "results", "report", "eda")
TAG_RE = re.compile(r"<[^>]+>")
ESR_MARGIN = 0.15  # same threshold as build_taxonomy_sheet.py


def norm_key(tag):
    return tag.strip("<>").strip().lower()


def esr_sign(triple):
    if triple is None:
        return "uncovered"
    neg, _neut, pos = triple
    diff = pos - neg
    if diff > ESR_MARGIN:
        return "POS"
    if -diff > ESR_MARGIN:
        return "NEG"
    return "NEU"


def load_corpus():
    """Mirror src/dataset.py._load_raw: rename text/label, dropna, map labels."""
    df = pd.read_csv(DATASET_CSV)
    text_col = next(c for c in df.columns if "text" in c.lower())
    label_col = next(c for c in df.columns
                     if any(k in c.lower() for k in ["sentimen", "label", "sentiment"]))
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]].dropna().reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["gold"] = df["label"].str.strip().str.lower()  # 'negative'/'positive'
    return df


def word_count_stripped(text):
    """Word count after stripping <TAG>s and raw unicode emoji (b0-style), so
    emoji markup doesn't inflate length."""
    t = TAG_RE.sub("", text)
    t = emoji_lib.replace_emoji(t, replace="")
    t = re.sub(r"\s+", " ", t).strip()
    return len(t.split()) if t else 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # emoji in output; Windows consoles default to cp1252
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_corpus()
    tag2emoji = reverse_map.load_tag_to_emoji()
    tag2sent = reverse_map.load_tag_to_sentiment()

    n_total = len(df)
    df["tags"] = df["text"].apply(lambda t: [norm_key(x) for x in TAG_RE.findall(t)])
    df["n_tags"] = df["tags"].apply(len)
    df["has_emoji"] = df["n_tags"] > 0
    df["words"] = df["text"].apply(word_count_stripped)

    emoji_df = df[df["has_emoji"]]
    non_df = df[~df["has_emoji"]]

    # ---- gold balance
    gold_all = df["gold"].value_counts().to_dict()
    gold_emoji = emoji_df["gold"].value_counts().to_dict()
    gold_non = non_df["gold"].value_counts().to_dict()

    # ---- per-emoji TYPE table (token count, tweet count, gold split, signs)
    tok_tweets = defaultdict(set)     # tag -> set of tweet indices
    tok_count = Counter()             # tag -> total occurrences
    tag_gold = defaultdict(Counter)   # tag -> {gold: n_tweets}
    for i, row in df.iterrows():
        seen = set()
        for tag in row["tags"]:
            tok_count[tag] += 1
            tok_tweets[tag].add(i)
            if tag not in seen:
                tag_gold[tag][row["gold"]] += 1
                seen.add(tag)

    rows = []
    for tag in sorted(tok_count, key=lambda t: -tok_count[t]):
        ch = tag2emoji.get(tag, "")
        triple = esr_map.get_esr_triple(ch) if ch else None
        rows.append({
            "tag": tag,
            "emoji": ch,
            "tokens": tok_count[tag],
            "tweets": len(tok_tweets[tag]),
            "gold_pos": tag_gold[tag].get("positive", 0),
            "gold_neg": tag_gold[tag].get("negative", 0),
            "master_sign": tag2sent.get(tag, "?"),
            "esr_sign": esr_sign(triple),
            "esr_pos": round(triple[2], 3) if triple else "",
            "esr_neg": round(triple[0], 3) if triple else "",
        })

    with open(os.path.join(OUT_DIR, "eda_emoji_table.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- usage polarity: TOKEN-level counts by master & ESR sign
    tok_master = Counter()
    tok_esr = Counter()
    for r in rows:
        tok_master[r["master_sign"]] += r["tokens"]
        tok_esr[r["esr_sign"]] += r["tokens"]
    # TYPE-level over emoji ACTUALLY USED in corpus
    type_master = Counter(r["master_sign"] for r in rows)
    type_esr = Counter(r["esr_sign"] for r in rows)
    # TYPE-level over the WHOLE lexicon (the 87/78 figure)
    lex_master = Counter(tag2sent.values())

    # ---- emoji-per-tweet distribution (among emoji-bearing)
    per_tweet = Counter(emoji_df["n_tags"])

    # ---- coverage
    n_types = len(rows)
    n_uncovered_esr = sum(1 for r in rows if r["esr_sign"] == "uncovered")
    tok_uncovered_esr = sum(r["tokens"] for r in rows if r["esr_sign"] == "uncovered")

    # ---------- summary ----------
    L = []
    def p(s=""):
        L.append(s); print(s)

    p("=== CORPUS EDA: emoji use vs. label ===")
    p(f"Total tweets (dropna text/label): {n_total}")
    p(f"  gold balance: {gold_all}")
    p(f"Emoji-bearing tweets: {len(emoji_df)}  ({len(emoji_df)/n_total*100:.1f}%)")
    p(f"  gold balance (emoji slice): {gold_emoji}")
    p(f"Non-emoji tweets: {len(non_df)}  ({len(non_df)/n_total*100:.1f}%)")
    p(f"  gold balance (non-emoji): {gold_non}")
    p()
    p(f"Emoji TOKENS (total occurrences): {sum(tok_count.values())}")
    p(f"Emoji TYPES (distinct tags used): {n_types}")
    p(f"  ESR-uncovered types: {n_uncovered_esr}/{n_types}  (tokens: {tok_uncovered_esr})")
    p()
    p("--- USAGE POLARITY (the type-vs-token point) ---")
    p(f"TOKEN-level by MASTER sign: {dict(tok_master)}")
    p(f"TOKEN-level by ESR sign:    {dict(tok_esr)}")
    p(f"TYPE-level (corpus-used) by MASTER: {dict(type_master)}")
    p(f"TYPE-level (corpus-used) by ESR:    {dict(type_esr)}")
    p(f"TYPE-level (WHOLE lexicon) by MASTER: {dict(lex_master)}  <- the 87neg/78pos figure")
    p()
    p("--- EMOJI PER TWEET (emoji-bearing only) ---")
    for k in sorted(per_tweet):
        p(f"  {k} emoji: {per_tweet[k]} tweets")
    p(f"  mean emoji/tweet: {emoji_df['n_tags'].mean():.2f}  median: {emoji_df['n_tags'].median():.0f}  max: {emoji_df['n_tags'].max()}")
    p()
    p("--- TWEET LENGTH (words, tags+unicode stripped) ---")
    p(f"  emoji-bearing:  mean {emoji_df['words'].mean():.1f}  median {emoji_df['words'].median():.0f}")
    p(f"  non-emoji:      mean {non_df['words'].mean():.1f}  median {non_df['words'].median():.0f}")
    p()
    p("--- TOP 15 EMOJI ---")
    p(f"{'tag':<38}{'emoji':<4}{'tok':>4}{'tw':>4}{'g+':>4}{'g-':>4}  {'mstr':>4} {'esr':>9}")
    for r in rows[:15]:
        p(f"{r['tag'][:37]:<38}{r['emoji']:<4}{r['tokens']:>4}{r['tweets']:>4}"
          f"{r['gold_pos']:>4}{r['gold_neg']:>4}  {r['master_sign'][:4]:>4} {r['esr_sign']:>9}")

    with open(os.path.join(OUT_DIR, "eda_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    # machine-readable dump for the figure script
    with open(os.path.join(OUT_DIR, "eda_stats.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_total": n_total, "gold_all": gold_all,
            "n_emoji_tweets": len(emoji_df), "gold_emoji": gold_emoji,
            "n_non_tweets": len(non_df), "gold_non": gold_non,
            "tokens_total": sum(tok_count.values()), "n_types": n_types,
            "tok_master": dict(tok_master), "tok_esr": dict(tok_esr),
            "type_master": dict(type_master), "type_esr": dict(type_esr),
            "lex_master": dict(lex_master),
            "per_tweet": {int(k): int(v) for k, v in per_tweet.items()},
            "len_emoji_words": emoji_df["words"].tolist(),
            "len_non_words": non_df["words"].tolist(),
            "esr_uncovered_types": n_uncovered_esr, "esr_uncovered_tokens": tok_uncovered_esr,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nwrote: eda_emoji_table.csv, eda_summary.txt, eda_stats.json -> {OUT_DIR}")


if __name__ == "__main__":
    main()
