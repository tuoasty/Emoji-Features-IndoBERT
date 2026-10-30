"""Rebuild the RQ4 error-analysis set and the annotation sheets (revision step C1).

RQ4 set = emoji tweets, seed-majority correctness (same rule as slice_report.py):
  persistent: wrong under B0 AND T1+L (master, cont.)
  broken:     right under B0, wrong under T1+L
Diffs the set against the v1 hand-labeled tweets (archive) by (fold, val_index)
and text, and carries over the v1 labels for tweets that remain.

Writes (UTF-8 with BOM, ';'-separated, like the v1 file):
  results/report/<tuning>/rq4_set.csv          internal: bucket, preds, v1 labels
  results/report/<tuning>/rq4_sheet_key.csv    sheet id -> (fold, val_index, bucket)
  results/report/<tuning>/rq4_sheet_annotator.csv  blind sheet (no bucket/labels; T1+L
                                                   prediction only, always != gold)
  results/report/<tuning>/rq4_sheet_kevin.csv      same order, v1 labels pre-filled

Usage: python analysis/rq4_set.py --tuning lora
"""
import argparse
import collections
import csv
import os
import random
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.reverse_map import load_tag_to_emoji  # noqa: E402
from src.preprocess import _clean_whitespace, _normalize_tag_key, _strip_tags  # noqa: E402
from src.results_io import RUN_TAGS, load_runs, report_dir  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dataset_tweet_sentiment_pilkada_DKI_2017.csv")
V1_LABELS = os.path.join(ROOT, "..", "..", "archive", "results_v1_testfold_selection",
                         "error_taxonomy_v1_handlabels.csv")
LABELS = ["text_dominant", "sarcasm", "emoji_contradicts", "mixed_polarity", "other"]
FLAGS = ["multi_target", "gold_doubtful"]
V1_TO_NEW = {"text-dominant": "text_dominant", "sarcasm": "sarcasm",
             "emoji-contradicts-text": "emoji_contradicts",
             "mixed-polarity": "mixed_polarity", "multi-target": "multi_target"}
GOLD = {0: "negative", 1: "positive"}
SHUFFLE_SEED = 20261029


def majority(runs, condition):
    """(fold, val_index) -> (majority pred, gold, text), emoji tweets only."""
    votes, gold, text = collections.defaultdict(list), {}, {}
    for (cond, fold, _seed), recs in runs.items():
        if cond != condition:
            continue
        for i, r in recs.items():
            if r["has_emoji"]:
                votes[(fold, i)].append(r["pred"])
                gold[(fold, i)] = r["true"]
                text[(fold, i)] = r["text"]
    return {k: (int(sum(v) > len(v) / 2), gold[k], text[k]) for k, v in votes.items()}


def render(text, tag_to_emoji):
    """'<FACE WITH TEARS OF JOY>' -> '<FACE WITH TEARS OF JOY>😂' (tag kept)."""
    return re.sub(r"<[^>]+>",
                  lambda m: m.group(0) + tag_to_emoji.get(_normalize_tag_key(m.group(0)), ""),
                  text)


def write_csv(path, rows, cols):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", default="lora", choices=RUN_TAGS)
    ap.add_argument("--f1", default="f1_master_ours")
    args = ap.parse_args()

    runs = load_runs(args.tuning)
    b0, t1, f1 = majority(runs, "b0"), majority(runs, "t1"), majority(runs, args.f1)
    paslon = pd.read_csv(DATA)["Pasangan Calon"]
    tag_to_emoji = load_tag_to_emoji()

    v1 = pd.read_csv(V1_LABELS, sep=";", encoding="utf-8-sig")
    v1 = {(int(r.fold), int(r.val_index)): r for r in v1.itertuples()}

    rows = []
    for k in sorted(set(b0) & set(f1)):
        pb, gold, _ = b0[k]
        pf, _, text = f1[k]
        b_ok, f_ok = pb == gold, pf == gold
        if f_ok:
            continue
        bucket = "persistent" if not b_ok else "broken"
        row = {"bucket": bucket, "fold": k[0], "val_index": k[1],
               "pasangan_calon": paslon.iloc[k[1]], "gold": GOLD[gold],
               "pred_b0": GOLD[pb], "pred_t1": GOLD[t1[k][0]], "pred_f1": GOLD[pf],
               "text": text}
        old = v1.get(k)
        if old is not None and _clean_whitespace(old.text) == _clean_whitespace(text):
            row["v1_status"] = "carried"
            row["v1_bucket"] = "broken" if old.bucket == "hurt_by_f1" else old.bucket
            row["v1_category"] = old.category
            for lab in str(old.category).split():
                row[V1_TO_NEW[lab]] = 1
        else:
            row["v1_status"] = "new"
        rows.append(row)

    dropped = [(k, r) for k, r in v1.items()
               if not any((x["fold"], x["val_index"]) == k for x in rows)]

    out = report_dir(args.tuning)
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "rq4_set.csv"), rows,
              ["bucket", "fold", "val_index", "pasangan_calon", "gold", "pred_b0", "pred_t1",
               "pred_f1", "v1_status", "v1_bucket", "v1_category", "text"] + LABELS + FLAGS[:1])

    order = list(range(len(rows)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    sheet, kevin, key = [], [], []
    for sid, idx in enumerate(order, 1):
        r = rows[idx]
        base = {"id": sid, "tweet": render(r["text"], tag_to_emoji),
                "text_without_emoji": _clean_whitespace(_strip_tags(r["text"])),
                "gold": r["gold"],
                # T1+L only: always the opposite of gold here, so it reveals no bucket
                "model_prediction": r["pred_f1"]}
        sheet.append(base)
        kevin.append({**base, **{c: r.get(c, "") for c in LABELS + FLAGS},
                      "v1_status": r["v1_status"]})
        key.append({"id": sid, "fold": r["fold"], "val_index": r["val_index"],
                    "bucket": r["bucket"]})
    cols = ["id", "tweet", "text_without_emoji", "gold", "model_prediction"] + LABELS + FLAGS + ["note"]
    write_csv(os.path.join(out, "rq4_sheet_annotator.csv"), sheet, cols)
    write_csv(os.path.join(out, "rq4_sheet_kevin.csv"), kevin, cols + ["v1_status"])
    write_csv(os.path.join(out, "rq4_sheet_key.csv"), key, ["id", "fold", "val_index", "bucket"])

    n = collections.Counter((r["bucket"], r["gold"]) for r in rows)
    print(f"RQ4 set [{args.tuning}, T1+L = {args.f1}]: {len(rows)} tweets")
    for b in ("persistent", "broken"):
        print(f"  {b:10s} {n[(b, 'positive')] + n[(b, 'negative')]:3d}"
              f"  (pos {n[(b, 'positive')]}, neg {n[(b, 'negative')]})")
    st = collections.Counter(r["v1_status"] for r in rows)
    print(f"  vs v1 ({len(v1)} tweets): carried {st['carried']}, new {st['new']},"
          f" dropped {len(dropped)}")
    moved = [r for r in rows if r["v1_status"] == "carried" and r["v1_bucket"] != r["bucket"]]
    print(f"  carried tweets whose bucket changed: {len(moved)}")
    for r in moved:
        print(f"    ({r['fold']},{r['val_index']}) {r['v1_bucket']} -> {r['bucket']}")
    print("  dropped (in v1, not in the new set):")
    for k, r in dropped:
        print(f"    {k} v1 {r.bucket:10s} {r.category:28s} {r.text[:70]}")
    print("  new (to label):")
    for r in rows:
        if r["v1_status"] == "new":
            print(f"    ({r['fold']},{r['val_index']}) {r['bucket']:10s} {r['gold']:8s} {r['text'][:70]}")
    print(f"-> {out}: rq4_set.csv, rq4_sheet_annotator.csv, rq4_sheet_kevin.csv, rq4_sheet_key.csv")


if __name__ == "__main__":
    main()
