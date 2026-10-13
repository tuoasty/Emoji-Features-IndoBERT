"""Does IndoBERT's tokenizer have any representation for a bare emoji glyph?

Supports the claim that the emoji signal lives in the name-TAG text, not in
the Unicode character -- i.e. why condition B1 (raw Unicode, reverse-mapped from
tags) performs no better than B0 (stripped). If every glyph maps to [UNK], then
B1's input is B0's input plus one uninformative token, which is exactly what the
accuracy says (74.90 vs 74.23 on the emoji subset, p=1.00).

Reads the 60 distinct emoji actually used in the corpus from
results/report/eda/eda_emoji_table.csv, so the check covers the real input
distribution rather than an arbitrary emoji list.

Run:  python analysis/tokenizer_check.py
Out:  results/report/eda/tokenizer_check.txt
"""
import csv
import os

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN = os.path.join(HERE, "results", "report", "eda")
MODEL = "indobenchmark/indobert-base-p1"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    unk = tok.unk_token

    with open(os.path.join(AN, "eda_emoji_table.csv"), encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["emoji"].strip()]

    out = []
    n_unk = n_tok_unk = n_tok_all = 0
    for r in rows:
        glyph = r["emoji"].strip()
        pieces = tok.tokenize(glyph)
        is_unk = len(pieces) == 0 or all(p == unk for p in pieces)
        occurrences = int(r["tokens"])
        n_unk += is_unk
        n_tok_all += occurrences
        n_tok_unk += occurrences if is_unk else 0
        out.append((r["tag"], glyph, pieces, occurrences, is_unk))

    lines = [
        "=== IndoBERT tokenizer vs bare emoji glyphs ===",
        f"model      : {MODEL}",
        f"vocab size : {tok.vocab_size}   unk token: {unk}",
        "",
        f"distinct emoji mapping to {unk} only : {n_unk}/{len(rows)}",
        f"occurrences mapping to {unk} only    : {n_tok_unk}/{n_tok_all}",
        "",
        "--- per emoji (sorted by corpus frequency) ---",
    ]
    for tag, glyph, pieces, occ, is_unk in sorted(out, key=lambda x: -x[3]):
        lines.append(f"{tag[:34]:36s} n={occ:>3}  {'UNK' if is_unk else 'OK '}  {pieces}")

    text = "\n".join(lines)
    with open(os.path.join(AN, "tokenizer_check.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
