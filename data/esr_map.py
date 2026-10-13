"""Novak Emoji Sentiment Ranking (ESR) lookup: {emoji_char: (neg, neut, pos)}.

Backed by the emosent-py package, which bundles Emoji_Sentiment_Data_v1.0.csv
(751 emojis) from the CLARIN.SI repository (http://hdl.handle.net/11356/1048),
the data table for Kralj Novak et al. 2015, "Sentiment of Emojis", PLOS ONE
(doi:10.1371/journal.pone.0144296).

emosent stores RAW annotation counts (e.g. 😂 -> negative=3614, neutral=4163,
positive=6845). We normalize by their sum (== occurrences) to recover the
published Neg/Neut/Pos [0..1] fractions (😂 -> 0.247 / 0.285 / 0.468).

Triple ordering is (neg, neut, pos) throughout — matches the ESR column order.
"""
from emosent.emosent import EMOJI_SENTIMENT_DICT


def _normalize(rank: dict):
    neg, neut, pos = rank["negative"], rank["neutral"], rank["positive"]
    total = neg + neut + pos
    if total == 0:
        return None
    return (neg / total, neut / total, pos / total)


def get_esr_triple(emoji_char: str):
    """Return normalized (neg, neut, pos) for one emoji, or None if not in ESR."""
    if not emoji_char:
        return None
    rank = EMOJI_SENTIMENT_DICT.get(emoji_char.strip())
    if rank is None:
        return None
    return _normalize(rank)


def load_emoji_to_esr() -> dict:
    """Return {emoji_char: (neg, neut, pos)} for all 751 ESR emojis (normalized)."""
    result = {}
    for char, rank in EMOJI_SENTIMENT_DICT.items():
        triple = _normalize(rank)
        if triple is not None:
            result[char] = triple
    return result
