"""Text preprocessing for each emoji-handling condition."""
import re
import emoji as emoji_lib


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _strip_unicode_emojis(text: str) -> str:
    return emoji_lib.replace_emoji(text, replace="")


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_tag_key(tag: str) -> str:
    """Strip angle-brackets and lowercase for dict lookup."""
    return tag.strip("<>").strip().lower()


# Polarity-filtered tag-text: which master-lexicon polarity each condition keeps.
POLARITY_FILTER = {"t1_pos": "positive", "t1_neg": "negative"}


def preprocess(text: str, condition: str, tag_to_emoji: dict = None,
               tag_to_sentiment: dict = None) -> str:
    """
    condition:
      b0 — strip all bracketed tags and raw Unicode emojis
      t1 — pass <TAG> tokens as-is; strip raw Unicode emojis only
      t1_pos / t1_neg — as t1, but keep only the <TAG>s whose master-lexicon
           polarity is positive / negative; the other tags are stripped
      b1 — strip pre-existing raw Unicode first, then replace <TAG> with Unicode emoji
           (reverse-map). Strip order matters: stripping after the reverse-map would
           delete the just-inserted emoji and silently reduce b1 to b0.
    """
    if condition == "b0":
        text = _strip_tags(text)
        text = _strip_unicode_emojis(text)
    elif condition == "t1":
        text = _strip_unicode_emojis(text)
    elif condition in POLARITY_FILTER:
        if tag_to_sentiment is None:
            raise ValueError(f"{condition} requires tag_to_sentiment dict")
        keep = POLARITY_FILTER[condition]
        text = _strip_unicode_emojis(text)
        def _filter_tag(m):
            key = _normalize_tag_key(m.group(0))
            return m.group(0) if tag_to_sentiment.get(key) == keep else ""
        text = re.sub(r"<[^>]+>", _filter_tag, text)
    elif condition == "b1":
        if tag_to_emoji is None:
            raise ValueError("b1 requires tag_to_emoji dict")
        text = _strip_unicode_emojis(text)
        def _replace_tag(m):
            key = _normalize_tag_key(m.group(0))
            return tag_to_emoji.get(key, "")
        text = re.sub(r"<[^>]+>", _replace_tag, text)
    else:
        raise ValueError(f"Unknown condition: {condition!r}. "
                         f"Supported: b0, t1, t1_pos, t1_neg, b1, f1")

    return _clean_whitespace(text)


_TAG_RE = re.compile(r"<[^>]+>")

# Feature dimensionality per fusion condition. Drives F1FusionModel's classifier
# input width and the dataset's "torch" format columns.
FEATURE_DIM = {
    "f1_master_ours": 2,   # [pos, neg] fraction
    "f1_master_ang": 3,    # [pos, neg, none] presence one-hot
    "f1_esr_ours": 3,      # [neg, neut, pos] mean ESR triple
    "f1_esr_ang": 3,       # [pos, neg, neutral/none] presence one-hot from ESR argmax
}


def _tags(text: str) -> list[str]:
    return _TAG_RE.findall(text)


def _feat_master_fraction(text: str, tag_to_sentiment: dict) -> list[float]:
    """[pos_frac, neg_frac] over <TAG> emojis (master_emoji is binary). [0,0] if none.

    Drops the structurally-always-zero neutral slot of the original 3-dim feature
    (master_emoji.csv labels only positive/negative).
    """
    pos = neg = 0
    for tag in _tags(text):
        s = tag_to_sentiment.get(_normalize_tag_key(tag))
        if s == "positive":
            pos += 1
        elif s == "negative":
            neg += 1
    total = pos + neg
    if total == 0:
        return [0.0, 0.0]
    return [pos / total, neg / total]


def _feat_master_anggraeni(text: str, tag_to_sentiment: dict) -> list[float]:
    """Anggraeni et al. Eq. 1 presence one-hot, positive-priority: [pos, neg, none].

    Faithful reimplementation: any positive emoji -> [1,0,0]; else any negative ->
    [0,1,0]; else (master_emoji has no neutral, so this means no emoji) -> [0,0,1].
    """
    has_pos = has_neg = False
    for tag in _tags(text):
        s = tag_to_sentiment.get(_normalize_tag_key(tag))
        if s == "positive":
            has_pos = True
        elif s == "negative":
            has_neg = True
    if has_pos:
        return [1.0, 0.0, 0.0]
    if has_neg:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _feat_esr_mean(text: str, tag_to_emoji: dict, emoji_to_esr: dict) -> list[float]:
    """Mean ESR triple [neg, neut, pos] over <TAG> emojis present in ESR. [0,0,0] if none.

    Pipeline: <TAG> -> Unicode char (master_emoji) -> ESR triple. Emojis absent from
    ESR (7 of the 60 dataset types, 8% of occurrences) contribute nothing.
    """
    triples = []
    for tag in _tags(text):
        ch = tag_to_emoji.get(_normalize_tag_key(tag))
        if not ch:
            continue
        triple = emoji_to_esr.get(ch)
        if triple is None:
            continue
        triples.append(triple)
    if not triples:
        return [0.0, 0.0, 0.0]
    n = len(triples)
    return [sum(t[i] for t in triples) / n for i in range(3)]


def _feat_esr_anggraeni(text: str, tag_to_emoji: dict, emoji_to_esr: dict) -> list[float]:
    """Anggraeni presence one-hot [pos, neg, neutral/none], polarity from ESR argmax.

    Each emoji's class = argmax over its ESR (neg, neut, pos) triple. Positive-priority:
    any positive emoji -> [1,0,0]; else any negative -> [0,1,0]; else (only neutral
    emojis, or none / not in ESR) -> [0,0,1].
    """
    has_pos = has_neg = False
    for tag in _tags(text):
        ch = tag_to_emoji.get(_normalize_tag_key(tag))
        if not ch:
            continue
        triple = emoji_to_esr.get(ch)  # (neg, neut, pos)
        if triple is None:
            continue
        cls = max(range(3), key=lambda i: triple[i])  # 0=neg, 1=neut, 2=pos
        if cls == 2:
            has_pos = True
        elif cls == 0:
            has_neg = True
    if has_pos:
        return [1.0, 0.0, 0.0]
    if has_neg:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def extract_emoji_feature(
    text: str,
    condition: str,
    tag_to_sentiment: dict = None,
    tag_to_emoji: dict = None,
    emoji_to_esr: dict = None,
) -> list[float]:
    """Dispatch to the emoji feature for a given fusion condition (see FEATURE_DIM)."""
    if condition == "f1_master_ours":
        return _feat_master_fraction(text, tag_to_sentiment)
    if condition == "f1_master_ang":
        return _feat_master_anggraeni(text, tag_to_sentiment)
    if condition == "f1_esr_ours":
        return _feat_esr_mean(text, tag_to_emoji, emoji_to_esr)
    if condition == "f1_esr_ang":
        return _feat_esr_anggraeni(text, tag_to_emoji, emoji_to_esr)
    raise ValueError(f"Unknown fusion condition: {condition!r}. Supported: {list(FEATURE_DIM)}")
