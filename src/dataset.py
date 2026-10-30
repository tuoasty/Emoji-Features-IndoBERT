"""Dataset loading and stratified k-fold splitting."""
import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.preprocess import (preprocess, extract_emoji_feature, tweet_tag_format,
                            POLARITY_FILTER, _TAG_RE)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PILKADA_PATH = os.path.join(DATA_DIR, "dataset_tweet_sentiment_pilkada_DKI_2017.csv")

DATASET_PATHS = {
    "pilkada": PILKADA_PATH,
}

N_SPLITS = 5
# Inner validation (dev) split, carved out of each training fold and used only to pick
# the checkpoint epoch; the held-out test fold is scored once, at that epoch. Stratified
# on label x has-emoji (raw text) and seeded independently of fold and training seed,
# so every condition, backbone and seed trains and selects on the same tweets.
DEV_FRAC = 0.1
DEV_SEED = 0

# Fusion conditions: 2x2 over {master, esr} lexicon source x {ours, anggraeni} encoding.
F1_CONDITIONS = ("f1_master_ours", "f1_master_ang", "f1_esr_ours", "f1_esr_ang")
_MASTER_CONDS = ("f1_master_ours", "f1_master_ang")
_ESR_CONDS = ("f1_esr_ours", "f1_esr_ang")


def _load_raw(dataset_name: str) -> pd.DataFrame:
    path = DATASET_PATHS[dataset_name]
    df = pd.read_csv(path)

    text_col = next(c for c in df.columns if "text" in c.lower())
    label_col = next(
        c for c in df.columns
        if any(k in c.lower() for k in ["sentimen", "label", "sentiment"])
    )
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]].dropna()

    unique_labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    df["label"] = df["label"].map(label2id)
    return df, label2id


def get_fold(dataset_name: str, condition: str, fold: int, backbone: str = "indobert"):
    """Return (train_dataset, dev_dataset, test_dataset, label2id) for a given fold.
    test = the held-out CV fold; dev = DEV_FRAC of the remaining training tweets."""
    tag_to_emoji = None
    tag_to_sentiment = None
    emoji_to_esr = None

    # b1, the polarity-filtered t1 variants and fusion need tag->sentiment/emoji;
    # esr fusion also needs the ESR map.
    if condition == "b1" or condition in POLARITY_FILTER or condition in F1_CONDITIONS:
        from data.reverse_map import load_tag_to_emoji, load_tag_to_sentiment
        tag_to_emoji = load_tag_to_emoji()
        tag_to_sentiment = load_tag_to_sentiment()
    if condition in _ESR_CONDS:
        from data.esr_map import load_emoji_to_esr
        emoji_to_esr = load_emoji_to_esr()

    df, label2id = _load_raw(dataset_name)
    has_emoji = df["text"].astype(str).str.contains(_TAG_RE).astype(int)

    if condition in F1_CONDITIONS:
        # Extract feature from raw text BEFORE tags are stripped.
        df["emoji_feature"] = df["text"].astype(str).apply(
            lambda t: extract_emoji_feature(
                t, condition,
                tag_to_sentiment=tag_to_sentiment,
                tag_to_emoji=tag_to_emoji,
                emoji_to_esr=emoji_to_esr,
            )
        )
        # All fusion cells share the t1 text channel: keep <TAG>s, strip raw unicode.
        df["text"] = df["text"].astype(str).apply(lambda t: preprocess(t, "t1"))
    elif condition == "b1":
        df["text"] = df["text"].astype(str).apply(
            lambda t: preprocess(t, "b1", tag_to_emoji=tag_to_emoji)
        )
    elif condition in POLARITY_FILTER:
        df["text"] = df["text"].astype(str).apply(
            lambda t: preprocess(t, condition, tag_to_sentiment=tag_to_sentiment)
        )
    else:
        df["text"] = df["text"].astype(str).apply(lambda t: preprocess(t, condition))
    if backbone == "indobertweet":
        df["text"] = df["text"].apply(tweet_tag_format)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    splits = list(skf.split(df, df["label"]))
    train_idx, test_idx = splits[fold]
    strata = (df["label"].values * 2 + has_emoji.values)[train_idx]
    train_idx, dev_idx = train_test_split(
        train_idx, test_size=DEV_FRAC, stratify=strata, random_state=DEV_SEED)

    def _ds(idx):
        return Dataset.from_pandas(df.iloc[idx].reset_index(drop=True))
    return _ds(train_idx), _ds(dev_idx), _ds(test_idx), label2id
