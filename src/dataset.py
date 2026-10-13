"""Dataset loading and stratified k-fold splitting."""
import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from src.preprocess import preprocess, extract_emoji_feature, POLARITY_FILTER

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PILKADA_PATH = os.path.join(DATA_DIR, "dataset_tweet_sentiment_pilkada_DKI_2017.csv")

DATASET_PATHS = {
    "pilkada": PILKADA_PATH,
}

N_SPLITS = 5

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


def get_fold(dataset_name: str, condition: str, fold: int):
    """Return (train_dataset, val_dataset, label2id) for a given fold."""
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

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    splits = list(skf.split(df, df["label"]))
    train_idx, val_idx = splits[fold]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    return train_ds, val_ds, label2id
