"""Build tag→emoji and tag→sentiment lookup dicts from master_emoji.csv."""
import os
import pandas as pd

_CSV_PATH = os.path.join(os.path.dirname(__file__), "master_emoji.csv")


def _normalize_tag(tag: str) -> str:
    """Strip surrounding whitespace/angle-brackets, lowercase."""
    return tag.strip().strip("<>").strip().lower()


def load_tag_to_emoji(csv_path: str = _CSV_PATH) -> dict[str, str]:
    """Return {normalized_tag: unicode_emoji}. Keys are lowercase, no brackets."""
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        key = _normalize_tag(str(row["Special Tag"]))
        result[key] = str(row["Emoji"])
    return result


def load_tag_to_sentiment(csv_path: str = _CSV_PATH) -> dict[str, str]:
    """Return {normalized_tag: sentiment_str}. Values: 'positive' or 'negative'."""
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        key = _normalize_tag(str(row["Special Tag"]))
        result[key] = str(row["Sentiment"]).strip().lower()
    return result
