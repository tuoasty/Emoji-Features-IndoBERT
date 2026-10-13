"""Download rizalespe dataset files and cache locally."""
import os
import requests

BASE = "https://raw.githubusercontent.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/master"

FILES = {
    "dataset_tweet_sentiment_pilkada_DKI_2017.csv": f"{BASE}/dataset_tweet_sentiment_pilkada_DKI_2017.csv",
    "master_emoji.csv": f"{BASE}/master_emoji.csv",
}

DATA_DIR = os.path.dirname(__file__)


def download(filename, url):
    dest = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest):
        print(f"  {filename}: already cached")
        return
    print(f"  {filename}: downloading...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
    print(f"  {filename}: saved ({len(r.content)} bytes)")


if __name__ == "__main__":
    for name, url in FILES.items():
        download(name, url)
    print("Done.")
