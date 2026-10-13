"""Where predictions and reports live, and how to load the predictions back.

results/preds/<tuning>/<run>.json   per-tweet predictions written by slice_eval.py
results/report/<tuning>/            outputs of slice_report.py and analysis/*.py
"""
import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def preds_dir(tuning: str) -> str:
    return os.path.normpath(os.path.join(RESULTS_DIR, "preds", tuning))


def report_dir(tuning: str) -> str:
    return os.path.normpath(os.path.join(RESULTS_DIR, "report", tuning))


def load_runs(tuning: str) -> dict:
    """runs[(condition, fold, seed)] = {val_index: record}."""
    runs = {}
    for path in glob.glob(os.path.join(preds_dir(tuning), "*.json")):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        runs[(d["condition"], d["fold"], d["seed"])] = {r["val_index"]: r for r in d["records"]}
    return runs
