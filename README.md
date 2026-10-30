# Emoji Features on IndoBERT-Based Indonesian Sentiment Analysis

Code for the paper *The Effect of Emoji Features on IndoBERT-Based Indonesian Sentiment Analysis*.

## Setup

```bash
pip install -r requirements.txt
python data/download.py      # fetches the Pilkada DKI 2017 corpus and master_emoji.csv
```

The corpus (900 tweets) and the emoji lexicon come from
[rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia](https://github.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia).
The Emoji Sentiment Ranking comes from the `emosent-py` package.

Tested with Python 3.10, torch 2.11 (CUDA 12.8), transformers 5.5.4, peft 0.19.1.

## Conditions

| code name | paper name | what the model sees |
|---|---|---|
| `b0` | B0 | emoji stripped |
| `b1` | B1 | emoji as raw Unicode |
| `t1` | T1 | emoji as text tags, e.g. `<Smiling Face>` |
| `f1_master_ours` | T1+L (master, cont.) | tag-text + continuous master-lexicon feature, fused at the classifier |
| `f1_master_ang` | T1+L (master, pres.) | tag-text + master-lexicon presence one-hot |
| `f1_esr_ours` | T1+L (ESR, cont.) | tag-text + mean ESR triple |
| `f1_esr_ang` | T1+L (ESR, pres.) | tag-text + ESR-argmax presence one-hot |
| `t1_pos` / `t1_neg` | T1-pos / — | tag-text, keeping only positive / only negative emoji (master lexicon); diagnostics |

`f1` = fusion; `ours` = the continuous encoding, `ang` = the presence encoding
reproduced from Anggraeni et al. The code keeps these names; the paper uses the
names in the second column.

Every condition runs under 5-fold stratified cross-validation × 3 seeds (0, 1, 42),
with one of three tuning modes: a frozen encoder with only the head trained
(`--tuning frozen`, lr 1e-3), LoRA (`--tuning lora`, lr 2e-4), or full fine-tuning
(`--tuning full`, lr 2e-5). All results in the paper use LoRA except the tuning
comparison.

**Checkpoint selection.** The test fold is never used to pick the epoch. 10% of each
training fold (72 tweets, stratified by label and emoji presence, seed 0; `DEV_FRAC`
and `DEV_SEED` in `src/dataset.py`) is held out as a development split, the same for
every condition and seed. Each run keeps the epoch with the best development macro-F1
(ties: lower development loss, then the earlier epoch), and its test fold is scored
once, at that epoch.

**Classifier heads.** The text-only and fusion heads use the same dropout on the
pooled vector and the same initialization, so T1 and T1+L differ only in the feature.

**Backbones.** `--backbone indobert` (default, `indobenchmark/indobert-base-p1`) or
`--backbone indobertweet` (`indolem/indobertweet-base-uncased`). For IndoBERTweet each
tag is rewritten in the colon form it saw in pretraining (`<OK HAND>` → `:ok_hand:`),
and its untrained pooler is trained with the LoRA adapters. Its results go to
`results/*/indobertweet_lora/`.

## Reproduce

```bash
# data and exploratory analysis (no training) -> results/report/eda/
python analysis/corpus_eda.py            # emoji counts by type and use, gold split
python analysis/emoji_predictiveness.py  # per-emoji gold share + Fisher tests
python analysis/corpus_eda_figures.py    # per-emoji gold figure
python analysis/tokenizer_check.py       # IndoBERT maps raw emoji to [UNK]

# train + per-tweet predictions -> results/preds/<tuning>/
python slice_eval.py --tuning frozen                          # 7 paper conditions, 105 runs
python slice_eval.py --tuning lora                            # same, LoRA
python slice_eval.py --tuning full                            # same, full fine-tuning
python slice_eval.py --tuning lora --conditions t1_pos t1_neg
python slice_eval.py --backbone indobertweet --tuning lora --conditions b0 t1 f1_master_ours

# tables -> results/report/
python compare_tuning.py                         # frozen vs LoRA vs full fine-tuning
python slice_report.py --tuning lora             # accuracy by slice, McNemar, failure cases
python slice_report.py --tuning indobertweet_lora
python analysis/fusion_mcnemar.py --tuning lora  # T1 vs the T1+L conditions, presence vs continuous
python analysis/polarity_gain.py --tuning lora   # gain by gold polarity
python analysis/polarity_filter.py --tuning lora # T1 vs T1-pos / T1-neg against B0
python analysis/revision_stats.py --tuning lora  # RQ3 Fisher tests, direction of changed predictions
python analysis/revision2_stats.py --tuning lora # metrics, bootstrap CIs, Holm, power
python analysis/revision2_stats.py --tuning indobertweet_lora
python analysis/revision2_stats.py --tuning full
python analysis/rq4_set.py --tuning lora         # RQ4 tweet set and blank annotation sheets
```

Run all commands from the repository root. Training writes per-tweet predictions to
`results/preds/`, which is not shipped; every report below is shipped.

## Statistics

Contrasts are tested per tweet with the exact McNemar test on the 165 emoji-bearing
tweets, using each tweet's majority prediction over the three seeds. Two contrasts are
primary, B0 vs T1 (RQ1) and T1 vs T1+L (master, cont.) (RQ2), Holm-corrected within
each backbone; every other test is exploratory and uncorrected. Primary contrasts get a
paired bootstrap 95% CI (10,000 resamples of tweets), and the McNemar power at n = 165
is computed at the observed discordant rate. All of this is in `revision2_stats.txt`.

## Where each paper result comes from

| paper | file in `results/report/` |
|---|---|
| §III Data and EDA (counts, usage by type and use) | `eda/eda_summary.txt`, `eda/eda_emoji_table.csv` |
| §III-D per-emoji predictiveness, Fig. 1 | `eda/emoji_predictiveness.txt`, `eda/eda_fig2_per_emoji_gold.pdf` |
| §IV-D LoRA vs frozen vs full fine-tuning | `compare_tuning.txt` |
| §V-A full fine-tuning: RQ1 gain does not replicate | `full/revision2_stats.txt` |
| Table II upper block, RQ1 (dilution, McNemar) | `lora/slice_report.txt` |
| RQ1/RQ2 per-tweet differences, 95% CIs, Holm p, power | `lora/revision2_stats.txt` |
| Table II lower block, IndoBERTweet | `indobertweet_lora/slice_report.txt`, `indobertweet_lora/revision2_stats.txt` |
| RQ1 raw emoji map to `[UNK]` | `eda/tokenizer_check.txt` |
| Table III, RQ2 (lexicon feature vs tag-text) | `lora/mcnemar_fusion.txt` |
| Table IV, metrics and confusion counts on emoji-bearing tweets | `lora/revision2_stats.txt` |
| Table V, RQ3 (gain by gold polarity) | `lora/polarity_gain.txt`, `lora/revision_stats.txt` |
| RQ3 T1-pos diagnostic | `lora/mcnemar_polarity_filter.txt` |
| Table VI, RQ4 (error taxonomy) | `lora/rq4_final.csv`, `lora/rq4_annotator_{1,2,3}.csv`, `lora/failure_cases.json` |

`lora/rq4_final.csv` holds the hand labels for the 33 emoji-bearing tweets that stay
wrong under both B0 and T1+L (master, cont.) (`persistent`, 29) or that T1+L breaks
(`broken`, 4). Each tweet was labeled independently by three annotators from a written
guide, seeing the tweet, its gold label and the wrong prediction but not the condition;
each label is the majority of the three. Each annotator's own labels are in
`lora/rq4_annotator_1.csv`, `_2.csv` and `_3.csv` (same tweets and columns), from which
the agreement (mean pairwise Cohen's κ) and the majority vote can be recomputed. A tweet can carry several labels
(`text_dominant`, `sarcasm`, `emoji_contradicts`, `mixed_polarity`, `other`), and
`multi_target` flags tweets that address more than one candidate. The labels are the
only result that cannot be regenerated from code.

## License

Code: MIT (see `LICENSE`). The dataset and the master emoji lexicon are not redistributed here; `data/download.py` fetches them from the rizalespe repository.
