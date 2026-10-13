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

# tables -> results/report/
python compare_tuning.py                         # frozen vs LoRA vs full fine-tuning
python slice_report.py --tuning lora             # accuracy by slice, McNemar, failure cases
python analysis/fusion_mcnemar.py --tuning lora  # T1 vs the T1+L conditions, presence vs continuous
python analysis/polarity_gain.py --tuning lora   # gain by gold polarity
python analysis/polarity_mix.py --tuning lora    # harm by emoji polarity mix
python analysis/polarity_filter.py --tuning lora # T1 vs T1-pos / T1-neg against B0
```

Run all commands from the repository root. Training writes per-tweet predictions to
`results/preds/`, which is not shipped; every report below is shipped.

## Where each paper result comes from

| paper | file in `results/report/` |
|---|---|
| §III Data and EDA (counts, usage by type and use) | `eda/eda_summary.txt`, `eda/eda_emoji_table.csv` |
| §III-D per-emoji predictiveness, Fig. 1 | `eda/emoji_predictiveness.txt`, `eda/eda_fig2_per_emoji_gold.pdf` |
| §IV-D LoRA vs frozen vs full fine-tuning | `compare_tuning.txt` (detail in `frozen/`, `full/`) |
| Table II, RQ1 (dilution, B0/B1/T1 McNemar) | `lora/slice_report.txt` |
| RQ1 raw emoji map to `[UNK]` | `eda/tokenizer_check.txt` |
| RQ2 (lexicon feature vs tag-text) | `lora/mcnemar_fusion.txt` |
| RQ3 (gain by gold polarity, T1-pos) | `lora/polarity_gain.txt`, `lora/polarity_mix.txt` |
| RQ4 (error taxonomy) | `lora/failure_cases.json`, `lora/error_taxonomy.csv` |

`lora/error_taxonomy.csv` holds the hand labels (one annotator) for the 36 emoji-bearing
tweets that stay wrong under both B0 and T1+L (master, cont.) (`persistent`, 30) or
that T1+L breaks (`hurt_by_f1`, 6). Its tweet set is the one in
`lora/failure_cases.json`. It is the only result that cannot be regenerated from code.

## License

Code: MIT (see `LICENSE`). The dataset and the master emoji lexicon are not redistributed here; `data/download.py` fetches them from the rizalespe repository.
