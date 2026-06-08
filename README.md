# recsys-mini

> A **minimal, runnable learning project** for industrial recommender systems — from a deep dive into MovieLens-1M to a full **recall → ranking → evaluation** pipeline, built up one stage at a time.
>
> **The goal is not to chase metrics**, but to make the "why we do it this way" of each stage crystal clear. The code exists to serve learning.

---

## Current project status

The full offline pipeline is implemented and runnable end to end:

| Stage | Script | Status |
|---|---|---|
| Data download (MovieLens-1M, one-shot) | `scripts/download_data.sh` | ✅ |
| EDA (8-section data exploration) | `scripts/01_eda.py` | ✅ |
| Sample prep (k-core → implicit → split → parquet) | `scripts/02_prepare_samples.py` | ✅ |
| Recall (ItemCF + Popular + RRF fusion) | `scripts/03_train_recall.py` | ✅ |
| Ranking (LightGBM CTR model) | `scripts/04_train_rank.py` | ✅ |
| End-to-end offline eval (recall → rank → Top-N) | `scripts/05_offline_eval.py` | ✅ |
| Filtering / re-ranking | — | ⏳ designed in docs, not yet coded |

---

## Pipeline at a glance

```
download_data.sh        raw ml-1m/*.dat
        │
  01_eda.py             8-section exploratory report (read-only)
        │
  02_prepare_samples    k-core filter → explicit→implicit → LOO/time split → parquet
        │
  03_train_recall       ItemCF + Popular, fused with RRF  →  recall_*.pkl
        │
  04_train_rank         negative sampling → features → LightGBM  →  ranker_lgb.pkl + feature_store.pkl
        │
  05_offline_eval       recall → ranking → Top-N, three comparable metric blocks
```

Each stage reads `conf/config.yaml` for all tunable parameters and writes its
artifacts to `artifacts/` (recall/rank steps) or `data/processed/` (prep step).

---

## Quick start

```bash
# 1. Install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download MovieLens-1M (~6 MB, one-time)
bash scripts/download_data.sh

# 3a. Run the whole pipeline at once
bash scripts/run_all.sh

# 3b. ...or run stage by stage
python scripts/01_eda.py              # explore the raw data
python scripts/02_prepare_samples.py  # build train/valid/test parquet
python scripts/03_train_recall.py     # train + evaluate recall routes
python scripts/04_train_rank.py       # train the LightGBM ranker
python scripts/05_offline_eval.py     # full-chain offline evaluation
```

The EDA output is **identical** to every number cited in
[`docs/en/01-data-source-analysis.md`](docs/en/01-data-source-analysis.md) —
running it once verifies every claim in the doc.

---

## Directory layout

```
recsys-mini/
├── conf/
│   └── config.yaml                # All tunable params (paths, k-core, recall, ranker, eval)
├── data/                          # Data landing zone (gitignored)
│   ├── raw/ml-1m/                 # .dat files after download
│   └── processed/                 # train/valid/test + movies/users parquet
├── artifacts/                     # Trained models (gitignored): recall_*.pkl, ranker_lgb.pkl, ...
├── docs/
│   ├── en/                        # English deep-dive docs (01–07)
│   └── zh/                        # Chinese companion notes (informal, *_CN.md gitignored)
├── scripts/
│   ├── download_data.sh           # Download ML-1M
│   ├── run_all.sh                 # Run the whole pipeline
│   ├── 01_eda.py                  # 8-section EDA
│   ├── 02_prepare_samples.py      # k-core → implicit → split → parquet
│   ├── 03_train_recall.py         # ItemCF + Popular + RRF fusion
│   ├── 04_train_rank.py           # LightGBM ranker
│   └── 05_offline_eval.py         # end-to-end offline eval
├── src/
│   ├── config.py                  # Config loading + path management
│   ├── data/                      # loader / sampler (negatives) / splitter (LOO, time)
│   ├── recall/                    # base / itemcf / popular / multi_recall (RRF)
│   ├── rank/                      # lgb_ranker (LightGBM CTR model)
│   ├── features/                  # builder (profiles + train-only stats, leak-free)
│   ├── eval/                      # recall_metrics (HitRate/Recall/NDCG/Coverage) + rank_metrics (AUC/GAUC)
│   └── utils/                     # io (logger, timer, pickle, log_metrics)
├── requirements.txt
└── README.md
```

---

## Results (MovieLens-1M, leave-one-out split)

**Recall routes** (evaluated on valid as ground truth):

| Route | Recall@10 | Recall@50 | Recall@200 | Coverage@200 |
|---|---|---|---|---|
| ItemCF | 5.29% | 20.08% | 50.15% | 69.89% |
| Popular | 4.74% | 15.48% | 36.97% | 22.29% |
| **RRF fusion** | **6.08%** | 19.50% | **50.20%** | 69.36% |

**Ranker** (LightGBM, on valid): `AUC ≈ 0.728`, `GAUC ≈ 0.712`.

`05_offline_eval.py` then prints three comparable blocks — recall-only,
recall+ranking, and recall straight-through (no ranking) — so you can attribute
the final Top-N quality to the right stage. See
[`docs/zh/09-offline-eval_CN.md`](docs/zh/09-offline-eval_CN.md) for how to read them.

> Numbers may vary slightly with config; they are reproducible by re-running the
> scripts. The point is the **method**, not squeezing the last decimal.

---

## Documentation

English deep-dive docs live in [`docs/en/`](docs/en); each completed phase ships
with a learning note where **every number is reproducible**.

| # | Doc | Content |
|---|---|---|
| 01 | [data-source-analysis](docs/en/01-data-source-analysis.md) | MovieLens-1M deep dive: health-check card, distributions, sparsity, vs. real UGC platforms |
| 02 | [cold-start](docs/en/02-cold-start.md) | The cold-start problem and production strategies |
| 03 | [item-understanding](docs/en/03-item-understanding.md) | Turning items into tags / embeddings / features |
| 04 | [recall](docs/en/04-recall.md) | Candidate generation: the funnel, eight recall families, SSB |
| 05 | [collaborative-filtering](docs/en/05-collaborative-filtering.md) | ItemCF / UserCF and the broader CF family |
| 06 | [popular-recall](docs/en/06-popular-recall.md) | Popularity recall as a fallback, and how to weight it |
| 07 | [multi-recall](docs/en/07-multi-recall.md) | Multi-route fusion with RRF |

> Chinese companion notes (informal study material, analogies, walkthroughs) are
> in [`docs/zh/`](docs/zh) (the `*_CN.md` files are gitignored).

---

## Dataset: MovieLens-1M

A movie rating dataset released by GroupLens, the de facto academic baseline for recommender systems.

| Dimension | Value |
|---|---|
| Users | 6,040 |
| Items (movies) | 3,706 |
| Interactions | 1,000,209 |
| Time span | 2000-04 ~ 2003-02 (1,038 days) |
| Rating scale | 1 ~ 5 integers |
| Matrix density | 4.47% |

> This dataset is **clean, modestly sized, and signal-poor**. Great for getting
> started, but also "the dataset most likely to give you false expectations about
> real recommendation scenarios" — see §2 of the main doc for the full breakdown.

---

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Data processing | pandas 2.x / numpy / pyarrow (parquet) |
| Recall | ItemCF (co-occurrence + IUF) · popularity · RRF fusion |
| Ranking | LightGBM (binary CTR) |
| Metrics | scikit-learn (AUC) · custom HitRate / Recall / NDCG / Coverage / GAUC |
| Config | YAML (pyyaml) |
| Pulled in later as needed | faiss / pytorch (two-tower, embeddings) |

> `requirements.txt` reflects exactly what the current pipeline needs. New
> packages are added only when entering the corresponding phase.
