# recsys-mini

> A **minimal, runnable learning project** for industrial recommender systems — starting from a deep dive into MovieLens-1M, then building up the full recall + ranking + re-ranking pipeline step by step.
>
> **The goal is not to chase metrics**, but to make the "why we do it this way" of each stage crystal clear. The code exists to serve learning.

---

## Current project status

The `main` branch only implements the "**data download + EDA exploration**" stage:

| | Status |
|---|---|
| One-shot MovieLens-1M download | ✅ |
| 8-section EDA output (rating / time / long-tail / sparsity / user profile / item metadata / ...) | ✅ |
| One in-depth data analysis doc ([`docs/01-data-source-analysis.md`](docs/01-data-source-analysis.md), 433 lines) | ✅ |
| Sample preprocessing / splitting / recall / ranking / re-ranking / evaluation | ⏳ See roadmap |

---

## Quick start

```bash
# 1. Install deps (only 2: pandas + pyyaml)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download MovieLens-1M (~6 MB, one-time)
bash scripts/01_download_data.sh

# 3. Run EDA
python scripts/01_movielens1m_eda.py
```

The EDA output is **identical** to every number cited in [`docs/01-data-source-analysis.md`](docs/01-data-source-analysis.md) — running it once verifies every claim in the doc.

---

## Directory layout

```
recsys-mini/
├── conf/
│   └── config.yaml                # Path config (raw_dir / processed_dir)
├── data/                          # Data landing zone (gitignored)
│   └── raw/ml-1m/                 # .dat files after download
├── docs/
│   └── 01-data-source-analysis.md # Deep dive into the data source (main doc)
├── scripts/
│   ├── 01_download_data.sh        # Download ML-1M
│   └── 01_movielens1m_eda.py      # 8-section EDA
├── src/
│   ├── config.py                  # Config loading + path management
│   ├── data/loader.py             # ML-1M three-table reader
│   └── utils/io.py                # logger
├── requirements.txt               # pandas + pyyaml
└── README.md
```

---

## Documentation

| # | Doc | Content | Lines |
|---|---|---|---|
| 01 | [data-source-analysis](docs/01-data-source-analysis.md) | MovieLens-1M deep dive: health-check card, rating / time / item / user distributions, sparsity, vs. real UGC platforms | 433 |

> Each completed phase ships with a corresponding `docs/` learning note. Every number is required to be **reproducible** (running the script reproduces the exact value).

---

## Dataset: MovieLens-1M

A movie rating dataset released by GroupLens in 1999, the de facto academic baseline for recommender systems.

| Dimension | Value |
|---|---|
| Users | 6,040 |
| Items (movies) | 3,706 |
| Interactions | 1,000,209 |
| Time span | 2000-04 ~ 2003-02 (1,038 days) |
| Rating scale | 1 ~ 5 integers |
| Matrix density | 4.47% |

> This dataset is **clean, modestly sized, and signal-poor**. Best for getting started, but also "**the dataset most likely to give you false expectations about real recommendation scenarios**" — see §2 "Dataset health-check card" and its sub-section "Key reading: the 4 distribution-extreme numbers" in the main doc for the full breakdown.

---


## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Data processing | pandas 2.x |
| Config | YAML (pyyaml) |
| Pulled in later as needed | pyarrow (parquet) / scipy (sparse matrix) / scikit-learn / lightgbm / faiss / pytorch |

> Current `requirements.txt` has only 2 lines (`pandas`, `pyyaml`), strictly reflecting what the `main` branch actually needs. New packages are added only when entering the corresponding phase.

---

