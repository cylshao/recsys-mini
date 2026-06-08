# MovieLens-1M Data Source Deep Dive

> The current  branch only implements the "**data download + EDA**" stage. Every number in this
> doc comes from the actual output of `scripts/01_eda.py` — running it once reproduces them all.
>
> Recall / ranking / re-ranking / evaluation modules **are not yet implemented**. So this doc focuses
> purely on "**the characteristics of the data source itself, and what they imply for a future
> recommender system**", and does not discuss model performance that does not yet exist.

## Table of contents

- [1. Project status](#1-project-status)
- [2. Dataset health-check card](#2-dataset-health-check-card)
- [3. Signal matrix (5/13)](#3-signal-matrix-513)
- [4. Rating distribution — true negatives are gold](#4-rating-distribution--true-negatives-are-gold)
- [5. Time distribution — fatally imbalanced](#5-time-distribution--fatally-imbalanced)
- [6. User behavior length — all heavy users](#6-user-behavior-length--all-heavy-users)
- [7. Item long tail — textbook power law](#7-item-long-tail--textbook-power-law)
- [8. Matrix sparsity — 4,500× denser than UGC](#8-matrix-sparsity--4500-denser-than-ugc)
- [9. User profile — male-skewed, young, white-collar](#9-user-profile--male-skewed-young-white-collar)
- [10. Item metadata — 90s / Drama dominated](#10-item-metadata--90s--drama-dominated)
- [11. Versus real UGC video platforms](#11-versus-real-ugc-video-platforms)
- [12. Reproducing every number in this doc](#12-reproducing-every-number-in-this-doc)
- [13. Preprocessing decisions baked into `02_prepare_samples.py`](#13-preprocessing-decisions-baked-into-02_prepare_samplespy)

---

## 1. Project status

The `main` branch's code currently covers only the **first two steps**:

```
[step 1] Download   scripts/01_download_data.sh           # ML-1M zip → data/raw/ml-1m
[step 2] EDA        scripts/01_eda.py                     # prints 8 sections of stats
```

The supporting library code:

```
src/config.py                # config loading + path management
src/data/loader.py           # load_movielens_1m (only function, returns raw ratings/movies/users)
src/utils/io.py              # logger
conf/config.yaml             # contains paths.raw_dir / paths.processed_dir
```

**This doc does not discuss the metrics of any implemented model**, it only answers one question:

> **If we were to build a full recommender system on this data in the future, what does the data itself tell us?**

---

## 2. Dataset health-check card

| Dimension | Value |
|---|---|
| Total interactions | **1,000,209** |
| Number of users | **6,040** |
| Number of items | **3,706** (metadata has 3,883; 177 movies were never rated) |
| Time span | 2000-04-25 ~ 2003-02-28 (**1,038 days**) |
| Rating scale | 1 ~ 5 (integer) |
| Min interactions per user | 20 (guaranteed at dataset packaging time) |
| Max interactions per user | 2,314 |
| Min interactions per item | 1 |
| Max interactions per item | 3,428 (= a blockbuster watched by 56.8% of users) |
| Matrix density | **4.47%** |

**One-line verdict**: the most classic baseline dataset in academic recsys — **clean structure, modest size, limited signal**. Best for getting started, and also "the dataset most likely to give you false expectations about real recommendation scenarios".

### Key reading: the 4 distribution-extreme numbers

The four numbers — `min/max interactions per user` and `min/max interactions per item` — look mundane, but each one corresponds to a specific recsys problem. A 2×2 view makes it clearer:

```
              Min end                Max end
     Users   20 (no cold start)      2,314 (heavy individual)
     Items    1 (long-tail dead zone)  3,428 (popularity bias)
```

**① Min interactions per user = 20 (guaranteed by the dataset)**
GroupLens applied k-core filtering before release, removing every user with fewer than 20 ratings. **Direct consequence**: the dataset contains **zero cold-start users, zero anonymous visitors, zero light users** — yet on real UGC platforms these segments make up 60%+ of traffic.
**Implication**: any metric you produce on ML-1M **completely fails to answer "how does the system perform on new users?"**. This is one of the largest blind spots of ML-1M, and it cannot be patched by algorithms — only by switching dataset.

**② Max interactions per user = 2,314**
The most active user alone rated 2,314 movies ≈ 62% of the entire catalog. Compared to the p50 user (96 ratings), **a single person ≈ 24 typical users' worth of samples**.
**Implication**: at training time, without sample weighting or grouping, the gradient gets dominated by a handful of super-active users — the model effectively learns "what these few people like". At evaluation time, without per-user grouping, **global Recall@K is hijacked by heavy users**. This is exactly why production systems prefer **GAUC** (per-user grouped AUC) over plain AUC.

**③ Min interactions per item = 1**
Some movies are rated by only 1 user; 5.5% of items have ≤ 2 interactions (see §7).
**Implication**: this is fatal for ItemCF — its similarity formula needs **co-occurrence**, and a single rating means a single co-occurrence event with massive statistical noise. **These items can never enter any item's top-K similarity list, and therefore never get recalled**. This is the root of the "cold items / long-tail unreachable" problem; you must rely on **content-based recall** (using genre) or **two-tower generalization** to reach them.

**④ Max interactions per item = 3,428 (= 56.8% of all users)**
The most popular movie was watched by more than half of all users. In ML-1M this is typically _American Beauty (1999)_ or _Star Wars_ — late-90s blockbusters.
**Implication**: this is the most direct source of **popularity bias**. Any model that maximizes likelihood will boost the predicted probability for this high-frequency item; at recall time it appears in the top-K of huge numbers of users; **Recall@K may look beautiful but you've essentially built "blindly recommend the blockbuster", with zero personalization value**. Production fights this with importance sampling, the `log(item_freq)` correction term in two-tower models, or re-ranking diversification.
**Compared to real platforms**: the top blockbusters on TikTok / Kuaishou penetrate < 1% of users, while ML-1M's top blockbuster penetrates 56.8%. So the strategy of "just push the blockbusters" can rack up great metrics on ML-1M but **smashes head-first into the wall on real platforms**.

### Putting the 4 numbers together

| Number | Problem it exposes | Where it's expanded |
|---|---|---|
| Min user = 20 | No cold-start users | §6 User behavior length |
| Max user = 2,314 | Heavy users dominate training/eval | §9 User profile + future design notes |
| Min item = 1 | Long tail unreachable | §7 Item long tail |
| Max item = 3,428 | Popularity bias | §8 Matrix sparsity |

**The two biggest "frauds"**: the user lower bound of 20 lulls you into thinking every user has enough data to learn from (sidestepping cold start); the item upper bound of 56.8% penetration lets your model get pretty numbers by "pushing blockbusters". Together they make any good metric on ML-1M **strongly misleading** — you think the model is great, but really the dataset is just too friendly.

---

## 3. Signal matrix (5/13)

Item-by-item comparison against the typical signal checklist of a UGC video platform:

| # | Signal | MovieLens-1M | Notes |
|---|---|---|---|
| 1 | user_id + item_id + timestamp | ✅ | the basics |
| 2 | explicit positive feedback (high rating) | ✅ | rating ≥ 4 |
| 3 | explicit negative feedback (low rating) | ✅ | rating ≤ 2 (should be utilized later) |
| 4 | user static profile | ✅ | gender / age / occupation / zip |
| 5 | item category | ✅ | 18 genres |
| 6 | impression log | ❌ | only ratings, no "saw but didn't rate" negatives |
| 7 | dwell / play-completion time | ❌ | a rating ≠ watch duration |
| 8 | likes / comments / shares | ❌ | multi-objective ranking impossible |
| 9 | follow relationships (social graph) | ❌ | no social-based recall |
| 10 | context (device / time-of-day / network) | ❌ | no contextual recommendation |
| 11 | item raw text / cover / source file | ❌ | only titles, no video / poster |
| 12 | item author / creator | ❌ | director / cast not provided in ML-1M |
| 13 | real-time feedback stream | ❌ | static dump, no Kafka |

**Score: 5/13**. The gaps cluster in three areas: **behavior diversity, content understanding, context**.

---

## 4. Rating distribution — true negatives are gold

```
rating=1     56,174  ( 5.62%)  ██
rating=2    107,557  (10.75%)  █████
rating=3    261,197  (26.11%)  █████████████
rating=4    348,971  (34.89%)  █████████████████
rating=5    226,310  (22.63%)  ███████████

positive (>=4): 575,281 (57.52%)
negative (<=2): 163,731 (16.37%)  ← 163K real negative-feedback samples
```

### Three insights

**Insight 1: the dataset is "optimistic"**
57.52% positive vs. only 16.37% negative. This is **classic self-selection bias**: users who choose to come and rate a movie are already biased toward rating things they liked. **Real UGC platforms typically have CTR of only 5–15%** — positive samples are extremely scarce.

**Insight 2: the 26% middle band cannot just be discarded**
Rating-3 makes up a quarter of the data and is neither clearly positive nor negative. The easiest move is to take only `rating >= 4` as positives and treat 3-star as "no interaction" — but that throws away information: the user **bothered to rate it**, meaning the item at least entered their attention. Production approaches:

- treat as "weak positive" (sample weight 0.3)
- or use as "exposure-but-no-click" negatives (when no impression log is available)

**Insight 3: the true negatives (16.37%) are gold; the naive "explicit → implicit" conversion throws them away**

The most naive implicit-feedback construction looks like this (pseudocode):

```python
def to_implicit_naive(df, threshold=4):
    return df[df["rating"] >= threshold].assign(label=1)   # 1-2 stars all dropped
```

If you build samples this way in the future, the ranking model will only ever see **"true positives vs. popularity-sampled fake negatives"** — and 163K **real negatives** (users who took the time to rate something 1–2 stars) get thrown away. These are sharper hard negatives than any sampling can produce.

> **Future design point**: when the sample-preprocessing step is introduced, design the interface as `to_implicit(df, pos_threshold=4, neg_threshold=2)` — output both `label=1` true positives and `label=0` real negatives, and prefer the latter over popularity-sampled negatives at ranker training time.

---

## 5. Time distribution — fatally imbalanced

```
Hottest month: 2000-11-01 → 290,793 interactions
Coldest month: 2002-10-01 →   1,014 interactions     287× difference

Last 7 days:    381 interactions  (0.0381%)
Last 30 days: 1,551 interactions  (0.1551%)
```

### Monthly trend (simplified)

```
2000-04   ▌  11k      ← dataset launch
2000-05   ▏▏▏  67k
...
2000-11  ████████████████████████  291k  ← peak, ~29% of total
...
2002-10                           1k    ← coldest
```

### What this means

- **Any "last X days as test" split produces a toy evaluation set**: the last 7 days hold only 381 interactions, ~0.06 per user.
- MovieLens is **not "time-series data"**, it is more like "**a one-time questionnaire + a long tail of trickle-in ratings**".
- It cannot be used to study "real-time recommendation", "freshness", or "cold-start new items".

### Implications for future split strategy

| Strategy | On ML-1M |
|---|---|
| Global time split (last N days as test) | **Unusable**: test set too small, not statistically significant |
| Per-user Leave-One-Out (last 1 positive per user) | **Preferred**: every active user enters test, ~6,000 users |
| Per-user time percentage (e.g. 80% / 10% / 10%) | Workable, but slightly more complex to implement |

> **Future task**: add `splitter.py` under `src/data` and implement `leave_one_out_split`. The current `main` branch **has no splitting code at all**.

> Caveat: LOO sacrifices "global temporal anti-leakage" (user A's train tail timestamp may be later than user B's valid timestamp). On a time-flat dataset like ML-1M this is acceptable, but for a time-sensitive dataset like KuaiRand you must switch back to a strict global time split.

---

## 6. User behavior length — all heavy users

```
Number of users: 6,040
Quantiles:
  p10:  27 ratings    ← even the lightest 10% of users have ≥ 27
  p25:  44
  p50:  96
  p75: 208
  p90: 400
  p95: 556
  p99: 906

Min: 20    Max: 2,314    Mean: 165.6
Users with ≥ 5 ratings: 6,040 (100.0%)
```

### This is a "heavy-user club"

MovieLens is shipped with a guarantee that every user has at least 20 ratings; the median is 96. **This means**:

| Real scenario | Missing in MovieLens |
|---|---|
| "Cold-start users" (< 5 actions) | zero |
| "Light users" (< 20 actions) | zero |
| "Dormant / churned users" | zero |

**A typical production user-behavior distribution looks like this**:

```
Real UGC platform:
  Anonymous visitors    ████████████  35%
  < 10 actions          ██████████    25%
  10-100 actions        ███████       18%
  100-1000 actions      ████          12%
  > 1000 actions        ██            10%
```

MovieLens completely skips the bottom 60%. **So no metric you produce on ML-1M can tell you "how the system performs on new users"**. This limit is unfixable in algorithmic terms — only switching datasets (KuaiRand contains a real new-user distribution) can address it.

---

## 7. Item long tail — textbook power law

```
Number of items: 3,706
Quantiles:
  p10:    7 ratings    ← 90% of items have ≥ 7 ratings
  p25:   33
  p50:  123
  p75:  350
  p90:  729
  p95: 1,051
  p99: 1,784

Min: 1    Max: 3,428    Mean: 269.9

→ Top 20% hot items (741 items) account for 65.2% of exposure
→ Only 452 items (12.2%) are needed to cover 50% of exposure
→ Cold items with ≤ 2 interactions: 203 items (5.5%)
```

### Long-tail curve (sketch)

```
Exposure
  3000 ┤▌
  2000 ┤▌▌▌
  1000 ┤▌▌▌▌▌▌▌▌▌▌
   500 ┤▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌
   100 ┤▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌
     1 ┤────────────────────────────────────── 2,000+ long-tail items
       └─────────────────────────────────────────→  item rank
        0  500  1000  1500  2000  2500  3000  3500
```

### Key observations

1. **12.2% of items account for 50% of exposure** — a textbook 80/20 (Pareto) law
2. **5.5% cold items** (≤ 2 interactions) basically cannot enter any ItemCF nearest-neighbor list, so **pure collaborative filtering will never recall them**
3. But compared to real UGC, **the tail is still not long enough**: TikTok adds millions of new videos per day and 90% die within 24 hours; the long tail is well over 80%

### Implications for future recall design

| Recall path | Can it reach cold items? |
|---|---|
| ItemCF | **No** (no co-occurrence, no similarity) |
| Popular | **No** (by definition only returns hot items) |
| Two-tower (DSSM) | **Partially** (depends on training-sample coverage) |
| Content-based recall (using genre) | **Yes** (genres alone enable similarity) |
| Look-alike | **Yes** (find similar items via similar "established" items) |

> **Future design principle**: at least one path in multi-channel recall must **cover cold items** (content-based recall on genre, or item-embedding similarity recall).

---

## 8. Matrix sparsity — 4,500× denser than UGC

```
Matrix shape:  6,040 × 3,706 = 22,384,240 cells
Filled cells:  1,000,209
Density:       4.4684%
```

### Compared to real platforms

| Platform | Users | Items | Typical density |
|---|---|---|---|
| **MovieLens-1M** | 6K | 4K | **4.47%** |
| Academic Tenrec | 5M | 2M | ~0.001% |
| TikTok / Kuaishou scale | hundreds of millions | tens of billions per year | < 0.0001% |

**MovieLens is 4,500 ~ 45,000× denser than real UGC**. This creates two illusions:

1. **Collaborative filtering looks artificially strong**: ItemCF approaches its ceiling on dense matrices, but degrades sharply on sparse ones
2. **The advantage of vector-based recall is muted**: two-tower may not noticeably beat ItemCF on ML-1M, but the gap widens by orders of magnitude on real data

> **Conclusion**: any "beautiful" metrics produced on ML-1M may **drop by an order of magnitude on real data**. Treat ML-1M as a sanity check, not as a final benchmark.

---

## 9. User profile — male-skewed, young, white-collar

```
Gender:  M 71.7%  /  F 28.3%       ← heavily imbalanced
Age:     25-34: 34.7%  (largest group)
         18-24: 18.3%
         35-44: 19.8%
         → 18-44 combined: 72.8%
Top occupations:
   4  (college student)  12.6%
   0  (other)            11.8%
   7  (executive)        11.2%
   1  (academic)          8.7%
  17  (technician)        8.5%
```

### This is the "American university campus + tech company, year 2000" demographic

Direct consequences:

- **gender / occupation have low discriminative power as features**: 71.7% male + students/technicians/executives — almost a homogeneous group
- The "personalization" you produce is largely "fine-grained slicing within an echo chamber", not "across-segment differentiation"
- Therefore in any future LightGBM model, `gender / age / occupation` will most likely have low feature importance

### Things to remember when using these features

- **`occupation` is a discrete ID 0~20 with 21 categories and no ordinal meaning**: when training LightGBM, always declare `categorical_feature=["occupation"]` — otherwise it gets treated as a continuous value (`tradesman=18` is incorrectly treated as larger than every occupation except `writer=20`), which is semantically wrong
- **`age` is a discrete bucket** (`{1, 18, 25, 35, 45, 50, 56}`), same advice — declare it categorical
- **`zip` is currently unused**, but in principle could enable geographic features (the first 3 digits of US ZIP codes carry state / region signal)

---

## 10. Item metadata — 90s / Drama dominated

```
Year distribution:
  1990s: 2,283 (58.8%)  ███████████████████████   ← dominant
  1980s:   598 (15.4%)  ██████
  1970s:   247 ( 6.4%)  ██
  2000s:   156 ( 4.0%)  █

Top genres (18 total, average 1.65 genres per item):
  Drama        : 1,603 (41.3%)
  Comedy       : 1,200 (30.9%)
  Action       :   503 (13.0%)
  Thriller     :   492 (12.7%)
  Romance      :   471 (12.1%)
```

### Implications

- **The "decade" of an item is a hidden dimension**. The EDA script already extracts `year` from `title` via the regex `r"\((\d{4})\)"`, but **does not persist it**. **User preference for old movies varies wildly**: a 25-year-old and a 55-year-old are entirely different populations.
- **Genres are extremely sparse** (avg 1.65 of 18, mostly zeros in multi-hot form), good for cross features but unsuitable as a primary feature
- Drama + Comedy account for 70%, meaning **the genre signal itself has limited discriminative power**

### "Free" feature shortlist for the future (no new data needed)

| Candidate feature | How to compute |
|---|---|
| Item decade | regex-extract from title (logic already exists in EDA script) |
| Item age (days) | `request_ts - first_seen_ts` (need to first build an item first-seen table) |
| User's preferred decade distribution | `groupby(user) × decade -> mean` |
| User's old vs. new movie preference ratio | derived ratio feature |
| Time-of-day of rating event (day-of-week / hour) | `ts.dt.dayofweek / ts.dt.hour` |

---

## 11. Versus real UGC video platforms

| Dimension | MovieLens-1M | TikTok / Kuaishou / Bilibili (typical) |
|---|---|---|
| Time window | 3-year cumulative | rolling 30~90 days |
| User scale | 6K | hundreds of millions DAU |
| Item scale | 4K | hundreds of millions in stock + tens of millions added daily |
| Per-user actions | avg 166 | heavy users: 100+ per day |
| Matrix sparsity | 4.47% | < 0.001% |
| Feedback types | 1 (rating) | 10+ (completion / like / comment / share / follow / not-interested / ...) |
| Item lifespan | static | 90% die within 24h |
| Real-time | none | sub-second feedback |
| Context | none | time-of-day / geo / device / network / previous swipe |
| Content features | genre tags | CV/NLP/ASR multi-modal embeddings |
| Social graph | none | follow / friends |
| Cold start | absent (filtered out by k-core) | core problem of the system |

### A tangible analogy

| | MovieLens-1M | Real UGC platform |
|---|---|---|
| Analogy | a long-established library's "reader borrowing log" | a 24/7 open street market |
| Pace | slow, stable, analyzable | fast, chaotic, ever-changing |
| Use | learn algorithms, run baselines | real business, real users, real money |

---

## 12. Reproducing every number in this doc

```bash
bash scripts/download_data.sh                     # download ML-1M (one-time)
.venv/bin/python scripts/01_eda.py                   # run EDA
```

The output will be **identical** to every number in §4 ~ §10 of this doc.

---

## 13. Preprocessing decisions baked into `02_prepare_samples.py`

The script `scripts/02_prepare_samples.py` turns the raw analysis above into 4 concrete preprocessing decisions. This section documents **what was decided** and, more importantly, **why** — so the rationale survives even if the script is rewritten.

### 13.1 Decision summary

| # | Parameter | Value | Rationale (one-liner) | Evidence |
|---|---|---|---|---|
| 1 | `min_user_inter` / `min_item_inter` (k-core) | **(5, 5)** | Drop sparse noise without losing real users | §6 + §7 |
| 2 | `positive_threshold` | **4** | 4–5 stars = "real like"; 1–3 = weak / noisy signal | §4 |
| 3 | `split_strategy` | **LOO** | Global time-split would leave ~30 eval users → unusable | §5 |
| 4 | `filter_eval_users` | **always on** | Otherwise unseen-in-train users artificially deflate Recall@K | (engineering invariant) |

### 13.2 Why these specific values

#### Decision 1 — k-core = (5, 5)

| Alternative | Trade-off | Verdict |
|---|---|---|
| k = 3 | Too lax — leaves "drive-by" users who hurt model quality | Reject |
| k = 5 ⭐ | Drops < 1% of rows (§6: min interactions per user = 20 anyway) | **Pick** |
| k = 10 | Over-filters — drops the realistic "light user" segment | Reject |

> **Reality check**: thanks to the dataset's built-in 20-interaction minimum (§6), k-core = (5, 5) is effectively a no-op on the user side and only filters a handful of one-shot movies on the item side. It's kept mainly as **a guard for future datasets**.

#### Decision 2 — positive_threshold = 4

| Threshold | Positives kept | Signal quality | Verdict |
|---|---|---|---|
| ≥ 3 | ~82% | Noisy: "meh" ratings polluting "like" | Reject |
| **≥ 4** ⭐ | ~58% (§4) | Clear: explicit thumbs-up | **Pick** |
| ≥ 5 | ~22% | Pure but too few — overfitting risk | Reject |

> **Industry alignment**: YouTube DNN, SASRec, SLIM, EASE — all foundational papers on ML-1M use **rating ≥ 4 as the positive label**. Picking 4 is a defensive choice that keeps results comparable to literature.

#### Decision 3 — `split_strategy = loo`

This is the single most consequential decision in the file. From §5:

> Two-thirds of the data is concentrated in a 30-day window in mid-2000. A global time-split with `test_days = 7` would leave **~30 unique users in test** — far too few for stable Recall@K.

LOO sidesteps this entirely:

| | Global time-split | Leave-one-out (LOO) ⭐ |
|---|---|---|
| Test users (with `min_inter = 5`) | ~30 | **6,032** |
| Defends against time leakage | ✓ (strictly) | ✗ (weakly — one user's test ts may precede another's train ts) |
| Stability of Recall@K | poor (high variance) | **good** |
| Industry use | streaming logs (Kuaishou, TikTok) | academic standard on MovieLens |

> **Verdict**: the cross-user "time leakage" in LOO is an accepted compromise — it does not break a recall model's training signal in any meaningful way on this dataset.

#### Decision 4 — `filter_eval_users` (always on)

```
LOO split  →  some valid/test rows reference users/items absent from train
                  ↓
Model trained on train  →  no embedding for those users/items
                  ↓
At eval time  →  always returns "no recommendation"  →  guaranteed miss
                  ↓
Recall@K is artificially deflated by N% — a metric artifact, not a model defect
```

**Solution**: drop those rows from valid/test before computing metrics. This is **not** "cheating" — it's removing rows the model is structurally unable to score. Every responsible recsys paper does this.

### 13.3 Tuning cheat sheet

When and why you might change each parameter:

| Goal | Change | Cost |
|---|---|---|
| **Tighter "like" definition** (higher-quality positives) | `positive_threshold: 4 → 5` | ~62% fewer positives; possible overfit on small data |
| **Accept weaker signals** (closer to implicit feedback) | `positive_threshold: 4 → 3` | +43% positives, but noise contaminates training |
| **Simulate "light user" cold start** | `min_user_inter: 5 → 3` | More sparse users, lower model accuracy, but more realistic |
| **Strict time-leakage prevention** | `split_strategy: loo → time` | Eval set collapses to ~30 users → unreliable comparisons |
| **More training data per eval user** | `loo_min_inter: 5 → 3` | More users qualify for eval, each with shorter train history |
| **Faster iteration in early dev** | `min_user_inter: 5 → 50` | Smaller dataset, faster to iterate; final results not comparable |

### 13.4 What changes when switching dataset

If you swap MovieLens-1M for **KuaiRand / Tenrec / MIND**, the same 4 decisions will likely change:

| Decision | MovieLens-1M | KuaiRand / Tenrec (typical) |
|---|---|---|
| k-core | (5, 5) — defensive, mostly no-op | (10, 10)+ — data is dense enough to afford it |
| positive_threshold | rating ≥ 4 | **N/A** — no explicit ratings; switch to `completion_rate ≥ 0.7` or `dwell_time ≥ 10s` |
| split_strategy | LOO | **time** — time-dense logs, time-split is the production-faithful choice |
| filter_eval_users | always on | still always on (engineering invariant) |

> The decisions are dataset-dependent. **Document them per dataset.** This section is the template for the next dataset's analogous §13.

### 13.5 Reproducing the preprocessing

```bash
bash scripts/download_data.sh                     # download ML-1M (one-time)
.venv/bin/python scripts/02_prepare_samples.py       # run the 5-stage pipeline
```

Outputs land in `data/processed/`:

```
train.parquet    ~563k rows  ⭐ feeds recall + ranking training
valid.parquet      ~6k rows  ⭐ early stopping / hyperparameter tuning
test.parquet       ~6k rows  ⭐ held out, final Recall@K / NDCG / Coverage
movies.parquet     ~4k rows  side info (title, genres)
users.parquet      ~6k rows  side info (gender, age, occupation, zip)
```

The script logs a full funnel summary at the end — the numbers it prints **must match** the figures cited in §13.1 and §13.2; any drift is a signal that either the script or this doc is out of sync.
