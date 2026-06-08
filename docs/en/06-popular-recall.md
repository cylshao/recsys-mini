# Popularity-based Recall

> **One-liner**: Popularity-based recall ranks items by a **global (or segment-level) interaction count**, ignoring who the user is. It is the **simplest, cheapest, and most reliable** recall route — not because it personalizes well (it does not), but because it is the **fallback that guarantees the system always has something to serve**. recsys-mini's `PopularRecaller` is a 36-line implementation of the global-static variant.
>
> **How to read this doc**: §1–§2 establish the role and notation; §3 formalizes the scoring; §4 enumerates the engineering pitfalls that separate a toy from a production fallback; §5 walks through the recsys-mini implementation; §6 covers the production-grade variants (time-decay, segmented, Wilson lower bound); §7–§8 cover failure modes and the route's place in a multi-route stack. For the broader recall context see [`04-recall.md`](04-recall.md) §3; for its personalized sibling see [`05-collaborative-filtering.md`](05-collaborative-filtering.md).

## Table of contents

- [1. Position in the recall stack](#1-position-in-the-recall-stack)
- [2. Notation and formal setup](#2-notation-and-formal-setup)
- [3. Scoring — from raw count to normalized score](#3-scoring--from-raw-count-to-normalized-score)
- [4. Engineering corrections that matter in production](#4-engineering-corrections-that-matter-in-production)
- [5. The recsys-mini implementation](#5-the-recsys-mini-implementation)
- [6. Production-grade variants](#6-production-grade-variants)
- [7. Failure modes and known limits](#7-failure-modes-and-known-limits)
- [8. When to use popularity recall and how to weight it](#8-when-to-use-popularity-recall-and-how-to-weight-it)
- [9. Production checklist](#9-production-checklist)
- [10. References](#10-references)
- [TL;DR](#tldr)

---

## 1. Position in the recall stack

Popularity recall is one of the eight recall families enumerated in [`04-recall.md`](04-recall.md) §2. Its defining property is **the input it deliberately throws away**:

| Signal type | Popularity recall uses it? |
|---|---|
| Aggregate item interaction count | **Yes — exclusively** |
| The identity of the requesting user | No (except to filter already-seen items) |
| User-item co-occurrence structure | No |
| Item content (titles, descriptions, images) | No |
| User profile / context | No (in the global variant) |

The consequence is twofold:

1. **It can never fail to return candidates** (as long as the catalog is non-empty), which makes it the **only route suitable as a system-wide fallback**.
2. **It is identical for every user** (modulo per-user already-seen filtering), so it carries **zero personalization signal**.

In the recall funnel popularity recall typically holds **5–10% of the candidate quota** — small by design. Its value is not measured by its standalone recall metric (which is mediocre) but by the three structural jobs only it can do:

| Role | What it guarantees |
|---|---|
| **Fallback** | The system never returns an empty result, even for a brand-new user |
| **Cold start** | A reasonable first-impression slate before any personalized signal exists |
| **Backfill** | Tops up the candidate pool when personalized routes return fewer than `K` items |

> See [`02-cold-start.md`](02-cold-start.md) for the cold-start regime where popularity recall is the default first move.

---

## 2. Notation and formal setup

Throughout this doc:

| Symbol | Meaning |
|---|---|
| `U` | Set of all users |
| `I` | Set of all items |
| `R` | The interaction table; each row is one `(user, item, ts)` event |
| `c_i` | The popularity count of item `i` (definition varies — see §4.4) |
| `I_u ⊆ I` | The set of items user `u` has already interacted with |
| `score(i)` | The normalized popularity score of item `i`, in `(0, 1]` |

The single **structural assumption** of popularity recall:

> **A1 (wisdom of the crowd)**: an item consumed by many users is, in expectation, a safe recommendation for an arbitrary user about whom nothing else is known.

This assumption is weak — it says nothing about *this particular* user — which is precisely why popularity recall is used as a **floor**, not as a primary personalized route.

---

## 3. Scoring — from raw count to normalized score

### 3.1 Raw popularity

The naive definition is a count of interactions per item:

```
c_i  =  |{ events in R with item = i }|
```

In pandas this is one line:

```python
pop = train_df.groupby("item_id").size().sort_values(ascending=False)
```

- `groupby("item_id")` groups all interaction rows by item
- `.size()` counts rows per group → the interaction count `c_i`
- `.sort_values(ascending=False)` ranks items from most to least popular

### 3.2 Why normalize the score

Popularity recall does not work in isolation — its output is fused with other recall routes (ItemCF, two-tower, content) before ranking. Raw counts (`c_i` can be in the thousands) are not on a comparable scale with, say, ItemCF's summed-similarity scores. To make fusion well-behaved, divide every count by the **maximum** count:

```
score(i)  =  c_i / max_j c_j      ∈ (0, 1]
```

The most popular item gets `1.0`; everything else scales down proportionally.

| Item | Raw count `c_i` | Normalized `score(i)` |
|---|---|---|
| most popular | 560 | 560/560 = **1.00** |
| second | 480 | 480/560 = 0.857 |
| third | 450 | 450/560 = 0.804 |

> This mirrors the per-query min-max normalization ItemCF applies for the same reason (see [`05-collaborative-filtering.md`](05-collaborative-filtering.md) §6.6). When the fusion strategy is RRF (rank-based, as in recsys-mini's `MultiRecaller`), normalization is less critical because RRF operates on ranks — but it costs nothing and future-proofs the route against learned fusion.

---

## 4. Engineering corrections that matter in production

"Sort by count" is a one-liner, but a production-grade fallback needs the four corrections below.

### 4.1 Already-seen filtering

The ranked list is global, but each user has seen a different subset of it. Recommending an item the user has already consumed is a self-evident defect, so the filter must be applied **per user, at recall time**:

```python
seen = self._user_history.get(user_id, set())
# while walking the ranked list: if item_id in seen: continue
```

Store the seen set as a `set`, not a `list`: membership testing is `O(1)` for a set versus `O(n)` for a list, and this test runs once per item in the ranked walk.

### 4.2 Popularity bias / the Matthew effect

Popularity recall is the **purest amplifier of the rich-get-richer dynamic**: popular items are surfaced to everyone → consumed more → become more popular. The symptom is **catastrophically low coverage**. recsys-mini's popularity route has a measured `Coverage@200` of just **22.29%** — meaning 78% of the catalog never appears in any user's popularity candidates.

Mitigations:

- **Low fusion weight** — cap popularity's contribution (recsys-mini uses `popular: 0.3` vs `itemcf: 1.0`)
- **Parallel long-tail routes** — content recall and Swing surface items popularity never will
- **Segmented popularity** (§6.2) — many small leaderboards spread exposure wider than one global list

### 4.3 Freshness / refresh cadence

Popularity is a **time-sensitive** quantity. A statically computed leaderboard cannot capture a trend that spikes after the leaderboard was built:

```
A title goes viral on Wednesday
  → the leaderboard was computed on Monday
  → the title is not in the list
  → it is never recalled until the next rebuild
```

The refresh cadence is a direct trade-off: hourly > daily > weekly for trend capture, but at proportionally higher compute cost. recsys-mini computes the leaderboard once in `fit` (static popularity) — sufficient for an offline benchmark, blind to real-time trends.

### 4.4 Count definition — events vs unique users

A single super-fan who replays one title 50 times will inflate that title's count if you count **events**. The more robust definition counts **distinct users**:

```python
# events (recsys-mini's current choice)
c_i = train_df.groupby("item_id").size()

# distinct users (more robust for repeat-consumption domains)
c_i = train_df.groupby("item_id")["user_id"].nunique()
```

On ML-1M the distinction barely matters (a rating dataset is near one-event-per-user-item), but in music / short-video domains where the same item is consumed repeatedly, **count distinct users** is the safer default.

---

## 5. The recsys-mini implementation

The implementation lives in `src/recall/popular.py` — 36 lines total. Walkthrough by responsibility.

### 5.1 Class state and fit

```11:24:src/recall/popular.py
class PopularRecaller(BaseRecaller):
    name = "popular"

    def __init__(self, topk: int = 200):
        self.topk = topk
        self._top_items: List[Tuple[int, float]] = []
        self._user_history: dict[int, set[int]] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        pop = train_df.groupby("item_id").size().sort_values(ascending=False)
        # score 归一化到 [0, 1] 便于多路融合
        max_p = pop.iloc[0]
        self._top_items = [(int(i), float(c) / max_p) for i, c in pop.items()]
        self._user_history = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
```

| Member | Type | Holds |
|---|---|---|
| `name = "popular"` | str | The route identifier used to look up its fusion weight |
| `topk` | int | Max leaderboard length (default 200) |
| `_top_items` | `List[(item_id, score)]` | The global leaderboard, popularity-descending, scores in `(0, 1]` |
| `_user_history` | `dict[user, set[item]]` | Each user's seen set, for already-seen filtering |

`fit` does exactly the §3 + §4.1 work:

| Line | Responsibility | Maps to |
|---|---|---|
| `groupby("item_id").size().sort_values(...)` | Count interactions, rank descending | §3.1 |
| `max_p = pop.iloc[0]` | Take the top count as the normalization denominator | §3.2 |
| list comprehension `float(c) / max_p` | Normalize every item to `(0, 1]` | §3.2 |
| `groupby("user_id")["item_id"].apply(set)` | Build each user's seen set | §4.1 |

> Like ItemCF, all work is front-loaded into `fit`; `recall` is a pure table walk. Offline cost is irrelevant; online latency must be minimal.

### 5.2 Recall — online scoring

```26:35:src/recall/popular.py
    def recall(self, user_id: int, k: int) -> List[Tuple[int, float]]:
        seen = self._user_history.get(user_id, set())
        out: List[Tuple[int, float]] = []
        for item_id, score in self._top_items:
            if item_id in seen:
                continue
            out.append((item_id, score))
            if len(out) >= k:
                break
        return out
```

| Line | Responsibility | Note |
|---|---|---|
| `seen = self._user_history.get(user_id, set())` | Fetch the user's seen set | **Unknown user → empty set → full leaderboard returned**; this default is the source of the route's fallback power |
| `for item_id, score in self._top_items:` | Walk the leaderboard top-down | Pre-sorted, start from the most popular |
| `if item_id in seen: continue` | Skip already-seen items | §4.1, `O(1)` set test |
| `out.append((item_id, score))` | Collect the candidate with its normalized score | |
| `if len(out) >= k: break` | Stop once `k` candidates are collected | No need to scan the full list |

### 5.3 Contrast with ItemCF's recall

The single table that explains why this route exists:

| | ItemCF | Popular |
|---|---|---|
| New / unknown user | returns `[]` (cannot recommend) | returns the **global leaderboard** (fallback) |
| Personalization | per-user | **identical for all users** (modulo seen-filtering) |
| Online work per query | double loop accumulating neighbor similarities | single leaderboard walk |
| Online complexity | `O(H · K)` | `O(min(|leaderboard|, k + |seen|))` |

ItemCF's "cannot recommend" cell for new users is exactly the gap popularity recall fills. The two are complementary, not redundant.

### 5.4 What is **not** in the current implementation

| Correction / variant | Status | Notes |
|---|---|---|
| Already-seen filtering | ✅ Implemented | per-user `set` |
| Score normalization for fusion | ✅ Implemented | divide by max count |
| Event-count popularity | ✅ Implemented | `groupby().size()` |
| Distinct-user popularity | ⏳ Optional | swap to `nunique("user_id")` (§4.4) |
| Time-decayed popularity | ⏳ Future | §6.1 |
| Segmented / contextual popularity | ⏳ Future | §6.2 |
| Wilson / Bayesian smoothing | ⏳ Future | §6.3 — relevant only when ranking by a *rate*, not a count |

---

## 6. Production-grade variants

recsys-mini ships the global-static variant. Production systems commonly add one or more of the following.

### 6.1 Time-decayed popularity

Naive popularity treats "500 plays last month" and "500 plays last week" identically, but recent interactions are more predictive of current demand. Weight each interaction by its age:

```
c_i  =  Σ_{events of i}  exp(−α · Δt)
```

where `Δt` is the event's age and `α` is tuned so that, e.g., a one-week-old event retains 50% of its weight. This is the cheapest way to make popularity trend-aware (§4.3).

### 6.2 Segmented / contextual popularity

A single global leaderboard is coarse. Compute separate leaderboards per:

- **Category** (a sci-fi leaderboard, a romance leaderboard, …)
- **Region** (per-city / per-locale)
- **Cohort** (age band, device, new vs returning)

This is "pseudo-personalization" — far cheaper than a model, but materially better UX than one global list, and it spreads exposure (mitigating §4.2).

### 6.3 Wilson lower bound / Bayesian smoothing

When the leaderboard ranks by a **rate** (e.g. like-rate, CTR) rather than a raw count, small samples produce spurious winners:

```
Item A: 2 plays, 2 likes   → 100% like-rate
Item B: 1000 plays, 900 likes → 90% like-rate
```

Ranking by raw rate puts A above B despite A's two-sample evidence. The **Wilson score interval lower bound** (or a Beta-Binomial Bayesian posterior) discounts low-confidence rates, so high-volume high-rate items rank robustly. Standard in news ranking and review-score sorting.

> Note: this matters only when ranking by a rate. recsys-mini ranks by a raw count, for which the relevant correction is the count definition (§4.4), not confidence smoothing.

### 6.4 Variant summary

| Variant | Solves | Complexity |
|---|---|---|
| Time-decayed | Trend lag (§4.3) | ⭐ |
| Segmented / contextual | Coarse global list, no personalization | ⭐⭐ |
| Wilson / Bayesian | Spurious small-sample winners (rate-based) | ⭐⭐ |
| Real-time streaming counts | Static leaderboard refresh lag | ⭐⭐⭐ |

---

## 7. Failure modes and known limits

| Failure mode | Symptom | Cause |
|---|---|---|
| **Zero personalization** | Everyone sees the same slate | The route does not look at user identity |
| **Matthew effect** | Long-tail items never surface (`Coverage@200 = 22.29%`) | Rich-get-richer amplification (§4.2) |
| **New-item cold start** | A just-added item has count 0 → never on the leaderboard | Same structural issue as ItemCF — no interactions, no signal |
| **Trend lag** | Static leaderboard misses sudden spikes | Refresh cadence (§4.3) |

The standalone numbers confirm the route is a floor, not a primary engine:

| Recall method | Recall@10 | Recall@50 | Recall@200 | Coverage@200 |
|---|---|---|---|---|
| ItemCF | 5.29% | 20.08% | **50.15%** | 69.89% |
| **Popular (fallback)** | 4.74% | 15.48% | 36.97% | **22.29%** |
| RRF fusion | 6.08% | 19.50% | 50.20% | 69.36% |

Reading the table:

1. Popularity trails ItemCF on every metric — expected; it is not the primary route.
2. Its `Recall@10` of 4.74% is close to ItemCF's 5.29%, confirming that mass-appeal hits cover a meaningful slice of users — the fallback quality is decent.
3. Its `Coverage@200` of 22.29% is the empirical fingerprint of the Matthew effect (§4.2) — the reason it gets a low fusion weight.

---

## 8. When to use popularity recall and how to weight it

### Always include it (as a fallback)

Every production recall stack should keep a popularity route, because it is the only one that **cannot return empty**. The question is never "should we have it" but "**how much weight should it get**".

### Weighting

```
High weight → candidate pool dominated by head items → personalization diluted → "same-for-everyone" UX
Low  weight → popularity only backfills when personalized routes underdeliver → safety net without crowding out
```

recsys-mini's `popular: 0.3` (vs `itemcf: 1.0`) is the "backfill but don't crowd out" setting — popularity stands quietly behind ItemCF and only steps in when needed.

### CF's place vs popularity's place

```
┌──────────────────────────────────────────────────────────────┐
│  Modern multi-route recall (typical production layout)        │
│                                                               │
│   Two-tower recall      30–40%  ── personalized, generalizing │
│   ItemCF / Swing        15–25%  ── personalized, popular-skew │
│   Inverted tag recall   10–15%  ── explainable, instant cold  │
│   Content recall         5–15%  ── new items, sparse regimes  │
│   Follow / social       10–20%  ── high-affinity traffic      │
│   Popularity fallback    5–10%  ── safety net  ◄── this doc   │
│                                                               │
│         ↓                                                     │
│   Quota allocation + dedup + RRF / learned fusion             │
│         ↓                                                     │
│   200–500 candidates → ranking                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Production checklist

A popularity route is ready to ship when it satisfies all of the following:

| Checkpoint | Question |
|---|---|
| **Count definition** | Events or distinct users? Documented and appropriate for the domain (§4.4)? |
| **Normalization** | Are scores in `(0, 1]` so they fuse sensibly with other routes? |
| **Already-seen filter** | Are seen items excluded **per user, online** (not pre-baked)? |
| **Fallback wiring** | When personalized routes return `[]`, does the multi-route layer actually fall back to this route? |
| **Backfill behavior** | When personalized routes return `< K`, does popularity top up the pool? |
| **Fusion weight** | Is the weight low enough to avoid crowding out personalization (e.g. ≤ 0.3 of the primary route)? |
| **Coverage monitoring** | Is the route's coverage tracked so the Matthew effect is visible? |
| **Refresh cadence** | Is the leaderboard rebuilt often enough for the domain's trend velocity? |
| **Segmentation** | For coarse global lists, is a category/region/cohort split worth the small cost? |

---

## 10. References

### Foundational / related work

| Source | Why it matters |
|---|---|
| Wilson, *Probable Inference, the Law of Succession, and Statistical Inference* (1927) | The Wilson score interval used in confidence-aware popularity ranking (§6.3) |
| Miller, *How Not To Sort By Average Rating* (Evan Miller, 2009) | The canonical engineering post on Wilson lower bound for rating sorts |
| Steck, *Item Popularity and Recommendation Accuracy* | RecSys'11 — the standard analysis of popularity bias and its effect on offline metrics |
| Cremonesi, Koren, Turrin, *Performance of Recommender Algorithms on Top-N Recommendation Tasks* | RecSys'10 — establishes the popularity baseline as a deceptively strong top-N competitor |
| Abdollahpouri et al., *Managing Popularity Bias in Recommender Systems* | FLAIRS'19 — mitigations for the Matthew effect (§4.2) |

### Related docs

- [`04-recall.md`](04-recall.md) — broader recall context; §3 places popularity among the eight recall families
- [`05-collaborative-filtering.md`](05-collaborative-filtering.md) — the personalized sibling route (ItemCF), with which popularity is fused
- [`02-cold-start.md`](02-cold-start.md) — the cold-start regime where popularity recall is the default first move
- `src/recall/popular.py` — the implementation walked through in §5
- `src/recall/multi_recall.py` — the RRF fusion layer that combines popularity with ItemCF

---

## TL;DR

> Popularity-based recall ranks items by a **global interaction count**, normalized to `(0, 1]` for fusion. recsys-mini's `PopularRecaller` implements the global-static variant in 36 lines (`src/recall/popular.py`): count, sort, normalize, and filter already-seen items per user.
>
> **Its value is not its standalone metric** (`Recall@200 = 36.97%`, well below ItemCF's 50.15%) but its **structural role as a fallback** — it is the only route that cannot return empty, which makes it indispensable for new users, cold start, and backfilling underfull candidate pools.
>
> **Its weaknesses are the Matthew effect** (`Coverage@200` of only 22.29%) **and zero personalization**. Therefore it is never used alone and is given a **deliberately low fusion weight** (recsys-mini: `0.3`), letting it backfill quietly behind ItemCF, two-tower, and content recall. Production upgrades — time decay, segmented leaderboards, Wilson smoothing — all exist to make the floor a little smarter without ever promoting it to the primary route.
