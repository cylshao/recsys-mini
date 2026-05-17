# Cold Start

> The **least sexy, most business-critical** chapter of recommender systems. Algorithm papers barely mention it, but PMs and applied scientists in production spend 50% of their time here.
>
> **Core question**: when behavior data does not exist, how do you make recommendations that beat "just push the blockbusters" using **every other available signal**?

## Table of contents

- [1. Why cold start is the #1 challenge](#1-why-cold-start-is-the-1-challenge)
- [2. The three kinds of cold start](#2-the-three-kinds-of-cold-start)
- [3. User cold start](#3-user-cold-start)
- [4. Item cold start](#4-item-cold-start)
- [5. System cold start](#5-system-cold-start)
- [6. Exploration vs. Exploitation (E&E) — the essence of cold start](#6-exploration-vs-exploitation-ee--the-essence-of-cold-start)
- [7. Where cold start lives in the architecture](#7-where-cold-start-lives-in-the-architecture)
- [8. Evaluation — the "uncertainty principle" of cold start](#8-evaluation--the-uncertainty-principle-of-cold-start)
- [9. Cold start status of recsys-mini / MovieLens](#9-cold-start-status-of-recsys-mini--movielens)
- [10. Further reading](#10-further-reading)

---

## 1. Why cold start is the #1 challenge

### The death spiral

```
New user → model knows nothing → can only push blockbusters as fallback
   ↓
Bad experience (everything looks the same and homogeneous)
   ↓
User doesn't click, doesn't like, doesn't watch
   ↓
No new behavior data → model knows even less
   ↓
Retention collapses → user churns
   ↓
DAU drops
```

**Compare to a power user**: the model has hundreds of past actions to draw on, recommendation quality is stable → user stays → more behavior → better model → virtuous cycle.

> **The gap between cold start and a power user is essentially "starting the flywheel" vs. "the flywheel is already spinning"**.

### How big is the business impact

| Platform | Cold-start-related metric | Why it matters |
|---|---|---|
| TikTok / Douyin / Kuaishou | New-user D1 / D7 / D30 retention | Determines the growth curve |
| YouTube | First-watch completion + first-subscribe rate | Drives long-term engagement and creator economy |
| Netflix | New-subscriber first-month retention | Directly affects churn (Netflix's #1 KPI) |
| Spotify | New-user first-week active days | Drives Premium subscription conversion |
| Instagram / Pinterest / Snapchat | New-user first-follow + first-save | Builds the personalization graph |
| Bilibili / Xiaohongshu | First-completion / first-interaction rate of new users | Direct conversion + segmentation |
| Amazon / Shopee / Mercado Libre | 14-day first-purchase rate of new shoppers | Determines lifetime value (LTV) |
| App Store / Google Play | First-7-day app retention | Determines supply-side (developer) health |

Every **percentage point of D1 retention** translates to hundreds of millions of GMV / tens of millions of DAU on a large-scale platform — true whether you're talking Meta, ByteDance, Google, or Amazon.

---

## 2. The three kinds of cold start

| Type | Trigger scenario | Pain point | Key tools |
|---|---|---|---|
| **User cold start** | Newly registered / dormant returner / anonymous visitor | No user behavior | Onboarding interest selection + demographic profile + Bandit exploration + Look-alike |
| **Item cold start** | New video uploaded / new SKU listed | No item behavior | **Item understanding (CV/NLP/tags)** + forced exposure budget + I2I look-alike |
| **System cold start** | New product line / new scenario / new region launched | No data anywhere | Editorial picks + rules + cross-scenario transfer |

The solutions are **different in approach but share a common idea**: **fill the behavior gap with non-behavioral signals that the model isn't great at on its own**.

---

## 3. User cold start

### "Coldness" tiers for users

| Stage | Action count | Treatment |
|---|---|---|
| **Fully cold** | 0 | Pure fallback: blockbusters + interest-selection card |
| **Very cold** | 1 ~ 5 | Untrustworthy signal, but "seed interests" begin to form |
| **Half cold** | 5 ~ 30 | Model usable, but keep some exploration |
| **Lukewarm** | 30 ~ 100 | Model-driven, but downweight cautiously |
| **Normal** | > 100 | Standard pipeline |

> Note: **a long-dormant user who returns** should also be treated as "half cold". A 1-year-old interest signal cannot be used directly.

### Five strategies

#### Strategy 1: Onboarding interest selection (most direct)

Force new users to pick 5–10 interest tags upon entering the app:

```
Pick at least 5 topics you're interested in:
☐ Food   ☐ Travel   ☐ Fitness   ☐ Tech  ...
```

**Pro**: get a strong signal in seconds
**Con**: hurts onboarding flow and causes dropout; the "interests" users pick ≠ true interests
**In practice**: nearly every content app does this — Netflix ("pick 3 titles you've enjoyed"), Spotify (favorite artists/genres), Pinterest (pick 5 topics), YouTube ("subscribe to creators"), Apple News (topic selection), TikTok / Xiaohongshu / Bilibili (interest-tag grid). ROI is well validated across markets

#### Strategy 2: Default recommendation based on demographic profile

```
A new user is observed as (female, 22, iPhone, NYC, signed up via Instagram ad) →
look up the high-CTR content for the "22-year-old female NYC iPhone user" cohort →
serve the top N items
```

**Pro**: needs no user behavior
**Con**: profile data itself is sparse (most users skip demographic fields globally — this is true on both Chinese and Western apps), regulations limit what you can use (GDPR, CCPA, COPPA), and demographic targeting can amplify bias
**In practice**: used as the fallback for "fully cold" users on every major platform (Pinterest leans on it heavily for interest seeding; Snapchat uses age + city; Spotify uses country for chart-based recs)

#### Strategy 3: Look-alike (find similar power users)

```
A new user's device / IP / phone model / acquisition channel →
find similar power users → see what they like
```

**How it works**:
- Run a nearest-neighbor search over device features (e.g. LSH)
- Take the average interest embedding of the top-K similar power users
- Use that as the new user's initial embedding

**Pro**: provides personalization even without explicit signals
**Con**: weak when device info is unavailable (web), and increasingly constrained by Apple's ATT, IDFA deprecation, and privacy regulations
**In practice**: pioneered by **Meta with "Lookalike Audiences"** (originally for Facebook ads, then extended to organic ranking and Instagram); also heavily used at YouTube, Pinterest, Snapchat, TikTok / Kuaishou

#### Strategy 4: Bandit exploration (the core E&E tool)

Every recommendation slot for a new user is a chance to **probe**. **Classic algorithm family**:

| Algorithm | Idea | Best for |
|---|---|---|
| **ε-greedy** | Recommend the best with prob 1−ε, explore randomly with prob ε | Easy to implement, baseline |
| **UCB** | Add an "uncertainty bonus" to each candidate (less-explored gets a boost) | Solid middle-of-the-road |
| **Thompson Sampling** | Sample from each candidate's CTR posterior | Most popular in industry ⭐ |
| **LinUCB / LinTS** | Bandits with features | More "personalized" exploration |

**Thompson Sampling intuition**:

```
The true CTR of a tag is unknown — model it with a Beta distribution:
  Beta(α = clicks + 1, β = impressions − clicks + 1)

When ranking, draw one sample from each candidate's Beta distribution and pick the largest.
  → Less-explored candidates: wide distribution → occasionally sample a high value → exploration
  → Well-explored candidates: narrow distribution → converges to true CTR
```

#### Strategy 5: Transfer learning (cross-app / cross-product)

If your company runs multiple products, **new users of a new product can borrow interest signal from older ones**. This is the single biggest cold-start advantage that "platform companies" hold over standalone apps. Canonical examples:

- **Meta Family of Apps** (Facebook + Instagram + WhatsApp + Threads) — IG signal seeds Threads cold start; FB social graph feeds IG "people you may know"; cross-app interest embeddings shared across the family
- **Google ecosystem** (Search + YouTube + Discover + Maps + Shopping + Ads) — YouTube watch history shapes the Discover feed; Search history powers Ads cold start; Maps signals inform local content recs
- **Apple** (Music + Podcasts + News + TV+ + App Store) — unified Apple ID enables cross-app interest transfer (within ATT/privacy boundaries)
- **Amazon** (retail + Prime Video + Music + Kindle + Audible) — purchase history bootstraps Prime Video recs; reading history bootstraps Audible
- **Tencent** (QQ + WeChat + Weishi + QQ Music) — QQ/WeChat profile bootstraps Weishi recommendation
- **ByteDance** (TikTok + Douyin + Toutiao + Lark + CapCut) — cross-app signal across the family

Caveats:
- **Legal compliance**: user consent, data isolation, regional regulations (GDPR in EU, CCPA in California, Apple's App Tracking Transparency, China's PIPL, India's DPDP Act). Some transfers that are legal in one region are blocked in another
- **Scenario shift**: WhatsApp messaging interests may not transfer cleanly to Threads news consumption; Amazon book purchases don't always predict Prime Video preferences
- **Antitrust scrutiny**: regulators in EU/US increasingly question whether bundling cross-product data unfairly disadvantages competitors

### Gradual fusion

The transition from cold to warm is **not a step function**, but a smooth ramp:

```
Weighted fusion:
  final_score = w_explore · explore_score + w_history · history_score

  Fully cold (0 actions):    w_explore = 1.0   w_history = 0
  Half cold (10 actions):    w_explore = 0.5   w_history = 0.5
  Lukewarm (50 actions):     w_explore = 0.2   w_history = 0.8
  Normal (>100 actions):     w_explore = 0.05  w_history = 0.95
                              ← always retain at least 5% exploration
                                to prevent filter bubbles
```

---

## 4. Item cold start

UGC platforms see **hundreds of thousands to millions** of new items per day, and 90% "die" within 24 hours (no further exposure). Item cold start **directly determines supply-side health**.

### "Coldness" tiers for items

| Stage | Cumulative impressions | Treatment |
|---|---|---|
| **Unexposed** | 0 | Indexing → item understanding → enter the cold-start pool |
| **Very cold** | 1 ~ 100 | Forced exposure budget (allocate test traffic) |
| **Half cold** | 100 ~ 1000 | Behavior signals start to be trustworthy; ramp into the main pipeline |
| **Normal** | > 1000 | Standard pipeline |

### Five strategies

#### Strategy 1: Content-similarity recall via item understanding ⭐

This is the **fundamental solution**:

```
New item ──→ multi-modal understanding ──→ video_emb
                                              │
                                              ▼
                       ANN index → find top-K similar established items
                                              │
                                              ▼
                Attach the new item to those established items' "interest set"
                                              │
                                              ▼
        Users who liked the established items → recall the new item
```

**Why this is the "fundamental solution"**: it bypasses the "needs behavior data" constraint and **delivers personalization purely from content signals**.

#### Strategy 2: Forced exposure budget (Boost Pool)

Give every new item a **fixed N impression chances** (say 500), inserted regardless of model score:

```
Final step of the recommendation pipeline:
  if no new-item exposure in this user's history and the cold-start pool is non-empty:
    force-insert 1 new item
```

**What it does**:
- Collects behavior data → model can learn
- Protects new-creator motivation (no traffic = no posts)
- Creates "breakout" opportunities (blockbusters always emerge from here)

**Key parameters**:
- Budget size (too small = no data; too large = hurts experience)
- Time window (must be used within X time, expires otherwise)
- User segmentation (allocate more to high-tolerance users)

#### Strategy 3: Author-based propagation

A new video is cold, but **the author may not be**:

```
New video (0 exposures) ─→ author has 1M existing followers
                                  │
                                  ▼
                    Push to active followers of the author
                    (via the "follow" recall path)
```

**Key insight**: **the author–follower relationship is a free signal for item cold start**. Used everywhere:
- **YouTube** — subscribers receive notifications and the new video gets a guaranteed slot in the Subscriptions tab
- **Instagram** — new posts appear in followers' feed first; Reels "Following" tab guarantees impressions
- **TikTok** — the "Following" feed is one of three top-level tabs and follower interactions feed back into the FYP score
- **Twitch** — live notifications + raid mechanics propagate new streams via the follower graph
- **Substack / Medium** — email push to subscribers is the entire item cold-start mechanism
- **Spotify / Apple Music** — "New Music Wednesday" pushes new releases to artist followers

Most platforms actively encourage "fans first, then break out".

#### Strategy 4: Look-alike to find similar established items

```
New-item emb ─→ ANN find top-K similar old items ─→ reuse their "user-preference profile"
```

Similar to Strategy 1, but the focus is on **reusing the established items' user distribution** rather than the recall path itself.

#### Strategy 5: Time-decayed popularity + exploration pool

Place new items in a **separate "exploration pool"**:

```
Exploration pool = items uploaded in the last 24h with cumulative exposure < 100
        ↓
        Explore via Bandit / time-decay + ε-greedy
        ↓
        Exposures → collect CTR → graduate (move to main pool) / drop
```

### The "horse race" mechanism for items

The core playbook at TikTok / Kuaishou — and visibly mirrored in YouTube's creator algorithm, Instagram Reels distribution, and Snapchat Spotlight:

```
New item uploaded
   ↓
[Round 1] Force 100 exposures to geographically/interest-similar target users
   ↓ collect CTR / completion / interaction
   ↓
[Round 2] CTR > threshold?
       Yes → bump traffic to 1k → next round
       No  → drop
   ↓
[Round 3] Still performing well after 1k exposures?
       Yes → bump to 10k → next round
       No  → drop
   ↓
... all the way up to a full-platform blockbuster
```

**This is the "horse race / traffic ladder"**. It is the core supply-side scheduling mechanism on UGC platforms.

---

## 5. System cold start

When a brand-new business / scenario / region launches, **there is no data anywhere**.

### Three typical situations

#### Situation 1: Brand-new app / brand-new scenario

Examples: Pinterest in 2010, Instagram in 2010, TikTok in 2017, Threads in 2023, Xiaohongshu in 2013, BeReal in 2020 — at day zero there was no user behavior to learn from.

**Countermeasures**:
1. **Editorial picks**: experts/editors curate tens of thousands of high-quality items into the catalog
2. **Rule-based recommendations**: simple ranking by popularity / recency / geography
3. **Seed-user operations**: invite KOLs to onboard, build the "content pool" first
4. **Cross-scenario transfer**: if the parent company has another product, transfer user interests over

#### Situation 2: Launching in a new region

Examples: TikTok entering Indonesia (2018), Netflix expanding to India (2016), Spotify launching across Latin America (2013), Disney+ rolling out to dozens of markets in months (2020+), YouTube Shorts going global (2021), Threads launching in the EU (2023, delayed by GDPR).

**Countermeasures**:
1. Reuse the parent company's global user behavior as a fallback
2. Localized onboarding (Indonesian users see local content first)
3. Local KOL matrix to bootstrap supply

#### Situation 3: New scenario within the same app

Example: home Feed → adding a "Nearby" tab.

**Countermeasures**:
1. Reuse home-feed users' interest embeddings
2. Layer scenario-specific rules on top (Nearby tab requires LBS)
3. Run AB tests, ramp up gradually

### The core logic of system cold start

> **"Cold start coverage ∝ proportion of editorial / rules / transferred content"**

```
Brand-new business (0 data):  100% editorial / rules
After 1 week of data:          70% editorial + 30% model
After 1 month of data:         30% editorial + 70% model
After 3 months of data:         5% editorial + 95% model
```

---

## 6. Exploration vs. Exploitation (E&E) — the essence of cold start

### The classic dilemma

```
Exploitation: recommend what the user is known to love → high CTR short term
Exploration:  recommend uncertain but potentially high-value items → low CTR short term, discovers new interests long term

   ┌──────── Short term ────────┐  ┌──────── Long term ────────┐
   │                             │  │                             │
   │ Exploit → CTR ↑ retention ↑ │  │ Filter bubble → boredom →    │
   │                             │  │ churn                       │
   │                             │  │                             │
   │ Explore → CTR ↓             │  │ New interests → diversity → │
   │                             │  │ retention                   │
   │                             │  │                             │
   └─────────────────────────────┘  └─────────────────────────────┘
```

### Cold start = the special period when exploration weight is highest

```
Weight setting:
  Fully cold user:     exploration 100%
  Half cold:           exploration 50%, exploitation 50%
  Normal:              exploration 5-10%, exploitation 90-95%

  Fully cold item:     100% via exploration pool
  Half cold:           30% via exploration + 70% via main pipeline
  Normal:              100% via main pipeline
```

### The cost of exploration

**It is not free**. Every "exploration" impression is a "exploitation" opportunity lost. The key when designing E&E systems is:

> **How do we trade the least "exploration cost" for the most "information gain"**?

This is the central question of the entire Multi-Armed Bandit field.

### Bandit difficulties in recommendation settings

Theoretical Bandits assume "each arm's reward is independent". Recommendation violates this:
- **Exploding number of arms** (hundreds of millions of items, impossible to probe one by one)
- **Reward is context-dependent** (CTR for the same item differs across users)
- **Reward is time-varying** (item popularity decays over time)
- **Reward is delayed** (completion signal arrives minutes later)

**Solution**: **Contextual Bandit + model** (LinUCB / Neural Bandit / Wide&Deep + Bandit head).

---

## 7. Where cold start lives in the architecture

In the classic five-stage recsys pipeline (data → recall → filter → rank → re-rank), cold start **is not a separate stage; it is woven into every stage**:

```
┌──────────────────────────────────────┐
│ ① Data source                        │
│   ★ item-understanding module emits   │
│     embeddings for cold items         │
│   ★ user demographic / device         │
│     fingerprints                      │
└────────────────┬─────────────────────┘
                 ▼
┌──────────────────────────────────────┐
│ ② Cold-start router                  │
│   - Returning user → standard pipeline│
│   - New user      → cold-start path   │
└────────────────┬─────────────────────┘
                 ▼
┌──────────────────────────────────────┐
│ ③ Recall                             │
│   ★ tag-based recall (using onboarding│
│     tags)                             │
│   ★ content-similarity recall         │
│     (cold items)                      │
│   ★ follow-path recall (cold items    │
│     borrow author's followers)        │
│   ★ exploration-pool recall           │
└────────────────┬─────────────────────┘
                 ▼
┌──────────────────────────────────────┐
│ ④ Filter                             │
│   (no special handling, but avoid    │
│    over-filtering new items)         │
└────────────────┬─────────────────────┘
                 ▼
┌──────────────────────────────────────┐
│ ⑤ Ranking                            │
│   ★ cold-start CTR model              │
│     (pure content features)           │
│   ★ Bandit / exploration bonus term   │
└────────────────┬─────────────────────┘
                 ▼
┌──────────────────────────────────────┐
│ ⑥ Re-ranking                         │
│   ★ enforce cold-start quotas         │
│   ★ higher diversity (especially      │
│     critical for new users)           │
└──────────────────────────────────────┘
```

### Dual-pipeline design

A common industry approach is to **run a "returning user" pipeline and a "cold-start" pipeline in parallel, with backend AB routing**:

```
                ┌─── Returning-user pipeline (standard 5 stages) ───┐
Request ─split─┤                                                    ├── output
                └─── Cold-start pipeline (looser rules) ────────────┘
```

**Why not merge them**:
- Different model hyper-parameters (cold start needs higher exploration, lower quality threshold)
- Different SLA (cold start can be slightly slower; returning users must be fast)
- Different monitoring metrics (cold start watches new-user retention; returning users watch CTR)

---

## 8. Evaluation — the "uncertainty principle" of cold start

### Offline evaluation is nearly impossible

| Issue | Why |
|---|---|
| No ground-truth for cold users | The dataset itself was k-core filtered |
| Exploration actions cannot be replayed | "If you had recommended something different, would the user have clicked?" cannot be answered from logs |
| Counterfactual estimation is hard | Counterfactual evaluation needs heavy machinery (IPW / Doubly Robust) |

### Core online metrics

| Dimension | Metric | Direction |
|---|---|---|
| **New-user retention** | D1 / D7 / D30 retention | up |
| **New-user first-interaction rate** | First like / follow / comment | up |
| **New-item breakout rate** | Share of items reaching ≥ threshold cumulative impressions within 24h of upload | up |
| **New-item traffic concentration** | Share of new-item impressions taken by the top 1% of new items | down (means traffic is well-spread) |
| **Cold-start pool digestion rate** | Share of items entering the pool that are eventually consumed | up |
| **Model exploration cost** | CTR loss on the exploration traffic | keep within budget |

### Special considerations for AB tests

Cold-start AB experiments are harder to design than typical ones:
- **New users enter the experiment slowly** (limited daily registrations)
- **Effect-observation window is long** (retention metrics need at least 7 days)
- **Sample imbalance** (a power-user experiment may have 1M samples, cold-start only 10k)

**Common practices**:
1. Allocate a larger traffic share to cold-start experiments (50:50 instead of 90:10)
2. Look at long-term metrics (D7 / D30) rather than short-term ones
3. Run multiple AB groups in parallel to shorten the iteration cycle

---

## 9. Cold start status of recsys-mini / MovieLens

### Status: nearly impossible to do for real

Recap from [`docs/01-data-source-analysis.md`](01-data-source-analysis.md):

| Cold-start dimension | Feasibility on MovieLens-1M |
|---|---|
| **Cold users** | ❌ k-core filtering at packaging time means every user has ≥ 20 actions |
| **Cold items** | ⚠️ 5.5% of items have ≤ 2 interactions, but **there is no upload-time dimension** |
| **New scenario** | ❌ single scenario only |
| **Multi-modal understanding** | ❌ no raw files, no CV/NLP understanding possible |

### But some "simulations" can be done

#### Simulation 1: artificially create cold users

Sort each user's actions by time and **use only the first K actions for training**, the rest for test:

```python
# Simulate "user has only K interactions" scenario
for k in [1, 3, 5, 10]:
    train_truncated = group_by_user.head(k)
    test_remaining  = group_by_user.tail(-k)
    # evaluate models under different coldness levels
```

This lets us verify: **the relative advantage of ItemCF / Popular / content-based recall under different K**.

#### Simulation 2: artificially create cold items

Drop all interactions of certain items from training, then check whether the model can still recall them:

```python
# Simulate "new items with zero behavior"
cold_items = sample(all_items, frac=0.1)
train_drop = train[~train["item_id"].isin(cold_items)]
# After training, measure cold-item coverage in recall results
```

This lets us verify: **pure collaborative filtering's "zero recall rate" on cold items**, motivating the need for content-based recall.

#### Simulation 3: title-based content recall

```python
title_emb = sentence_transformer.encode(movies["title"])
# For cold items: use emb to find top-K similar established items → recall their users
# For cold users: give exploration tags → traverse genre inverted index
```

This **actually runs end-to-end**. It is the most complete cold-start demo possible on the current dataset.

### recsys-mini evolution roadmap

| Priority | Change | Notes |
|---|---|---|
| ⭐⭐⭐ | Disable the `filter_eval_users` option | Let cold-start users into evaluation; observe model behavior at 0 / 1 / 5 actions |
| ⭐⭐⭐ | Implement `sentence-transformer + I2I` | Assign embeddings to cold items, do content-based recall |
| ⭐⭐ | Implement coldness-bucketed evaluation | Group users by action-count and compute metrics per bucket |
| ⭐⭐ | Add forced-exposure-budget simulation | Force-insert cold items at re-rank time, measure diversity impact |
| ⭐ | Simple ε-greedy exploration | Add ε probability of random selection at recall output |
| ⭐ | Thompson Sampling baseline | Maintain a Beta distribution per genre |

> **For end-to-end cold-start experiments**, the ultimate recommendation is still to switch to **KuaiRand** (which contains real new users, new items, and `video_emb`).

---

## 10. Further reading

### Classic papers

- **LinUCB (WWW'10)** — foundational Contextual Bandit work for news recommendation, by Yahoo
- **Thompson Sampling for Contextual Bandits (NeurIPS'13)** — theoretical analysis of TS
- **DropoutNet (NeurIPS'17)** — train with dropout to make models robust to cold start
- **MeLU (KDD'19)** — Meta-learning for cold start
- **HEATER (KDD'20)** — auxiliary information for item cold start
- **CB2CF (RecSys'19)** — predict CF embedding from content

### Engineering practice (industry blogs and talks)

- **Netflix Tech Blog** — extensive series on cold start, including "Recommending Items to More Than a Billion People" and posts on contextual bandits in production
- **YouTube** — Covington et al., "Deep Neural Networks for YouTube Recommendations" (RecSys'16) covers fresh-content boosting and freshness features
- **Meta Engineering** — Lookalike Audience whitepapers; "Scaling the Instagram Explore recommendations system" (2023) discusses cold ranking at scale
- **Spotify R&D** — "Personalizing Spotify Home" series; Discover Weekly cold-start design
- **Pinterest Engineering** — PinSage (KDD'18) and "Pinnability" cover image-driven cold start at scale
- **Airbnb** — "Real-time Personalization using Embeddings for Search Ranking" (KDD'18) — the canonical embedding-based listing cold start
- **LinkedIn Engineering** — cold-start handling for job recommendations and feed
- **Snapchat / Snap Inc.** — talks on Spotlight ranking and the creator monetization horse race
- **TikTok / Kuaishou** — public talks on the "traffic ladder" / "horse race" mechanism
- **Xiaohongshu** — RecSys talk on "the cold-start algorithm system"
- **Weibo ML team** — "new-user cold start" series

### Related docs

- [`01-data-source-analysis.md`](01-data-source-analysis.md) — why MovieLens cannot really test cold start

---

## One-line summary

> **Cold start = use these four keys — content understanding + demographic profile + look-alike + exploration — to unlock the door of personalized recommendation when no behavior signal exists**.
>
> It is not a single algorithm; it is **a cross-cutting system that runs through data, recall, ranking, and re-ranking**.
>
> For recsys-mini: "simulation experiments" on MovieLens are possible, but doing real cold start **requires switching datasets**.
