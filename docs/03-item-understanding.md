# Item Understanding

> **One-liner**: In a recommender system, an **item** is "the thing being recommended". **Item understanding** is the practice of turning that thing into something a downstream system can directly consume — **tags, embeddings, scores, and relations**.
>
> Its quality directly determines three things:
> **whether a brand-new item can be served at all** · **whether your two-tower recall has any signal to learn from** · **whether ranking can cross the chasm of behavioral sparsity**.

> **How to read this doc**: 17 sections, ordered from intuition to engineering. Sections 1–7 are metaphors and concepts; 8–17 are models and architecture. If you only want to "understand what it is", stop after section 7.

## Table of contents

- [1. What is an item?](#1-what-is-an-item)
- [2. What "item" means on different platforms](#2-what-item-means-on-different-platforms)
- [3. An item has four faces](#3-an-item-has-four-faces)
- [4. Why you cannot rely on the ID alone](#4-why-you-cannot-rely-on-the-id-alone)
- [5. Items and users are siblings](#5-items-and-users-are-siblings)
- [6. Five business reasons item understanding matters](#6-five-business-reasons-item-understanding-matters)
- [7. What items look like in this project (recsys-mini)](#7-what-items-look-like-in-this-project-recsys-mini)
- [8. The four output forms](#8-the-four-output-forms)
- [9. Multimodal — decomposing a video](#9-multimodal--decomposing-a-video)
- [10. Structured understanding (tagging, NER, topic, KG)](#10-structured-understanding-tagging-ner-topic-kg)
- [11. Quality and risk assessment](#11-quality-and-risk-assessment)
- [12. Training paradigms](#12-training-paradigms)
- [13. Where outputs are consumed in the five-stage pipeline](#13-where-outputs-are-consumed-in-the-five-stage-pipeline)
- [14. A typical UGC video architecture](#14-a-typical-ugc-video-architecture)
- [15. What is possible in recsys-mini](#15-what-is-possible-in-recsys-mini)
- [16. If you switch datasets](#16-if-you-switch-datasets)
- [17. Further reading](#17-further-reading)
- [TL;DR](#tldr)

---

## 1. What is an item?

The simplest possible metaphor:

> A supermarket has thousands of products on its shelves. Each product is an item. The job of a recommender system is to **pick a handful from a warehouse of millions** and put them on the **small shelf in front of you** — the six tiles on a phone home screen, TikTok's next video, Netflix's hero row, Amazon's "recommended for you".

So:

> **item = the thing being recommended**
>
> **Anything that can be clicked, watched, bought, or consumed is an item.**

It does not have to be a product.

---

## 2. What "item" means on different platforms


| Platform                        | What an item is                    |
| ------------------------------- | ---------------------------------- |
| TikTok / Reels / Shorts         | A short video                      |
| YouTube                         | A video                            |
| Netflix / Disney+               | A movie or series                  |
| Spotify / Apple Music           | A track, album, or podcast episode |
| Amazon / Shopee / Mercado Libre | A SKU                              |
| Pinterest                       | A Pin (image)                      |
| Instagram / Xiaohongshu         | A post or Reel                     |
| Meituan / DoorDash              | A restaurant or dish               |
| Uber / Lyft                     | A car or driver                    |
| Twitter / X                     | A tweet                            |
| LinkedIn                        | A job posting or feed post         |
| Booking / Airbnb                | A room or listing                  |
| Substack / Medium               | An article                         |
| **This project (MovieLens-1M)** | **A movie**                        |


> **Takeaway**: "Building a recommender" does not mean "recommending products in an e-commerce store". **If your scenario involves picking a few things to show out of many possibilities, you are doing recommendation, and those things are items**.

---

## 3. An item has four faces

The most common rookie mistake is to think `item_id = 2858` is the whole story. In reality, an item has four facets:

```
┌─────────────────────────────────────────────────────┐
│                    an item                          │
├─────────────────────────────────────────────────────┤
│  ① ID          : item_id = 2858                     │
│  ② Content     : title, poster, plot, cast, length  │
│  ③ Statistics  : 3,428 views, average rating 4.32   │
│  ④ Time        : released 1999, ingested, decay     │
└─────────────────────────────────────────────────────┘
```

Mapped onto MovieLens's most popular movie, *American Beauty*:


| Face | Meaning    | For this movie                                                             |
| ---- | ---------- | -------------------------------------------------------------------------- |
| ①    | ID         | `item_id = 2858`                                                           |
| ②    | Content    | title `American Beauty (1999)`, genre `Drama                               |
| ③    | Statistics | 3,428 ratings (most in the dataset), average rating 4.32                   |
| ④    | Time       | released 1999; the dataset was collected in 2000, so this is a "fresh hit" |


> **Takeaway**: item understanding is essentially about **converting the latter three faces (content / statistics / time) into something a model can eat**.

---

## 4. Why you cannot rely on the ID alone

> **One line**: **ID alone works for old items. For new items, it is blind.**

A concrete example:

> A creator uploads a brand-new video and the database assigns it `item_id = 99999999`.
>
> The system looks at this number and thinks: **"Never seen. Never clicked. Looks nothing like any past item_id."** → It cannot be served.
>
> But —
>
> - The title says "5-minute Tiramisu recipe"
> - The thumbnail shows a cake
> - The creator has 500,000 baking fans
> - It is tagged "Dessert"
>
> **These content signals tell the system: it looks like every dessert tutorial we already have**. Now it can be served to baking enthusiasts.

That is the entire purpose of **item understanding**:

> **Letting a brand-new item find its lookalikes with zero behavior**.

The standard playbook (used by every major platform):


| Raw signal           | How it becomes an embedding                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| Title text           | BERT / RoBERTa / LLM encoder                                                                         |
| Thumbnail / cover    | CLIP / ViT                                                                                           |
| Video content        | Frame sampling + video model (SlowFast / VideoSwin / V-JEPA)                                         |
| Audio / BGM          | Wav2Vec / AudioCLIP / Whisper (ASR)                                                                  |
| Creator              | The creator's follower base (see `[02-cold-start.md](02-cold-start.md)` §4 author-based propagation) |
| Category / tag       | One-hot or learned tag embedding                                                                     |
| Knowledge-graph node | Graph embedding (TransE / GraphSAGE)                                                                 |


These embeddings feed two-tower recall and ranking, so **the new item finds the right audience on its very first impression**.

---

## 5. Items and users are siblings

Recommendation is matchmaking — both sides need a profile. Note the **symmetry**:

```
        User side                       Item side
        ─────────                       ─────────
        age / gender / city             category / tag / duration
        watch history       ─ match ─→  audience history
        interest embedding              content embedding
        activity / retention            popularity / CTR / completion rate
```

Internally, most large platforms treat **User Profile** and **Item Profile** as a single concern:

- Same schema
- Same feature-engineering team
- Same storage (KV / feature store)
- Often a shared embedding space — that is the whole point of two-tower architectures

> **Takeaway**: When learning item understanding, **always think of it in parallel with user understanding**. The mental model becomes much cleaner.

---

## 6. Five business reasons item understanding matters


| #   | Reason                     | Why it needs item understanding                                                                                  |
| --- | -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1   | **New-item cold start**    | Zero behavior; only "content" can identify the item (see `[02-cold-start.md](02-cold-start.md)`)                 |
| 2   | **Long-tail distribution** | 90% of a million-item catalog has fewer than 100 interactions; CF cannot learn them, only content can            |
| 3   | **Multi-route recall**     | Beyond CF you need "users who watched tag A also watched tag B", "fans of this director also like that director" |
| 4   | **Explainability**         | "Because you watched *Inception*" beats "because your user_id is similar to a cluster"                           |
| 5   | **Trust & safety**         | Content understanding is also the gatekeeper for "is this NSFW / violent / copyright-violating?"                 |


Plus a sixth, often-overlooked one:


| #   | Reason                                        | Why it needs item understanding                                                                                                                                 |
| --- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6   | **Cross-platform / cross-language expansion** | Content embeddings are language- and platform-agnostic, so they transfer across markets (Netflix launching in India, Spotify in Latin America all rely on this) |


---

## 7. What items look like in this project (recsys-mini)

```
movies.dat           ← the MovieLens-1M item table
─────────
3,883 movies = 3,883 items

Per-item fields currently available:
  - item_id   (integer)
  - title     ("American Beauty (1999)")
  - year      (regex-extracted from title)
  - genres    ("Drama|Romance")
```

Reality check: **on the item side, MovieLens has almost nothing beyond metadata**.


| Dimension               | Available?                                    |
| ----------------------- | --------------------------------------------- |
| ID                      | Yes                                           |
| Title text              | Yes (very short — 3–4 words on average)       |
| Release year            | Yes (regex from title)                        |
| Genre                   | Yes (18 categories, multi-hot)                |
| Poster / still image    | No                                            |
| Plot synopsis           | No                                            |
| Cast / director         | No                                            |
| Trailer video           | No                                            |
| User tags / review text | No (ML-1M does not include them; ML-25M does) |


→ **This dataset cannot demonstrate real multimodal content understanding** (see `[01-data-source-analysis.md](01-data-source-analysis.md)` for full limitations). But the things you *can* do are **more than you'd expect** — see section [15](#15-what-is-possible-in-recsys-mini).

> **Dividing line**: from here on the doc shifts from **concepts and intuition** to **models, data structures, and architecture**. If you only wanted to understand "what it is", you can stop now. If you intend to build it, keep reading.

---

## 8. The four output forms

The output of item understanding is **not just "an embedding"**. There are four forms that coexist in production:


| Form                     | Example                                     | Storage                            | Main consumer                         |
| ------------------------ | ------------------------------------------- | ---------------------------------- | ------------------------------------- |
| **Discrete tag**         | "pets-cat-cute", "food-baking-cake"         | Inverted index                     | Tag recall, filtering, explainability |
| **Continuous embedding** | 128–768 dim float32                         | ANN index (FAISS / ScaNN / Milvus) | Vector recall, ranking features       |
| **Quality / risk score** | `quality=0.87, clickbait=0.12`              | KV (`item_id → float`)             | Ranking weights, filter fallback      |
| **Relation**             | "top-100 similar items", "author → fanbase" | Graph / inverted index             | I2I recall, diversification           |


### Why all four (instead of just embeddings)?


| Form      | Pros                                                          | Cons                                                        |
| --------- | ------------------------------------------------------------- | ----------------------------------------------------------- |
| Tag       | Interpretable, controllable, instant generation for new items | Limited expressiveness; missed labels are a real problem    |
| Embedding | Highly expressive, captures similarity automatically          | Black box; cold items need to be encoded first; memory cost |
| Score     | One scalar; trivial to integrate; rule-friendly               | Low information content                                     |
| Relation  | Directly indexable, lowest latency                            | Maintenance overhead (N items → N² relations)               |


> **Takeaway**: **production systems run all four in parallel**, each handling what it does best.

---

## 9. Multimodal — decomposing a video

UGC video understanding is **three parallel pathways plus one fusion step**:

```
┌─ Text pathway ──────────────────────────────────┐
│                                                 │
│  title    ┐                                     │
│  desc     ┼─→  BERT / RoBERTa / ERNIE  ─→ text_emb │
│  danmaku  ┘                            (768d)   │
│  captions  →   tokenize + keywords ─→ keywords  │
└────────────────────────────────────────────────┘

┌─ Vision pathway ────────────────────────────────┐
│                                                 │
│  cover image ──→ ResNet / CLIP-ViT ─→ cover_emb │
│                                       (512d)    │
│  video    ──→ frame sampling (1/sec) ┐          │
│                                       │          │
│                                       ▼          │
│         SlowFast / VideoSwin / V-JEPA            │
│                                       │          │
│                                       ▼          │
│                    video_emb (1024d)             │
└────────────────────────────────────────────────┘

┌─ Audio pathway ─────────────────────────────────┐
│                                                 │
│  BGM    ──→ AudioCLIP / Wav2Vec ─→ audio_emb    │
│  speech ──→ ASR (Whisper)       ─→ text → above │
└────────────────────────────────────────────────┘

                         ▼
              ┌─────────────────────┐
              │ Multimodal fusion   │
              │ (CLIP / ALBEF /     │
              │  BLIP / X-VLM)      │
              └─────────────────────┘
                         │
                  multimodal_emb (256–512d)
                         │
                         ▼
              Write to item feature store + ANN index
```

### Models at a glance


| Modality | Classic                | Modern (2023+)                                                  |
| -------- | ---------------------- | --------------------------------------------------------------- |
| Text     | Word2Vec / fastText    | BERT / RoBERTa / ERNIE / Sentence-Transformers / LLM-as-encoder |
| Image    | ResNet / EfficientNet  | CLIP / ViT / DINOv2 / SAM                                       |
| Video    | C3D / I3D              | SlowFast / VideoSwin / V-JEPA / VideoMAE                        |
| Audio    | MFCC / Mel-spectrogram | AudioCLIP / Wav2Vec2 / Whisper (ASR)                            |
| Fusion   | Concat + MLP           | CLIP / ALBEF / BLIP / X-VLM / LLaVA                             |


### Why fuse modalities instead of using one?

```
Cover only :  "person in white coat"     → could be a doctor / chemist / chef / actor
Title only :  "today I'll show you how"  → too generic
BGM only   :  "calm piano"               → shared by thousands of videos

All three  :  "chemistry experiment tutorial"  → precise
```

> **Takeaway**: the cross-modality balance of **complementarity vs. redundancy** is where fusion models pay off.

---

## 10. Structured understanding (tagging, NER, topic, KG)

Not every kind of item understanding needs deep learning. **For many cases, structured methods have the better ROI**.

### Tagging

Map content onto a predefined taxonomy:

```
content ──→ multimodal model ──→ "food / cooking / home-cooked / recipe / tomato-eggs"
                                      │
                                      ▼
                              Inverted index (tag → item_id list)
                                      │
                                      ▼
                              User interest tag → inverted lookup
```

**Key design points**:

- **A taxonomy is a product decision, not an algorithmic one** (1,000 vs. 10,000 tags; how fine to go)
- **Hierarchical tags** (top-level / sub-level / leaf) all serve different purposes
- **Tag confidence is critical** — low-confidence tags pollute the inverted index

### Named Entity Recognition (NER)

Identify people, places, things, brands in the content:

```
"Just had hotpot at Haidilao in Sanlitun"  → location: Sanlitun, brand: Haidilao
"Just landed in NYC, going to Joe's Pizza" → location: NYC, brand: Joe's Pizza
```

Powers "check-in content" and "brand content" search and recall.

### Topic classification

Coarser content topics — typically **LDA / BERTopic / CLIP-text clustering**. Used for trending topics, new-topic incubation, and hotspot tracking.

### Knowledge-graph anchoring

Attach the item to KG nodes:

```
"Marvel new trailer" → KG: [Movie] → [Marvel] → [Avengers series]
```

Powers "series" and "fan-oriented" recommendations.

---

## 11. Quality and risk assessment

The other side of item understanding is **"is this safe to recommend?"**


| Category                | Signal                             | Model                      |
| ----------------------- | ---------------------------------- | -------------------------- |
| **Base quality**        | Resolution, bitrate, composition   | CV quality model           |
| **Clickbait**           | Title-vs-content mismatch          | Text-video alignment model |
| **NSFW / vulgar**       | Adult / violent / vulgar content   | Multimodal classifier      |
| **Compliance**          | Political / sensitive / copyright  | Keyword + model ensemble   |
| **Cold-start CTR**      | Predict CTR before any behavior    | Content-only CTR model     |
| **In-category quality** | Relative quality within a category | LTR (LambdaRank)           |


**Two uses for the quality score**:

1. **Filter**: drop items below a threshold pre-recall
2. **Weight**: in the ranking formula, `score *= quality_score`

---

## 12. Training paradigms

Where do these models come from? Three mainstream routes, **typically stacked together in production**.

### Self-supervised pretraining


|          |                                                                                                  |
| -------- | ------------------------------------------------------------------------------------------------ |
| Examples | CLIP (image-text contrastive), SimCSE (sentence contrastive), MAE (image reconstruction), V-JEPA |
| Idea     | No labels needed; learn general embeddings from massive unlabeled data                           |
| Pros     | Doesn't depend on business labels; transferable                                                  |
| Cons     | Gap with downstream tasks                                                                        |


### Multitask supervised


|          |                                                                     |
| -------- | ------------------------------------------------------------------- |
| Examples | Predict [tag + category + quality + CTR] simultaneously             |
| Idea     | Treat every business task as supervision, sharing a common backbone |
| Pros     | Embeddings align directly with business                             |
| Cons     | A label-poor task can drag the whole model down                     |


### Behavior-aligned


|          |                                                                                  |
| -------- | -------------------------------------------------------------------------------- |
| Examples | Train item embeddings so that "items consumed by the same user" cluster together |
| Idea     | This is the training objective of **Item2Vec / SASRec / Two-Tower**              |
| Pros     | Embeddings are natively friendly to the recommendation task                      |
| Cons     | Cold items still cannot be learned (no behavior to align on)                     |


> **In practice**: stack all three. Self-supervised pretraining produces a general embedding; multitask + behavior-aligned fine-tune it for the business.

---

## 13. Where outputs are consumed in the five-stage pipeline

Item-understanding outputs appear throughout the data → recall → filter → ranking → re-rank pipeline:

```
┌──────── Data layer ────────┐
│  raw item → understanding   │
│            → 4 outputs      │
└───────────────┬─────────────┘
                │
   ┌────────────┼─────────────┐
   ▼            ▼             ▼
┌ Recall ┐  ┌ Filter ┐    ┌ Rank ┐
│        │  │        │    │      │
│ I2I    │  │ T&S    │    │ emb  │
│ U2I    │  │ qual.  │    │ feat │
│ tag    │  │ rate   │    │      │
│ cold   │  │ block  │    │      │
└────────┘  └────────┘    └──┬───┘
                              ▼
                         ┌ Rerank ┐
                         │ div.   │
                         │ scatter│
                         │ (use   │
                         │  emb)  │
                         └────────┘
```

**Typical uses**:


| Output            | Recall                                                | Ranking                                               | Re-rank                    |
| ----------------- | ----------------------------------------------------- | ----------------------------------------------------- | -------------------------- |
| **Embedding**     | Two-tower U2I / I2I neighbors / cold-start look-alike | Concat into MLP, compute user-item similarity feature | DPP / MMR for diversity    |
| **Tag**           | Tag inverted recall                                   | Categorical + cross                                   | Same-tag scattering        |
| **Quality score** | Drop low-quality                                      | Ranking weight                                        | Force-promote high-quality |
| **Relation**      | Direct I2I top-K                                      | Relation features (same author?)                      | Same-author scattering     |


---

## 14. A typical UGC video architecture

```
                  ┌─────────────────────────────┐
new video upload  │ Upload + transcode + frames │
        ───────→  └──────────────┬──────────────┘
                                 │
                ┌────────────────┼────────────────┐
                ▼                ▼                ▼
        ┌──────────────┐  ┌──────────────┐ ┌──────────────┐
        │ Vision       │  │ Text         │ │ Audio        │
        │ ResNet/CLIP  │  │ BERT/ERNIE   │ │ Wav2Vec/ASR  │
        └──────┬───────┘  └──────┬───────┘ └──────┬───────┘
               │                 │                │
               └─────────────────┼────────────────┘
                                 ▼
                      ┌─────────────────────┐
                      │ Multimodal fusion   │
                      │ (CLIP)              │
                      └──────────┬──────────┘
                                 │
            ┌────────────────────┼─────────────────────┐
            ▼                    ▼                     ▼
      ┌──────────┐        ┌──────────┐          ┌──────────┐
      │ Tag pred │        │ T&S      │          │ Quality  │
      │ (tag)    │        │ (drop/   │          │ (CTR /   │
      │          │        │  flag)   │          │  score)  │
      └────┬─────┘        └────┬─────┘          └────┬─────┘
           │                   │                     │
           └────────┬──────────┴────────┬────────────┘
                    ▼                   ▼
          ┌─────────────────┐  ┌─────────────────┐
          │  Item feature   │  │  ANN vector     │
          │  store          │  │  index          │
          │  (Redis/HBase)  │  │ (FAISS/Milvus)  │
          └────────┬────────┘  └────────┬────────┘
                   │                    │
                   └─────────┬──────────┘
                             ▼
                Consumed by main recommendation path
                (recall / filter / ranking / re-rank)
```

**Engineering SLAs**:


| Stage                                  | Target        |
| -------------------------------------- | ------------- |
| Upload → indexed (incl. understanding) | < 5 min (p95) |
| Single embedding inference             | < 1 s         |
| Online feature lookup                  | p99 < 5 ms    |
| ANN retrieval (billion-scale)          | p99 < 20 ms   |


---

## 15. What is possible in recsys-mini

### Current state

The dataset only has:

```
movies.dat:  item_id  |  title                                  |  genres
              5       | "Father of the Bride Part II (1995)"   | "Comedy"
```

**Missing**: posters, trailers, plot summaries, cast lists, reviews, captions.

So **multimodal is completely off the table**. But pure structured / shallow-NLP work has surprisingly broad scope.

### Doable today (1–2 hours, no new deps)


| #   | Task                           | How                                                           | Payoff                        |
| --- | ------------------------------ | ------------------------------------------------------------- | ----------------------------- |
| 1   | Extract year from title        | regex `\((\d{4})\)` (already done in `01_movielens1m_eda.py`) | Decade feature for ranking    |
| 2   | Item age                       | `request_ts - first_seen_ts`                                  | Old- vs. new-movie preference |
| 3   | Genre embedding                | 18-dim multi-hot → 8-dim dense (PMI / SVD)                    | Friendlier for ranking models |
| 4   | Behavioral Item2Vec            | Treat each user's history as a "sentence", train Word2Vec     | Embeddings for cold items too |
| 5   | Quality score = average rating | `item_avg_rating`                                             | Direct ranking weight         |
| 6   | Item popularity bucketing      | Bucket by `inter_count` into 5 bins                           | Long-tail-aware feature       |


### Medium effort (0.5–1 day, add sentence-transformers)


| #   | Task                   | How                                                     | Payoff                                           |
| --- | ---------------------- | ------------------------------------------------------- | ------------------------------------------------ |
| 7   | Title NLP embedding    | `sentence-transformers/all-MiniLM-L6-v2` encodes titles | Content-based recall + cold items get embeddings |
| 8   | Embedding-based I2I    | Encode all titles, cosine top-100 into index            | Content recall route                             |
| 9   | Genre text + embedding | "Comedy Romance" as a short text → embedding            | More expressive than multi-hot                   |


### Out of reach (cannot do on ML-1M, must switch dataset)


| Missing capability      | Missing data                            |
| ----------------------- | --------------------------------------- |
| Poster CV embedding     | No images                               |
| Trailer video embedding | No videos                               |
| Plot semantics          | No synopses                             |
| Review sentiment        | ML-1M has no reviews                    |
| User tags (folksonomy)  | ML-1M doesn't include tags; ML-25M does |


> **Conclusion**: on ML-1M the ceiling for item understanding is **"title NLP embedding + genre embedding + derived statistical features"**. Beyond that, you must switch dataset or self-scrape IMDB / TMDB extensions.

---

## 16. If you switch datasets

These datasets let item understanding go from "almost nothing to learn" to "the full multimodal stack":


| Dataset                           | What it offers                                                                                    | Best for                                                         |
| --------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **MovieLens-25M + Tags Genome**   | User-supplied tag relevance matrix                                                                | Tag semantics, tag-based recall                                  |
| **KuaiRand / KuaiSAR** (Kuaishou) | `video_duration`, `tag`, `author_id`, `music_id`, `**video_emb` (64-dim, platform-pretrained)** ⭐ | Real video recsys with the **CV training cost absorbed for you** |
| **MIND** (Microsoft News)         | News title + body + category                                                                      | Text-heavy multimodal                                            |
| **Amazon Reviews 2023**           | Product description + image + review text                                                         | E-commerce multimodal                                            |
| **Yelp Open Dataset**             | Business description + reviews + geo                                                              | LBS recommendation                                               |
| **H&M Personalized Fashion**      | Product image + text + color / style attributes                                                   | Fashion multimodal                                               |
| **Spotify Million Playlist**      | Track metadata + playlist co-occurrence                                                           | Audio / sequence recommendation                                  |


**Why KuaiRand is special** — it ships with `video_emb` directly. That means:

- **Vector recall works out of the box** — no need to train a CV model yourself
- **Cold items have embeddings** — solves half the cold-start problem
- **Embeddings double as ranking features** — U-I and I-I similarity

That is why `[README.md](../README.md)` lists "switch to KuaiRand" as a long-term roadmap item — **the foundation of item understanding is provided**, leaving you free to focus on recall, ranking, and re-rank.

---

## 17. Further reading

### Classic papers


| Paper                         | Venue        | Why it matters                                          |
| ----------------------------- | ------------ | ------------------------------------------------------- |
| **CLIP**                      | ICML'21      | The starting point of all modern image-text embeddings  |
| **Item2Vec**                  | ML4Sigmod'16 | Foundational behavior-aligned embedding                 |
| **YouTube DNN**               | RecSys'16    | First to use content embeddings as two-tower input      |
| **PinSage**                   | KDD'18       | Pinterest's image-driven I2I                            |
| **Airbnb Listing Embeddings** | KDD'18       | The canonical "session co-occurrence" listing embedding |
| **MIND**                      | CIKM'19      | ByteDance multimodal recommendation                     |
| **Q2L**                       | NeurIPS'21   | Video multi-label understanding                         |
| **V-JEPA**                    | Meta, 2024   | Video self-supervised, latest generation                |


### Engineering posts

- **Netflix Tech Blog** — "Recommending What Video to Watch Next" and the CTR / item-embedding series
- **Spotify R&D** — Discover Weekly's item-embedding journey
- **Meta Engineering** — Instagram Explore / Reels multimodal understanding
- **Pinterest Engineering** — PinSage / SearchSage / "Pinnability"
- **ByteDance** — short-video multimodal practice
- **Kuaishou Big Data** — short-video content-understanding pipeline
- **Alibaba Taobao** — product image / title understanding (e-commerce analogue)

### Datasets

- **KuaiRand / KuaiSAR** — Kuaishou; ships with `video_emb`
- **MIND** (Microsoft News) — news recommendation with text
- **MovieLens-25M + Tags Genome** — extended MovieLens with tag-relevance matrix
- **Amazon Reviews 2023** — e-commerce multimodal
- **H&M Personalized Fashion** — fashion multimodal

### Related docs

- `[01-data-source-analysis.md](01-data-source-analysis.md)` — why MovieLens-1M has so little to offer on the item side
- `[02-cold-start.md](02-cold-start.md)` — item understanding is the lifeline of cold start

---

## TL;DR

> **Item = the thing being recommended**. It has four faces: **ID, content, statistics, time**.
>
> ID alone serves old items. Feeding "content" into the model is what lets a brand-new item find the right audience on its very first impression. That is the entire problem item understanding solves.
>
> In engineering terms: the **outputs** are tags + embeddings + scores + relations (four forms coexisting); the **inputs** are text + image + video + audio + behavior statistics (multimodal fusion); the **consumers** are recall + ranking + re-rank + filtering + cold start (every stage of the pipeline).

