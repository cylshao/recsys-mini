# 过滤 (Filter) + 重排 (Rerank)

> 推荐链路里**最不性感、但最影响产品体感**的两段。算法论文几乎不讲，但**生产环境 30% 的迭代来自这里**。
>
> 过滤管「**不能推什么**」，重排管「**怎么排得既准又不无聊**」。

## 目录

- [一、为什么独立成两段](#一为什么独立成两段)
  - [过滤为什么独立](#过滤为什么独立)
  - [重排为什么独立](#重排为什么独立)
- [二、过滤层 (Filter)](#二过滤层-filter)
  - [七大过滤类别](#七大过滤类别)
  - [优先级 — 去重 > 黑名单 > 频控 > 其他](#优先级--去重--黑名单--频控--其他)
  - [过滤的位置选择](#过滤的位置选择)
- [三、过滤的工程实现](#三过滤的工程实现)
  - [1. 去重 — Bloom Filter](#1-去重--bloom-filter)
  - [2. 黑名单 — Set / Bitmap](#2-黑名单--set--bitmap)
  - [3. 频控 — Sliding Window Counter](#3-频控--sliding-window-counter)
  - [4. 风控 — 双重缓存](#4-风控--双重缓存)
  - [5. 实时性要求](#5-实时性要求)
  - [过滤层的"陷阱"](#过滤层的陷阱)
- [四、重排层 (Rerank) 概述](#四重排层-rerank-概述)
  - [重排的"四件套"](#重排的四件套)
  - [重排的核心张力](#重排的核心张力)
- [五、多样性 — MMR / DPP](#五多样性--mmr--dpp)
  - [为什么需要多样性](#为什么需要多样性)
  - [MMR (Maximal Marginal Relevance)](#mmr-maximal-marginal-relevance)
  - [DPP (Determinantal Point Process)](#dpp-determinantal-point-process)
  - [工程实现要点](#工程实现要点)
- [六、打散 (Spread)](#六打散-spread)
  - [三种打散](#三种打散)
  - [打散的"度"](#打散的度)
- [七、业务规则与强插](#七业务规则与强插)
  - [强插场景](#强插场景)
  - [强插架构](#强插架构)
  - [强插的"度"](#强插的度)
  - [频控与冷却](#频控与冷却)
- [八、Listwise Rerank — PRM / PEAR](#八listwise-rerank--prm--pear)
  - [为什么需要 Listwise](#为什么需要-listwise)
  - [PRM (Personalized Re-ranking Model)](#prm-personalized-re-ranking-model)
  - [PEAR (Pinterest 2022)](#pear-pinterest-2022)
  - [Listwise Rerank 的工程难点](#listwise-rerank-的工程难点)
- [九、生态与平衡](#九生态与平衡)
  - [重排的"超出推荐"职责](#重排的超出推荐职责)
  - [生态目标的量化](#生态目标的量化)
- [十、recsys-mini 的现状 + 改造方案](#十recsys-mini-的现状--改造方案)
  - [现状](#现状)
  - [改造方案](#改造方案)
- [十一、扩展阅读](#十一扩展阅读)
  - [经典论文](#经典论文)
  - [工程](#工程)
  - [相关文档](#相关文档)
- [一句话总结](#一句话总结)

---

## 一、为什么独立成两段

### 过滤为什么独立

**核心理由：硬规则不应污染模型**。

```
反模式: 让排序模型学"不要推已看过的"
        ↓
        模型把 80% 容量浪费在学这个"硬规则"
        ↓
        真正的"软偏好"反而学不好
```

**正确的姿势**：
- **硬规则用 if-else**(过滤层)
- **软偏好用模型**（排序层）
- 两者职责清晰

### 重排为什么独立

**核心理由：精排追求"准"，会丢"多样性"**.

```
精排单点最优:    [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]
                 全是同一作者/类目, 单点 CTR 最高
                 但用户两屏后就会划走

重排修正:        [A1, B1, A2, C1, A3, B2, ...]
                 单点略降, 体验大涨
```

**精排和重排的目标根本不同**：精排是**单点 CTR 最大**，重排是**整页/整 session 留存最大**。

---

## 二、过滤层 (Filter)

### 七大过滤类别

| 类别 | 例子 | 数据来源 |
|---|---|---|
| **去重** | 已看 / 已点过 | 用户曝光历史 (Bloom Filter) |
| **黑名单** | 拉黑作者 / 不感兴趣 tag | 用户偏好 KV |
| **频控** | 同作者 1h 不超过 3 个 | 实时计数器 |
| **业务规则** | 未成年人不看恐怖类 | 用户属性 + 类目映射 |
| **合规风控** | 下架视频 / 风控命中 | 内容审核系统 |
| **库存** | 商品库存 = 0 (电商) | 库存服务 |
| **质量** | 标题党/低俗压制 | 内容质量分 |

### 优先级 — 去重 > 黑名单 > 频控 > 其他

```
请求 ──→ 候选 ─┬─→ 去重(必过)
                ├─→ 黑名单(必过)
                ├─→ 风控(必过)
                ├─→ 频控(可降级)
                ├─→ 业务规则(可降级)
                └─→ 质量阈值(可降级)
                        ↓
                    通过候选
```

**关键设计**：
- **必过 = 一票否决**（过不了直接扔）
- **可降级 = 候选不足时放宽**（避免空结果）

### 过滤的位置选择

| 位置 | 优势 | 劣势 |
|---|---|---|
| 召回前 | 节省后续算力 | 各召回路重复实现，维护难 |
| **召回后**（主流） | 集中维护，可加复杂规则 | 召回浪费一些算力 |
| 排序后 | 处理实时性最高的信号 | 浪费排序算力 |

**主流做法**：**召回后做主过滤，排序后做"动态实时过滤"**(如视频刚被下架).

---

## 三、过滤的工程实现

### 1. 去重 — Bloom Filter

```
用户曝光历史可能 1 千万条
精确存储: 1KW × 8 byte = 80MB / 用户  ❌

Bloom Filter:
  - 1KW 元素, 0.1% 假阳率 → 仅需 ~17MB / 用户
  - O(1) 查询
  - 缺点: 有假阳(可能误过滤一些没看过的)
```

### 2. 黑名单 — Set / Bitmap

```
用户拉黑了 50 个作者
存储: hash set
查询: O(1)
```

### 3. 频控 — Sliding Window Counter

```
key: user_id:author_id
value: [t1, t2, t3, ...]  最近 N 次曝光时间

查询: 过滤 < 1h 前的, count() < 3 ?
存储: Redis ZSET (sorted set)
```

### 4. 风控 — 双重缓存

```
全局风控池: bloom filter + 定时刷新 (10 分钟)
紧急下架: pubsub 实时通知 → 内存 set
```

### 5. 实时性要求

| 信号 | 实时性 | 实现 |
|---|---|---|
| 已看过滤 | 1-5 分钟 | Kafka → Flink → Redis |
| 风控下架 | < 30 秒 | Pub/Sub 主动推送 |
| 频控 | 准实时 | 滑动窗口 |

### 过滤层的"陷阱"

**陷阱 1：候选被过滤光**

```
召回 1000 → 各路过滤 → 剩 5 个    ❌ 排序没法选
```

**解决**：
- 召回阶段就要给冗余（多召回 30~50%）
- 过滤层要监控"过滤后剩余量"，触发降级

**陷阱 2:Bloom Filter 假阳损失**

`false positive rate = 0.1%` 意味着 0.1% 的好物品被误过滤。看似小，但**对长尾物品来说是致命的**（它本来就少有曝光机会）。

**陷阱 3：频控配置随业务变**

- 短视频：同作者 1h 不超过 3 个
- 长视频：同作者 7d 不超过 1 个
- 直播：同主播随时可重复

---

## 四、重排层 (Rerank) 概述

### 重排的"四件套"

```
精排输出 Top-100
       │
       ▼
   ┌─────────────────────────────────┐
   │  ① 多样性 (MMR / DPP)            │
   │     避免同质化                    │
   ├─────────────────────────────────┤
   │  ② 打散 (Spread)                 │
   │     按作者/类目/标签分散          │
   ├─────────────────────────────────┤
   │  ③ 业务规则 / 强插                │
   │     广告位 / 运营内容 / 新作扶持  │
   ├─────────────────────────────────┤
   │  ④ Listwise Rerank (可选)        │
   │     考虑上下文位置的 listwise 模型│
   └─────────────────────────────────┘
       │
       ▼
   最终 Top-N (e.g. 10)
```

### 重排的核心张力

| 维度 | 精排倾向 | 重排修正 |
|---|---|---|
| 单点 CTR | 越高越好 | 牺牲一些 |
| 多样性 | 不关心 | 强制提升 |
| 新颖性 | 不关心 | 留 explore 空间 |
| 业务诉求 | 模型不知道 | 强插实现 |
| 用户长期价值 | 短视 | 兼顾 |

> **重排是"精排的对手"，不是"精排的下游"**.它必须打破精排的局部最优，换全局最优。

---

## 五、多样性 — MMR / DPP

### 为什么需要多样性

```
精排输出: [A、A、A、A、B、A、A、A、C、A]   ← 8 个 A
用户体验: 第 3 个 A 后就开始划走
留存:     ↓
```

> **同类内容连续出现 ≥ 3 次，留存指标显著下降**（各家平台都在公开演讲里印证过）。

### MMR (Maximal Marginal Relevance)

**思路**：每选一个，在"相关性"和"与已选 list 的相似性"之间权衡。

```python
def mmr(candidates, lambda_=0.7):
    selected = []
    while len(selected) < K:
        best = None
        best_score = -inf
        for c in candidates:
            if c in selected:
                continue
            relevance = c.score                                    # 精排分
            sim_to_selected = max(sim(c, s) for s in selected)     # 与已选最相似度
            mmr_score = lambda_ * relevance - (1 - lambda_) * sim_to_selected
            if mmr_score > best_score:
                best, best_score = c, mmr_score
        selected.append(best)
    return selected
```

- **优点**：简单、O(K·N) 复杂度
- **缺点**：贪心，不一定全局最优

### DPP (Determinantal Point Process)

**思路**：用矩阵的行列式来度量"集合的多样性".行列式越大，集合越分散。

```
P(S) ∝ det(L_S)   (S ⊆ candidates)

L_S = 子集 S 对应的核矩阵
    = relevance × similarity 的组合
```

**直觉**：
- 行列式大 = 向量近似线性无关 = 多样性高
- 单个 item 的 relevance 高 + 整体相似度低 → det 大

- **优点**：理论优雅，效果通常 > MMR
- **缺点**：计算复杂(O(N³)，近似算法 O(K·N²))

### 工程实现要点

**相似度怎么算**：
- emb 余弦相似度（最准，但需要 emb）
- 类目重叠（简单）
- 作者相同（强约束）
- tag Jaccard(平衡)

**多样性的"度"怎么调**：
- λ 越大 = 越偏精排分（不多样）
- λ 越小 = 越多样（可能损失 CTR）
- **典型值 λ = 0.5 ~ 0.7**，通过 AB 调

---

## 六、打散 (Spread)

**和多样性的区别**：多样性是"内容差异"，打散是"位置约束".

### 三种打散

#### 1. 同作者打散

```
原: [A1, A2, A3, A4, B1, B2, B3, ...]    ← 4 个 A
后: [A1, B1, A2, B2, A3, B3, A4, ...]    ← 间隔出现
```

**实现** (典型贪心):
```python
def spread_by_author(items, max_consecutive=1, min_gap=3):
    result = []
    last_seen = {}  # author -> last position
    pending = []
    for item in items:
        a = item.author
        if a in last_seen and len(result) - last_seen[a] < min_gap:
            pending.append(item)  # 暂时跳过
        else:
            result.append(item)
            last_seen[a] = len(result) - 1
    # 把 pending 重新插入合适位置
    return result + pending
```

#### 2. 同类目打散

```
策略: 一屏 (前 N 个) 内最多 K 个同类目
配置: 短视频 6 屏内最多 2 个美食
```

#### 3. 标签 / 主题打散

最细粒度的打散，需要 tag 体系完整。

### 打散的"度"

- 太严：候选被过度打散，头部 item 被往后排，损失 CTR
- 太松：同质化严重

> **打散通常 -2% 短期 CTR，但 +3-5% 7天留存**。是典型的"短期换长期".

---

## 七、业务规则与强插

### 强插场景

| 场景 | 例子 |
|---|---|
| 广告位 | 第 4、第 8 强插 |
| 运营内容 | 编辑精选、活动卡 |
| 新作扶持 | 新作者 / 新内容流量倾斜 |
| 关注流 | 关注作者新视频高优 |
| 通知 / 弹窗 | 系统通知插入 |

### 强插架构

```python
def merge_with_rules(rec_list, ad_slots, op_slots, boost_pool):
    """
    rec_list:   重排好的推荐列表 (10 个)
    ad_slots:   广告必须出现的位置 [4, 8]
    op_slots:   运营卡 [(2, op_card_xxx)]
    boost_pool: 新内容池, 强插概率 30%
    """
    result = []
    for pos in range(10):
        if pos in op_positions:
            result.append(op_card)
        elif pos in ad_positions:
            result.append(ads.next())
        elif random.random() < 0.3 and pos > 0 and boost_pool:
            result.append(boost_pool.next())
        else:
            result.append(rec_list.pop(0))
    return result
```

### 强插的"度"

- 广告：占比 5-15%(收入 vs 体验权衡)
- 运营：占比 < 5%
- 新作扶持：占比 5-10%(供给侧健康度)

> **强插的总占比通常 ≤ 25%**.超过这个数体验显著下降。

### 频控与冷却

```
广告频控:
  - 同广告 24h 内最多 3 次
  - 用户关闭广告后,24h 内不再出现
  - 用户付费后,广告占比降至 5%
```

---

## 八、Listwise Rerank — PRM / PEAR

### 为什么需要 Listwise

精排是 **pointwise**：对每个 item 独立打分。

但实际上，**item 的 CTR 受周围 item 影响**：

```
[A、B、C]: A 的 CTR = 5%
[A、X、Y]: A 的 CTR = 7%   ← 因为 X、Y 比 B、C 差,A 反而被衬托
```

### PRM (Personalized Re-ranking Model)

```
输入: [item_1, item_2, ..., item_N]  (精排 top-N)
模型: Transformer Encoder
      ↓
输出: 重排后的顺序

学习目标: 使 list 整体 NDCG 最大化
```

**关键设计**：
- Self-attention 让每个 item 感知整个 list 的上下文
- 加 position embedding 学位置偏置
- 通常作为精排后的"二次排序"

### PEAR (Pinterest 2022)

**核心改进**：同时建模"用户已交互"和"重排候选"，做更长的 attention.

### Listwise Rerank 的工程难点

| 难点 | 说明 |
|---|---|
| **训练样本** | 需要 list-level 标签 (整 list 的留存)，很难拿 |
| **反事实** | "如果重排成不同顺序，用户会怎么反应"无法离线评估 |
| **延迟** | Transformer 推理比 GBDT 重 |
| **稳定性** | 上下文依赖，小改动影响大 |

> **不是所有平台都做 listwise rerank**.多数还是 MMR/DPP + 业务规则。

---

## 九、生态与平衡

### 重排的"超出推荐"职责

| 维度 | 例子 | 工具 |
|---|---|---|
| **新作者扶持** | 新人前 10 条强保底曝光 | 流量阶梯 |
| **长尾内容曝光** | 保留 N% 的位置给冷门 | 配额机制 |
| **创作者多样性** | 不让头部创作者垄断流量 | 作者分布平滑 |
| **用户分群体验** | 新用户多探索，老用户多利用 | 用户画像 + 重排策略 |
| **品类生态** | 防止单品类称霸全平台 | 类目占比约束 |

> **这些不是算法，是产品策略**。但工程实现都在重排层。

### 生态目标的量化

```
平台健康度 = w1 · 新作者活跃率
           + w2 · 长尾内容曝光占比
           + w3 · 品类基尼系数 (反向)
           + w4 · 用户多样性熵
```

每一项都需要重排层主动优化，不能寄希望于精排。

---

## 十、recsys-mini 的现状 + 改造方案

### 现状

| 模块 | 状态 |
|---|---|
| 过滤 | **没有独立模块**，在 `popular.py:30` / `itemcf.py:88` 内部做"已看过滤" |
| 重排 | **完全没有**，直接用 LightGBM 排序后的 Top-N |
| 多样性 | 没有 |
| 打散 | 没有 |
| 业务规则 | 没有 |

### 改造方案

#### Step 1: 抽出过滤层 (P0,半天)

```
src/filter/
├── base.py
└── seen_filter.py   ── 已看过滤
```

```python
# src/filter/base.py
class BaseFilter:
    def filter(self, user_id: int,
               candidates: List[Tuple[int, float]]
               ) -> List[Tuple[int, float]]:
        raise NotImplementedError

class FilterChain:
    def __init__(self, filters: List[BaseFilter]):
        self.filters = filters

    def apply(self, user_id, candidates):
        for f in self.filters:
            candidates = f.filter(user_id, candidates)
        return candidates
```

```python
# src/filter/seen_filter.py
class SeenFilter(BaseFilter):
    def __init__(self, user_history: dict[int, set[int]]):
        self.user_history = user_history

    def filter(self, user_id, candidates):
        seen = self.user_history.get(user_id, set())
        return [(i, s) for i, s in candidates if i not in seen]
```

**收益**：从 ItemCF / Popular 内部去掉"已看过滤"，**职责清晰**，后续可以加更多 filter (黑名单 / 频控).

#### Step 2: 加重排层 — MMR 多样性 (P1,1 天)

```
src/rerank/
├── base.py
├── mmr.py        ── MMR 多样性
└── spread.py     ── 按 genre 打散
```

```python
# src/rerank/mmr.py
def mmr_rerank(scored_items: List[Tuple[int, float]],
               item_genres: dict[int, set[str]],
               top_n: int,
               lambda_: float = 0.7,
               ) -> List[Tuple[int, float]]:
    selected = []
    selected_genres = set()
    candidates = scored_items.copy()

    while len(selected) < top_n and candidates:
        best, best_idx, best_score = None, None, -float("inf")
        for idx, (item, score) in enumerate(candidates):
            genres = item_genres.get(item, set())
            overlap = len(genres & selected_genres) / max(len(genres), 1)
            mmr_score = lambda_ * score - (1 - lambda_) * overlap
            if mmr_score > best_score:
                best, best_idx, best_score = (item, score), idx, mmr_score
        selected.append(best)
        selected_genres |= item_genres.get(best[0], set())
        candidates.pop(best_idx)
    return selected
```

#### Step 3: 端到端评估对比 (P1,0.5 天)

在 `05_offline_eval.py` 加入两个对照：
- 不重排：Top-10 精排直出
- MMR 重排：Top-10 多样性优化

报告四组指标：
- Recall@10 / NDCG@10 (准度)
- ILD@10 (Intra-List Diversity, 多样性)
- Coverage@10 (覆盖率)

**预期**：
- Recall@10 略降（0~3%）
- ILD@10 涨 30%+
- Coverage@10 涨 5-10%

#### Step 4: 加更多 filter (P2,1-2 天)

```
src/filter/
├── seen_filter.py        (已有)
├── blacklist_filter.py   (按 user 黑名单 — 模拟)
├── freq_cap_filter.py    (按 genre 限频 — 模拟)
└── chain.py              (调度)
```

**作用**：**真正体验生产过滤层的复杂度**，理解 if-else 的累积维护成本。

---

## 十一、扩展阅读

### 经典论文

- **MMR (SIGIR'98)** — 多样性的奠基，从 IR 时代就有
- **DPP for Reco (NeurIPS'18)** — `Practical Diversified Recommendations on YouTube with Determinantal Point Processes`
- **PRM (RecSys'19)** — `Personalized Re-ranking for Recommendation`
- **PEAR (CIKM'22)** — Pinterest 的 Listwise rerank
- **Adaptive DPP (KDD'19)** — 多样性参数自适应

### 工程

- 美团技术 "推荐系统重排" 系列
- 阿里 "搜索推荐重排" 公开演讲
- Pinterest 工程博客 "Diversity in Recommendation"

### 相关文档

- [`01-recsys-overview.md`](01-recsys-overview.md) — 过滤/重排在五段式中的位置
- [`05-recall.md`](05-recall.md) — 召回的下游
- [`06-ranking.md`](06-ranking.md) — 重排的上游

---

## 一句话总结

> **过滤 = 把"不能推的"硬剔除，职责简单但坑多；重排 = 把"模型最优"修成"用户体感最优"，是产品策略落地的最后一道闸门**。
>
> 这两段算法含量低、工程含量高、产品含量极高 —— **是真正区分"会做模型"和"懂业务"的分水岭**。
