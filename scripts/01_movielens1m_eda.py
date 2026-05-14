from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter

import pandas as pd

from src.config import load_config
from src.data.loader import load_movielens_1m
from src.utils.io import get_logger


def hr(title: str = "", char: str = "─", width: int = 72) -> None:
    if title:
        print(f"\n{char * 3} {title} {char * (width - len(title) - 5)}")
    else:
        print(char * width)


def main() -> None:
    cfg = load_config()
    log = get_logger("eda")
    raw_dir = cfg.path("raw_dir")

    ratings, movies, users = load_movielens_1m(raw_dir)
    log.info(f"loaded: ratings={len(ratings):,}  movies={len(movies):,}  users={len(users):,}")

    # ========================================================
    # 1. Rating distribution
    # ========================================================
    hr("1. Rating distribution")
    rcnt = ratings["rating"].value_counts().sort_index()
    total = len(ratings)
    for r, c in rcnt.items():
        bar = "█" * int(c / total * 50)
        print(f"  rating={r}  {c:>9,}  ({c/total*100:5.2f}%)  {bar}")
    print(f"  positive (>=4): {(ratings['rating']>=4).sum():,} ({(ratings['rating']>=4).mean()*100:.2f}%)")
    print(f"  negative (<=2): {(ratings['rating']<=2).sum():,} ({(ratings['rating']<=2).mean()*100:.2f}%)")

    # ========================================================
    # 2. Time distribution
    # ========================================================
    hr("2. Time distribution")
    ts = ratings["ts"]
    print(f"  Time span: {ts.min()}  ~  {ts.max()}  ({(ts.max()-ts.min()).days} days)")
    monthly = ratings.set_index("ts").resample("MS").size()
    print(f"  Monthly counts (first 5 / last 5):")
    for d, c in monthly.head(5).items():
        print(f"    {d.date()}  {c:>9,}")
    print("    ...")
    for d, c in monthly.tail(5).items():
        print(f"    {d.date()}  {c:>9,}")
    print(f"  Hottest month: {monthly.idxmax().date()} → {monthly.max():,} interactions")
    print(f"  Coldest month: {monthly.idxmin().date()} → {monthly.min():,} interactions")

    last7 = ratings[ts >= ts.max() - pd.Timedelta(days=7)]
    last30 = ratings[ts >= ts.max() - pd.Timedelta(days=30)]
    print(f"  Last 7 days:  {len(last7):>9,} interactions  ({len(last7)/total*100:.4f}%)")
    print(f"  Last 30 days: {len(last30):>9,} interactions  ({len(last30)/total*100:.4f}%)")
    print(f"  → Data is heavily concentrated in the early period; global time-based split is not viable!")

    # ========================================================
    # 3. User behavior length distribution
    # ========================================================
    hr("3. User behavior length distribution (user_id -> n_inter)")
    user_n = ratings.groupby("user_id").size()
    pct = user_n.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).astype(int)
    print(f"  Number of users: {len(user_n):,}")
    print(f"  Quantiles:")
    for q, v in pct.items():
        print(f"    p{int(q*100):>2}: {v:>5,} times")
    print(f"  Min: {user_n.min():,}    Max: {user_n.max():,}    Mean: {user_n.mean():.1f}")
    print(f"  Users with >=5 interactions: {(user_n>=5).sum():,} ({(user_n>=5).mean()*100:.1f}%)")

    # ========================================================
    # 4. Item popularity distribution (long tail)
    # ========================================================
    hr("4. Item popularity distribution (item_id -> n_inter)")
    item_n = ratings.groupby("item_id").size().sort_values(ascending=False)
    pct = item_n.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).astype(int)
    print(f"  Number of items: {len(item_n):,}")
    print(f"  Quantiles:")
    for q, v in pct.items():
        print(f"    p{int(q*100):>2}: {v:>5,} times")
    print(f"  Min: {item_n.min():,}    Max: {item_n.max():,}    Mean: {item_n.mean():.1f}")

    cumsum = item_n.cumsum() / item_n.sum()
    p20_items = (cumsum <= 0.5).sum() + 1
    print(f"  Top 20% hot items ({int(0.2*len(item_n))} items) account for: {item_n.head(int(0.2*len(item_n))).sum()/item_n.sum()*100:.1f}% of exposure")
    print(f"  Only {p20_items} items ({p20_items/len(item_n)*100:.1f}%) needed to cover 50% of exposure")
    print(f"  Cold items with only 1-2 interactions: {(item_n<=2).sum():,} ({(item_n<=2).mean()*100:.1f}%)")

    # ========================================================
    # 5. Matrix sparsity
    # ========================================================
    hr("5. user-item matrix sparsity")
    n_user = ratings["user_id"].nunique()
    n_item = ratings["item_id"].nunique()
    n_pair = len(ratings)
    density = n_pair / (n_user * n_item)
    print(f"  Matrix shape: {n_user:,} × {n_item:,} = {n_user*n_item:,} cells")
    print(f"  Filled cells: {n_pair:,}")
    print(f"  Density: {density*100:.4f}%")
    print(f"  Sparsity: {(1-density)*100:.4f}%")
    print(f"  Compare: real UGC platforms have density < 0.001%; MovieLens is dense by comparison!")

    # ========================================================
    # 6. User profile distribution
    # ========================================================
    hr("6. User static profile")
    print(f"  Gender:")
    for g, c in users["gender"].value_counts().items():
        print(f"    {g}: {c:>5,} ({c/len(users)*100:.1f}%)")
    print(f"  Age (MovieLens discretized buckets):")
    age_map = {1: "<18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
    for a, c in users["age"].value_counts().sort_index().items():
        label = age_map.get(a, str(a))
        bar = "█" * int(c / len(users) * 40)
        print(f"    {label:<8} ({a:>2}): {c:>5,} ({c/len(users)*100:5.1f}%)  {bar}")
    print(f"  Occupation: {users['occupation'].nunique()} categories, top 5:")
    occ_map = {0:"other", 1:"academic", 2:"artist", 3:"clerical", 4:"college", 5:"customer",
               6:"doctor", 7:"executive", 8:"farmer", 9:"homemaker", 10:"K-12 student",
               11:"lawyer", 12:"programmer", 13:"retired", 14:"sales", 15:"scientist",
               16:"self-employed", 17:"technician", 18:"tradesman", 19:"unemployed", 20:"writer"}
    for o, c in users["occupation"].value_counts().head(5).items():
        print(f"    {o:>2} ({occ_map.get(o,'?'):<14}): {c:>5,} ({c/len(users)*100:.1f}%)")

    # ========================================================
    # 7. Item (movie) metadata
    # ========================================================
    hr("7. Item (movie) metadata")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    print(f"  Year distribution:")
    decade = (movies["year"] // 10 * 10).value_counts().sort_index()
    for d, c in decade.items():
        if pd.notna(d):
            bar = "█" * int(c / len(movies) * 40)
            print(f"    {int(d)}s: {c:>5,} ({c/len(movies)*100:5.1f}%)  {bar}")

    print(f"\n  Genre distribution:")
    all_g = Counter()
    for g in movies["genres"]:
        all_g.update(g.split("|"))
    avg_g = movies["genres"].str.split("|").str.len().mean()
    print(f"  {len(all_g)} genres total, average {avg_g:.2f} genres per item")
    print(f"  Top 10:")
    for g, c in all_g.most_common(10):
        bar = "█" * int(c / len(movies) * 30)
        print(f"    {g:<14}: {c:>5,} ({c/len(movies)*100:5.1f}%)  {bar}")

    # ========================================================
    # 8. Signal richness: what is this dataset missing
    # ========================================================
    hr("8. Signal richness comparison (vs UGC video platforms)")
    available = {
        "user_id + item_id + timestamp": True,
        "explicit positive feedback (high rating)": True,
        "explicit negative feedback (low rating)": True,    # 1-2 stars can be treated as weak negative
        "user static profile": True,
        "item category": True,
        "impression log": False,                             # only ratings, no impressions
        "dwell time / play completion": False,
        "like / comment / share": False,
        "follow relationship (social graph)": False,
        "context (device / time-of-day / network)": False,
        "item raw text / cover / video file": False,
        "item author / creator": False,
        "real-time feedback stream": False,
    }
    for k, v in available.items():
        print(f"  [{'✓' if v else '✗'}] {k}")
    n_have = sum(available.values())
    print(f"\n  Has {n_have}/{len(available)} core signal types → main gaps are 'behavior diversity + content understanding + context'")


if __name__ == "__main__":
    main()
