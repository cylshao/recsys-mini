"""MovieLens-1M exploratory data analysis (EDA).

Read-only analysis, nothing is written to disk. Used to understand the data
before modeling:
  1) rating distribution   2) time distribution   3) user activity length
  4) item long tail        5) matrix sparsity      6) user profile
  7) item metadata         8) signal richness

All thresholds (positive/negative feedback, k-core) are read from
conf/config.yaml to stay consistent with downstream scripts.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import Config, load_config
from src.data.loader import load_movielens_1m
from src.utils.io import get_logger

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
WEAK_NEGATIVE_MAX = 2  # rating <= 2 treated as weak negative (mirrors positive_threshold)

AGE_MAP = {1: "<18", 18: "18-24", 25: "25-34", 35: "35-44",
           45: "45-49", 50: "50-55", 56: "56+"}
OCC_MAP = {0: "other", 1: "academic", 2: "artist", 3: "clerical", 4: "college",
           5: "customer", 6: "doctor", 7: "executive", 8: "farmer", 9: "homemaker",
           10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired",
           14: "sales", 15: "scientist", 16: "self-employed", 17: "technician",
           18: "tradesman", 19: "unemployed", 20: "writer"}


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------
def section(title: str, char: str = "─", width: int = 72) -> None:
    """Print a titled section divider."""
    if not title:
        print(char * width)
        return
    fill = max(0, width - len(title) - 5)
    print(f"\n{char * 3} {title} {char * fill}")


def bar(frac: float, width: int = 50) -> str:
    """Render a fraction in [0, 1] as a fixed-width bar chart."""
    frac = min(max(frac, 0.0), 1.0)
    return "█" * int(frac * width)


def pct(count: float, total: float) -> float:
    """Safe percentage (returns 0 when total is 0)."""
    return 100.0 * count / total if total else 0.0


def print_quantiles(counts: pd.Series, probs: list[float] = QUANTILES) -> None:
    """Print quantiles plus min/max/mean of a count Series."""
    q = counts.quantile(probs).astype(int)
    print("  Quantiles:")
    for p, v in q.items():
        print(f"    p{int(round(p * 100)):>2}: {v:>6,}")
    print(f"  Min: {int(counts.min()):,}    "
          f"Max: {int(counts.max()):,}    "
          f"Mean: {counts.mean():.1f}")


# ----------------------------------------------------------------------
# Analysis modules
# ----------------------------------------------------------------------
def analyze_rating(ratings: pd.DataFrame, pos_thr: int) -> None:
    section("1. Rating distribution")
    total = len(ratings)
    rcnt = ratings["rating"].value_counts().sort_index()
    for r, c in rcnt.items():
        print(f"  rating={r}  {c:>9,}  ({pct(c, total):5.2f}%)  {bar(c / total)}")

    n_pos = int((ratings["rating"] >= pos_thr).sum())
    n_neg = int((ratings["rating"] <= WEAK_NEGATIVE_MAX).sum())
    print(f"  positive (>={pos_thr}): {n_pos:,} ({pct(n_pos, total):.2f}%)")
    print(f"  negative (<={WEAK_NEGATIVE_MAX}): {n_neg:,} ({pct(n_neg, total):.2f}%)")


def analyze_time(ratings: pd.DataFrame) -> None:
    section("2. Time distribution")
    ts = ratings["ts"]
    total = len(ratings)
    print(f"  Time span: {ts.min()}  ~  {ts.max()}  ({(ts.max() - ts.min()).days} days)")

    monthly = ratings.set_index("ts").resample("MS").size()
    print("  Monthly counts (first 5 / last 5):")
    for d, c in monthly.head(5).items():
        print(f"    {d.date()}  {c:>9,}")
    print("    ...")
    for d, c in monthly.tail(5).items():
        print(f"    {d.date()}  {c:>9,}")
    print(f"  Hottest month: {monthly.idxmax().date()} -> {monthly.max():,} interactions")
    print(f"  Coldest month: {monthly.idxmin().date()} -> {monthly.min():,} interactions")

    t_max = ts.max()
    for days in (7, 30):
        n = int((ts >= t_max - pd.Timedelta(days=days)).sum())
        print(f"  Last {days:>2} days: {n:>9,} interactions  ({pct(n, total):.4f}%)")
    print("  -> Data is heavily concentrated in the early period; "
          "a global time-based split is not viable.")


def analyze_user_length(ratings: pd.DataFrame, min_user_inter: int) -> None:
    section("3. User behavior length distribution (user_id -> n_inter)")
    user_n = ratings.groupby("user_id").size()
    print(f"  Number of users: {len(user_n):,}")
    print_quantiles(user_n)
    n_active = int((user_n >= min_user_inter).sum())
    print(f"  Users with >={min_user_inter} interactions: "
          f"{n_active:,} ({pct(n_active, len(user_n)):.1f}%)")


def analyze_item_popularity(ratings: pd.DataFrame) -> None:
    section("4. Item popularity distribution (item_id -> n_inter)")
    item_n = ratings.groupby("item_id").size().sort_values(ascending=False)
    exposure = int(item_n.sum())
    print(f"  Number of items: {len(item_n):,}")
    print_quantiles(item_n)

    n_hot = int(0.2 * len(item_n))
    hot_exposure = int(item_n.head(n_hot).sum())
    cum_share = item_n.cumsum() / exposure
    items_for_half = int((cum_share < 0.5).sum()) + 1
    n_cold = int((item_n <= 2).sum())

    print(f"  Top 20% hot items ({n_hot:,} items) account for "
          f"{pct(hot_exposure, exposure):.1f}% of exposure")
    print(f"  Only {items_for_half:,} items ({pct(items_for_half, len(item_n)):.1f}%) "
          f"cover 50% of exposure")
    print(f"  Cold items with 1-2 interactions: {n_cold:,} ({pct(n_cold, len(item_n)):.1f}%)")


def analyze_sparsity(ratings: pd.DataFrame) -> None:
    section("5. user-item matrix sparsity")
    n_user = ratings["user_id"].nunique()
    n_item = ratings["item_id"].nunique()
    n_pair = len(ratings)
    cells = n_user * n_item
    density = pct(n_pair, cells)
    print(f"  Matrix shape: {n_user:,} x {n_item:,} = {cells:,} cells")
    print(f"  Filled cells: {n_pair:,}")
    print(f"  Density:  {density:.4f}%")
    print(f"  Sparsity: {100 - density:.4f}%")
    print("  Compare: real UGC platforms have density < 0.001%; "
          "MovieLens is dense by comparison.")


def analyze_user_profile(users: pd.DataFrame) -> None:
    section("6. User static profile")
    n_users = len(users)

    print("  Gender:")
    for g, c in users["gender"].value_counts().items():
        print(f"    {g}: {c:>5,} ({pct(c, n_users):.1f}%)")

    print("  Age (MovieLens discretized buckets):")
    for a, c in users["age"].value_counts().sort_index().items():
        label = AGE_MAP.get(a, str(a))
        print(f"    {label:<8} ({a:>2}): {c:>5,} ({pct(c, n_users):5.1f}%)  {bar(c / n_users, 40)}")

    print(f"  Occupation: {users['occupation'].nunique()} categories, top 5:")
    for o, c in users["occupation"].value_counts().head(5).items():
        print(f"    {o:>2} ({OCC_MAP.get(o, '?'):<14}): {c:>5,} ({pct(c, n_users):.1f}%)")


def analyze_item_metadata(movies: pd.DataFrame) -> None:
    section("7. Item (movie) metadata")
    n_movies = len(movies)

    # Release year is the trailing "(YYYY)" in the title; anchor to end of string
    # and use expand=False to get a Series, avoiding stray digits inside the title.
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).astype("float")
    n_missing_year = int(year.isna().sum())
    print("  Year distribution by decade:")
    decade = (year // 10 * 10).dropna().astype(int).value_counts().sort_index()
    for d, c in decade.items():
        print(f"    {d}s: {c:>5,} ({pct(c, n_movies):5.1f}%)  {bar(c / n_movies, 40)}")
    if n_missing_year:
        print(f"    (year unparsed: {n_missing_year:,})")

    print("\n  Genre distribution:")
    genre_lists = movies["genres"].str.split("|")
    all_g: Counter = Counter()
    for gs in genre_lists:
        all_g.update(gs)
    avg_g = genre_lists.str.len().mean()
    print(f"  {len(all_g)} genres total, average {avg_g:.2f} genres per item")
    print("  Top 10:")
    for g, c in all_g.most_common(10):
        print(f"    {g:<14}: {c:>5,} ({pct(c, n_movies):5.1f}%)  {bar(c / n_movies, 30)}")


def analyze_signal_richness() -> None:
    section("8. Signal richness comparison (vs UGC video platforms)")
    available = {
        "user_id + item_id + timestamp": True,
        "explicit positive feedback (high rating)": True,
        "explicit negative feedback (low rating)": True,
        "user static profile": True,
        "item category": True,
        "impression log": False,
        "dwell time / play completion": False,
        "like / comment / share": False,
        "follow relationship (social graph)": False,
        "context (device / time-of-day / network)": False,
        "item raw text / cover / video file": False,
        "item author / creator": False,
        "real-time feedback stream": False,
    }
    for k, v in available.items():
        print(f"  [{'OK' if v else '--'}] {k}")
    n_have = sum(available.values())
    print(f"\n  Has {n_have}/{len(available)} core signal types -> "
          "main gaps are 'behavior diversity + content understanding + context'.")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(cfg: Config | None = None) -> None:
    cfg = cfg or load_config()
    log = get_logger("eda")

    dataset = cfg["dataset"]
    pos_thr = int(dataset["positive_threshold"])
    min_user_inter = int(dataset["min_user_inter"])

    ratings, movies, users = load_movielens_1m(cfg.path("raw_dir"))
    log.info(f"loaded: ratings={len(ratings):,}  movies={len(movies):,}  users={len(users):,}")

    analyze_rating(ratings, pos_thr)
    analyze_time(ratings)
    analyze_user_length(ratings, min_user_inter)
    analyze_item_popularity(ratings)
    analyze_sparsity(ratings)
    analyze_user_profile(users)
    analyze_item_metadata(movies)
    analyze_signal_richness()


if __name__ == "__main__":
    main()
