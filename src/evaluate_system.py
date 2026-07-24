"""
Evaluation harness for the hybrid recommender (item-based CF + popularity
fallback).

Protocol: leave-last-out. For every customer with 2+ distinct items, their
single most recent interaction is held out as a "test" case and removed
from training; the model is fit on everything else. This mirrors how the
system is actually used — predict the *next* thing a customer does based on
what they've done so far.

TWO JOBS, TWO NUMBERS. 25% of held-out events are the customer returning to an
item they had already viewed. The recommender deliberately never recommends
something already in a customer's history — a homepage full of things they have
already seen is a bad homepage — so those cases are structurally unwinnable
here. They are not a model failure; they are a different product feature (a
"recently viewed" row, which is a database query on the backend, not a model).

Averaging the two together produced a single number that understated the model
by grading it on work it was never assigned. So the report leads with DISCOVERY
hit rate — cases where the next click really was on something new — and reports
reminder cases separately, where the expected score is exactly zero.

Note this does not change hyperparameter tuning: reminder cases score 0 for
every configuration, and what fraction of cases are reminders depends only on
the data, so discovery hit rate is a constant multiple of blended hit rate and
both rank configurations identically.

Metrics:
    Hit Rate      % of test customers for whom the held-out item actually
                  appears in their top-N recommendation list. With exactly
                  one held-out item per customer this is also Recall@N.
    NDCG          Like Hit Rate but discounted by rank — a hit at position 1
                  is worth 1.0, at position 10 about 0.29.
    ARHR / MRR    Average Reciprocal Hit Rate (1/rank instead of 1).
    Coverage      % of the model's catalog that gets recommended to at
                  least one test customer (checks the model isn't just
                  recommending the same handful of items to everyone).
    Diversity     Average dissimilarity between items within a single
                  recommendation list (are the 10 items actually different
                  from each other, or all near-duplicates?).
    Novelty       How non-obvious the recommended items are, based on how
                  unpopular they are overall (recommending only the
                  best-sellers scores low; surfacing relevant long-tail
                  items scores high).

Two reference points are reported alongside the model, because a hit rate on
its own means nothing without them:
    Reachable     % of test cases whose held-out item is even in the model's
                  catalog at all. This is the hard ceiling on Hit Rate — you
                  cannot recommend an item you filtered out of training.
    Popularity    what a top-N-bestsellers-for-everybody ranker scores on the
                  same cases. The CF model has to beat this to justify itself.

    RMSE, MAE     Reported last, and deliberately de-emphasised. RetailRocket
                  has no explicit ratings, so these compare a predicted
                  implicit event weight against the held-out event's weight.
                  They are computable but close to meaningless: they only
                  cover the minority of cases where the held-out item has
                  neighbor overlap, and they measure calibration of a
                  made-up scale rather than whether the ranking is any good.
                  Judge this system on Hit Rate / NDCG.

Usage:
    python evaluate_system.py
"""

import math
import random
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_events
from recommender import HybridRecommender
from s3_data import fetch
from sessions import build_ordered_history_index

# Anchored to this file's location, not the current working directory, so
# `python evaluate_system.py` works whether you run it from src/, the
# project root, or anywhere else.
SRC_DIR = Path(__file__).resolve().parent
EVENTS_PATH = str(SRC_DIR.parent / "data" / "raw" / "events.csv")

TOP_N = 10
TOP_N_ALSO_EVALUATE = 20  # report metrics at both list lengths, since Hit Rate/ARHR are sensitive to N

# Tuned defaults — produced by the coordinate-descent sweep in tune.py, which
# scored HR@10 0.0837 -> 0.1813 against the original settings on a held-out
# test half that was not used during tuning.
#
# Note on MIN_ITEM_INTERACTIONS: 2 maximises hit rate, but 3 costs only ~0.7%
# relative hit rate while lifting catalog coverage from 26% to 35%, and 5
# lifts it to 47% for ~5% relative hit rate. On a marketplace where seller
# exposure matters, that trade is worth making deliberately rather than
# defaulting to whichever number tops the leaderboard.
MIN_USER_INTERACTIONS = 2
MIN_ITEM_INTERACTIONS = 2
MIN_KNOWN_INTERACTIONS = 1
K_NEIGHBORS = 100
CF_PARAMS = dict(
    weight_scheme="log",
    use_idf=True,
    shrinkage=50.0,
    similarity_power=0.5,
    history_weight_scheme="binary",
    max_user_items=200,
    # --- sequence awareness ---
    position_decay=0.7,          # older history items count less
    co_occurrence_unit="visitorid",   # 'sessionid' scored worse here, see sessions.py
    sequence_weight=0.2,         # share of the score from directed transitions
    sequence_window=3,
    sequence_position_decay=1.0,
    sequence_max_items=3,
    current_session_only=True,   # score against the current session only
    max_history_items=5,
)

EVAL_SAMPLE_SIZE = 5000   # number of test customers to evaluate against (full set is ~288k)
RANDOM_SEED = 42


def leave_last_out_split(events_df):
    """
    Vectorized leave-last-out split: for every customer with 2+ distinct
    items, pull out their most recent event as a test case.

    Returns:
        train_events_df: everything except the held-out rows
        test_cases: [(visitorid, held_out_itemid, held_out_weight), ...]
    """
    sorted_df = events_df.sort_values("timestamp")
    distinct_counts = sorted_df.groupby("visitorid")["itemid"].transform("nunique")
    eligible = sorted_df[distinct_counts >= 2]
    test_rows = eligible.groupby("visitorid").tail(1)

    train_events_df = events_df.drop(index=test_rows.index)
    test_cases = list(test_rows[["visitorid", "itemid", "weight"]].itertuples(index=False, name=None))
    return train_events_df, test_cases


def build_history_lookup(train_events_df, visitor_ids) -> dict:
    """
    {visitorid: [(itemid, weight, last_timestamp), ...]} for a specific set of
    customers only — building this for every customer in a multi-million row
    log is wasteful when we only need it for the sampled evaluation set.

    Chronologically ordered, oldest first. This matters: position decay and the
    sequence model both read recency from list order, so a plain groupby (which
    sorts by itemid) would leave them silently measuring nothing.
    """
    subset = train_events_df[train_events_df["visitorid"].isin(set(visitor_ids))]
    return build_ordered_history_index(subset)


def evaluate_rating_accuracy(item_cf, test_cases, history_lookup):
    """RMSE and MAE of predicted event-weight vs actual, on the held-out
    interactions. See the module docstring for why this is a footnote."""
    squared_errors = []
    abs_errors = []
    skipped = 0

    for visitorid, held_out_item, actual_weight in test_cases:
        history = history_lookup.get(visitorid, [])
        predicted = item_cf.predict_score(history, held_out_item)
        if predicted is None:
            skipped += 1
            continue
        error = predicted - actual_weight
        squared_errors.append(error ** 2)
        abs_errors.append(abs(error))

    rmse = math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else float("nan")
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")
    return rmse, mae, skipped, len(squared_errors)


def evaluate_ranking_and_quality(rec, test_cases, history_lookup, top_n=TOP_N, with_quality=True):
    """
    Hit Rate, NDCG, ARHR, Coverage, Diversity, Novelty — all derived from the
    top-N recommendation list generated for each test customer. Also returns
    the reachability ceiling and a popularity-only baseline for reference.

    with_quality=False skips Diversity/Novelty, which are the expensive parts
    and irrelevant while grid-searching for hit rate.
    """
    catalog = rec.item_cf.item_id_map
    catalog_size = len(catalog)
    total_pop_weight = sum(rec.popularity.scores.values())
    pop_baseline_list = rec.popularity.recommend(n=top_n)

    hits = 0
    cf_hits = 0
    cf_served = 0
    reachable = 0
    pop_baseline_hits = 0
    reciprocal_ranks = []
    ndcg_scores = []
    all_recommended_items = set()
    diversity_scores = []
    novelty_scores = []

    # Reminder vs discovery — see the module docstring. Counted separately
    # because they are two different jobs and only one of them is this
    # model's.
    reminder_cases = 0
    reminder_hits = 0
    discovery_cases = 0
    discovery_hits = 0
    discovery_ndcg = []
    discovery_reachable = 0

    for visitorid, held_out_item, _ in test_cases:
        history = history_lookup.get(visitorid, [])
        recs = rec.recommend(user_id=None, user_history=history, n=top_n)
        recommended_items = [item for item, score, source in recs]

        served_by_cf = any(source == "item_cf" for _, _, source in recs)
        cf_served += served_by_cf

        # Did the customer already know about this item before the held-out
        # event? If so it is a reminder case, not a recommendation case.
        already_seen = any(item == held_out_item for item, *_ in history)
        if already_seen:
            reminder_cases += 1
        else:
            discovery_cases += 1
            if held_out_item in catalog:
                discovery_reachable += 1

        # Can this case even be answered? An item filtered out of training can
        # never be recommended, so it caps the achievable hit rate.
        if held_out_item in catalog:
            reachable += 1
        if held_out_item in pop_baseline_list:
            pop_baseline_hits += 1

        # --- Hit Rate / ARHR / NDCG ---
        if held_out_item in recommended_items:
            hits += 1
            cf_hits += served_by_cf
            rank = recommended_items.index(held_out_item) + 1
            reciprocal_ranks.append(1.0 / rank)
            ndcg_scores.append(1.0 / math.log2(rank + 1))
            if already_seen:
                reminder_hits += 1
            else:
                discovery_hits += 1
                discovery_ndcg.append(1.0 / math.log2(rank + 1))
        else:
            reciprocal_ranks.append(0.0)
            ndcg_scores.append(0.0)
            if not already_seen:
                discovery_ndcg.append(0.0)

        all_recommended_items.update(recommended_items)

        if not with_quality:
            continue

        # --- Diversity: average pairwise dissimilarity within this list ---
        # Batched as one cosine_similarity call over the small (<=N x N) sub-matrix
        # of recommended items, rather than one sklearn call per pair — same
        # result, much faster over many test customers.
        known_indices = [catalog[i] for i in recommended_items if i in catalog]
        if len(known_indices) >= 2:
            sim_matrix = cosine_similarity(rec.item_cf.item_vectors[known_indices])
            n = sim_matrix.shape[0]
            num_pairs = n * (n - 1) / 2
            pair_sim_sum = (sim_matrix.sum() - n) / 2  # drop the diagonal (self-similarity = 1), each pair counted twice
            diversity_scores.append(1 - (pair_sim_sum / num_pairs))

        # --- Novelty: -log2(popularity probability), averaged over the list ---
        for item in recommended_items:
            pop = rec.popularity.scores.get(item, 0)
            prob = pop / total_pop_weight if total_pop_weight else 0
            if prob > 0:
                novelty_scores.append(-math.log2(prob))

    n_cases = len(test_cases) or 1

    def mean(values):
        return sum(values) / len(values) if values else float("nan")

    return {
        "hit_rate": hits / n_cases,
        "ndcg": mean(ndcg_scores),
        "arhr": mean(reciprocal_ranks),
        "coverage": len(all_recommended_items) / catalog_size if catalog_size else float("nan"),
        "diversity": mean(diversity_scores),
        "novelty": mean(novelty_scores),
        "reachable_rate": reachable / n_cases,
        # Hit rate measured only against cases that were actually answerable —
        # separates "the ranker is bad" from "the item wasn't in the catalog".
        "hit_rate_of_reachable": hits / reachable if reachable else float("nan"),
        "cf_served_rate": cf_served / n_cases,
        "cf_hit_rate": cf_hits / cf_served if cf_served else float("nan"),
        "popularity_baseline_hit_rate": pop_baseline_hits / n_cases,
        "n_cases": len(test_cases),

        # --- reminder vs discovery ---
        "reminder_rate": reminder_cases / n_cases,
        "reminder_hit_rate": reminder_hits / reminder_cases if reminder_cases else float("nan"),
        "discovery_rate": discovery_cases / n_cases,
        "discovery_hit_rate": discovery_hits / discovery_cases if discovery_cases else float("nan"),
        "discovery_ndcg": mean(discovery_ndcg),
        # The genuine ceiling for this model: item is in the catalog AND the
        # customer had not already seen it.
        "discovery_ceiling": discovery_reachable / n_cases,
        "discovery_hit_rate_of_reachable": (
            discovery_hits / discovery_reachable if discovery_reachable else float("nan")
        ),
    }


def load_and_split(sample_size=EVAL_SAMPLE_SIZE, seed=RANDOM_SEED, verbose=True):
    """Shared setup: load events, hold out each customer's last interaction,
    and sample the customers to evaluate on."""
    if verbose:
        print(f"Loading events from {EVENTS_PATH} ...")
    # fetch() downloads events.csv from S3 to the local cache on first use,
    # then returns the same local path EVENTS_PATH points at.
    events = load_events(fetch("raw/events.csv"))
    if verbose:
        print(f"Total events: {len(events):,}")
        print("Splitting leave-last-out train/test ...")

    train_events_df, test_cases = leave_last_out_split(events)
    if verbose:
        print(f"Train events: {len(train_events_df):,} | Total eligible test cases: {len(test_cases):,}")

    rng = random.Random(seed)
    if sample_size and len(test_cases) > sample_size:
        test_cases = rng.sample(test_cases, sample_size)
    return train_events_df, test_cases


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_events_df, test_cases = load_and_split()
    print(f"Evaluating on a sample of {len(test_cases):,} customers.")

    print("Fitting model on training data ...")
    rec = HybridRecommender(
        min_known_interactions=MIN_KNOWN_INTERACTIONS,
        k_neighbors=K_NEIGHBORS,
        **CF_PARAMS,
    )
    rec.fit(
        train_events_df,
        min_user_interactions=MIN_USER_INTERACTIONS,
        min_item_interactions=MIN_ITEM_INTERACTIONS,
    )

    visitor_ids = [v for v, _, _ in test_cases]
    history_lookup = build_history_lookup(train_events_df, visitor_ids)

    print("\n" + "=" * 62)
    print("EVALUATION REPORT")
    print("=" * 62)
    print(f"Catalog after filtering   : {len(rec.item_cf.item_id_map):,} items")
    print(f"Test customers evaluated  : {len(test_cases):,}")

    for top_n in (TOP_N, TOP_N_ALSO_EVALUATE):
        print(f"\nComputing metrics at N={top_n} ...")
        m = evaluate_ranking_and_quality(rec, test_cases, history_lookup, top_n=top_n)

        print("-" * 62)
        print(f"Top-N = {top_n}")
        print("  --- DISCOVERY: what this model is actually responsible for ---")
        print(f"  Discovery Hit Rate      : {m['discovery_hit_rate']:.4f}   <-- HEADLINE METRIC")
        print(f"  Discovery NDCG          : {m['discovery_ndcg']:.4f}")
        print(f"  Discovery ceiling       : {m['discovery_ceiling']:.4f}   (in catalog AND not already seen)")
        print(f"  Discovery HR / ceiling  : {m['discovery_hit_rate_of_reachable']:.4f}   (share of the achievable captured)")
        print(f"  Discovery cases         : {m['discovery_rate']*100:.1f}% of test customers")
        print("  " + "-" * 58)
        print("  --- REMINDER: backend 'recently viewed' territory, not this model ---")
        print(f"  Reminder cases          : {m['reminder_rate']*100:.1f}% of test customers")
        print(f"  Reminder Hit Rate       : {m['reminder_hit_rate']:.4f}   (structurally 0 — we exclude seen items)")
        print("  " + "-" * 58)
        print("  --- BLENDED: the two jobs averaged together; kept for continuity ---")
        print(f"  Hit Rate                : {m['hit_rate']:.4f}   ({m['hit_rate']*100:.1f}% of ALL customers, including unwinnable reminder cases)")
        print(f"  NDCG                    : {m['ndcg']:.4f}")
        print(f"  ARHR                    : {m['arhr']:.4f}")
        print("  " + "-" * 58)
        print(f"  Coverage                : {m['coverage']:.4f}   ({m['coverage']*100:.1f}% of catalog recommended at least once)")
        print(f"  Diversity               : {m['diversity']:.4f}   (0 = all near-identical, 1 = maximally dissimilar)")
        print(f"  Novelty                 : {m['novelty']:.4f}   (higher = less-obvious items)")
        print(f"  Served by item_cf       : {m['cf_served_rate']*100:.1f}%")
        print(f"  Popularity-only baseline: {m['popularity_baseline_hit_rate']:.4f}   (what we have to beat)")

    print("\nComputing RMSE / MAE (see module docstring — footnote metric) ...")
    rmse, mae, skipped, n_scored = evaluate_rating_accuracy(rec.item_cf, test_cases, history_lookup)
    print("-" * 62)
    print(f"  RMSE                    : {rmse:.4f}   (on {n_scored:,} predictable pairs, {skipped:,} skipped)")
    print(f"  MAE                     : {mae:.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
