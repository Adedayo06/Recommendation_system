"""
Hyperparameter tuning for the item-based CF recommender.

Search strategy: coordinate descent, not a full cartesian grid. Each stage
sweeps one axis while holding the current best configuration fixed, then the
winner of that stage is carried into the next. A full grid over these seven
axes would be thousands of fits; this gets most of the benefit in a few dozen.

Protocol: the held-out cases are split into a VALIDATION half (used to pick
the configuration) and a TEST half (used once, at the end, to report the
result). Tuning and reporting on the same sample is how you end up with
hyperparameters fitted to the evaluation sample rather than to the problem —
the gap between the two numbers below tells you how much of that happened.

Costwise, the popularity model, the category map, and the per-customer history
index are all independent of the CF hyperparameters, so they are built once up
front and reused across every configuration. Only ItemBasedCF is refit.

Usage:
    python tune.py
"""

import copy
import random
import time

import numpy as np

from evaluate_system import (
    EVENTS_PATH,
    build_history_lookup,
    evaluate_ranking_and_quality,
    load_and_split,
)
from item_cf import ItemBasedCF
from recommender import HybridRecommender
from sequence_rules import SequentialRules
from sessions import with_sessions

TOP_N = 10
TUNING_SAMPLE = 6000     # split in half: 3000 validation / 3000 test
RANDOM_SEED = 42

# The configuration this project started from, kept in the sweep so the report
# shows the delta rather than an unanchored number.
BASELINE_CONFIG = dict(
    min_user_interactions=3,
    min_item_interactions=3,
    min_known_interactions=2,
    k_neighbors=20,
    weight_scheme="raw",
    use_idf=False,
    shrinkage=0.0,
    similarity_power=1.0,
    history_weight_scheme="raw",
    max_user_items=None,
    position_decay=1.0,
    co_occurrence_unit="visitorid",
    sequence_weight=0.0,
    sequence_window=3,
    sequence_position_decay=0.7,
    sequence_max_items=5,
    current_session_only=False,
    max_history_items=None,
)

# Starting point for the search: the winner of the previous (sequence-blind)
# sweep, so the sequence axes below are measured as an increment on top of an
# already-tuned model rather than credited with its gains.
START_CONFIG = dict(
    min_user_interactions=2,
    min_item_interactions=2,
    min_known_interactions=1,
    k_neighbors=100,
    weight_scheme="log",
    use_idf=True,
    shrinkage=50.0,
    similarity_power=0.5,
    history_weight_scheme="binary",
    max_user_items=200,
    # Sequence-awareness axes, all starting at their no-op values.
    position_decay=1.0,          # 1.0 = history is an unordered bag
    co_occurrence_unit="visitorid",
    sequence_weight=0.0,         # 0.0 = directed transition model disabled
    sequence_window=3,
    sequence_position_decay=0.7,
    sequence_max_items=5,
    current_session_only=False,
    max_history_items=None,
)

# Each stage is (axis_name, [values to try]). Order matters a little: the
# axes with the largest expected effect go first, so later stages tune on top
# of an already-sensible configuration.
STAGES = [
    # --- sequence-awareness axes, swept first on top of the previous winner ---
    ("position_decay", [1.0, 0.95, 0.9, 0.8, 0.7, 0.5]),
    ("sequence_weight", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]),
    ("sequence_window", [1, 2, 3, 5]),
    ("sequence_position_decay", [1.0, 0.7, 0.5]),
    ("sequence_max_items", [1, 3, 5, 10]),
    ("current_session_only", [False, True]),
    ("max_history_items", [None, 5, 10, 20]),
    ("co_occurrence_unit", ["visitorid", "sessionid"]),
    # --- re-check the core axes now that the profile has changed shape ---
    ("shrinkage", [10.0, 25.0, 50.0, 100.0]),
    ("k_neighbors", [50, 100, 200]),
    ("history_weight_scheme", ["raw", "log", "binary"]),
    ("similarity_power", [0.5, 1.0, 1.5]),
]

CF_KEYS = {
    "k_neighbors", "weight_scheme", "use_idf", "shrinkage",
    "similarity_power", "history_weight_scheme", "max_user_items",
    "position_decay", "co_occurrence_unit",
}
HYBRID_KEYS = {
    "sequence_weight", "sequence_window", "sequence_position_decay",
    "sequence_max_items", "current_session_only", "max_history_items",
}

# Refitting the CF is the expensive part, and most of the sequence axes don't
# touch it. Cache fitted CF models (and their sequence-rule companions) by the
# subset of the config that actually determines them.
_CF_CACHE = {}
_SR_CACHE = {}


def _cf_key(config):
    return tuple(sorted(
        (k, v) for k, v in config.items()
        if k in CF_KEYS or k in ("min_user_interactions", "min_item_interactions")
    ))


def fit_cf(config, train_events_df):
    """Fit just the item-based CF for one configuration, with caching."""
    key = _cf_key(config)
    if key not in _CF_CACHE:
        cf = ItemBasedCF(**{k: v for k, v in config.items() if k in CF_KEYS})
        events = train_events_df
        if config["co_occurrence_unit"] == "sessionid":
            events = with_sessions(train_events_df)
        cf.fit(
            events,
            min_user_interactions=config["min_user_interactions"],
            min_item_interactions=config["min_item_interactions"],
        )
        _CF_CACHE.clear()  # only ever need the current one; keeps memory flat
        _SR_CACHE.clear()
        _CF_CACHE[key] = cf
    return _CF_CACHE[key]


def fit_sequence_rules(config, cf, train_events_df):
    """Fit the directed transition model, cached on (catalog, window)."""
    if config["sequence_weight"] <= 0:
        return None
    key = (_cf_key(config), config["sequence_window"])
    if key not in _SR_CACHE:
        _SR_CACHE[key] = SequentialRules(window=config["sequence_window"]).fit(
            train_events_df, cf.item_id_map
        )
    return _SR_CACHE[key]


def score_config(config, rec, train_events_df, test_cases, history_lookup, top_n=TOP_N):
    """Fit for this config, hot-swap into the shared hybrid, evaluate."""
    started = time.time()
    cf = fit_cf(config, train_events_df)
    rec.item_cf = cf
    rec.min_known_interactions = config["min_known_interactions"]

    # sequence_window is vestigial as a bare attribute — the real window lives
    # in rec.sequence_rules, swapped in below — but harmless to set.
    for key in HYBRID_KEYS:
        setattr(rec, key, config[key])

    sr = fit_sequence_rules(config, cf, train_events_df)
    if sr is not None:
        rec.sequence_rules = sr

    metrics = evaluate_ranking_and_quality(
        rec, test_cases, history_lookup, top_n=top_n, with_quality=False
    )
    metrics["seconds"] = time.time() - started
    return metrics


def format_row(label, m):
    return (
        f"  {label:<34} HR@10 {m['hit_rate']:.4f}  NDCG {m['ndcg']:.4f}  "
        f"cov {m['coverage']:.3f}  cf-served {m['cf_served_rate']*100:5.1f}%  "
        f"({m['seconds']:.0f}s)"
    )


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_events_df, all_cases = load_and_split(sample_size=TUNING_SAMPLE, seed=RANDOM_SEED)

    half = len(all_cases) // 2
    val_cases, test_cases = all_cases[:half], all_cases[half:]
    print(f"Validation cases: {len(val_cases):,} | Held-out test cases: {len(test_cases):,}")

    print("Building shared popularity model / category map / history index (once) ...")
    started = time.time()
    rec = HybridRecommender()
    rec.fit(train_events_df, min_user_interactions=2, min_item_interactions=2)
    print(f"  done in {time.time()-started:.0f}s")

    val_history = build_history_lookup(train_events_df, [v for v, _, _ in val_cases])
    test_history = build_history_lookup(train_events_df, [v for v, _, _ in test_cases])

    print("\n" + "=" * 78)
    print("REFERENCE POINTS (validation set)")
    print("=" * 78)
    baseline_metrics = score_config(BASELINE_CONFIG, rec, train_events_df, val_cases, val_history)
    print(format_row("original config", baseline_metrics))

    best_config = copy.deepcopy(START_CONFIG)
    best_metrics = score_config(best_config, rec, train_events_df, val_cases, val_history)
    print(format_row("search starting point", best_metrics))

    print("\n" + "=" * 78)
    print("COORDINATE-DESCENT SWEEP (validation set)")
    print("=" * 78)

    for axis, values in STAGES:
        print(f"\n-- {axis} (current best: {best_config[axis]!r}) --")
        stage_best_value = best_config[axis]
        stage_best_metrics = best_metrics

        for value in values:
            if value == best_config[axis]:
                print(format_row(f"{axis}={value!r}  [current]", best_metrics))
                continue
            candidate = copy.deepcopy(best_config)
            candidate[axis] = value
            metrics = score_config(candidate, rec, train_events_df, val_cases, val_history)
            marker = ""
            if metrics["hit_rate"] > stage_best_metrics["hit_rate"]:
                stage_best_value = value
                stage_best_metrics = metrics
                marker = "  <-- best so far"
            print(format_row(f"{axis}={value!r}", metrics) + marker)

        best_config[axis] = stage_best_value
        best_metrics = stage_best_metrics
        print(f"   => keeping {axis}={stage_best_value!r}  (HR@10 {best_metrics['hit_rate']:.4f})")

    print("\n" + "=" * 78)
    print("BEST CONFIGURATION")
    print("=" * 78)
    for key in sorted(best_config):
        print(f"  {key:<26} {best_config[key]!r}")

    print("\n" + "=" * 78)
    print("FINAL SCORES ON THE HELD-OUT TEST HALF (not used during tuning)")
    print("=" * 78)
    baseline_test = score_config(BASELINE_CONFIG, rec, train_events_df, test_cases, test_history)
    tuned_test = score_config(best_config, rec, train_events_df, test_cases, test_history)

    print(format_row("original config", baseline_test))
    print(format_row("tuned config", tuned_test))

    lift = (tuned_test["hit_rate"] / baseline_test["hit_rate"] - 1) * 100 if baseline_test["hit_rate"] else float("nan")
    print(f"\n  Hit Rate @10 : {baseline_test['hit_rate']:.4f} -> {tuned_test['hit_rate']:.4f}  ({lift:+.1f}%)")
    print(f"  NDCG @10     : {baseline_test['ndcg']:.4f} -> {tuned_test['ndcg']:.4f}")
    print(f"  Validation HR: {best_metrics['hit_rate']:.4f}  (gap to test HR shows tuning overfit)")
    print("=" * 78)
    print("\nCopy the best configuration into evaluate_system.py (K_NEIGHBORS / CF_PARAMS /")
    print("MIN_* constants) to make it the default, then run evaluate_system.py for the")
    print("full report including diversity, novelty and coverage.")


if __name__ == "__main__":
    main()
