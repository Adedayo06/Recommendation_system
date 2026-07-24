"""
Train the production recommender on the full event log and save it to disk.

This differs from evaluate_system.py in one important way: it trains on
*everything*. The evaluation script deliberately holds out each customer's most
recent interaction so it has something honest to score against; that held-out
data is real signal you want in the deployed model, so it goes back in here.

Hyperparameters come from evaluate_system.py, which is the single place they
are defined — change them there and both scripts follow.

Usage:
    python train_model.py
    python train_model.py --include-history     # self-contained offline artifact
"""

import argparse
import time
from pathlib import Path

from data_loader import load_events
from evaluate_system import (
    CF_PARAMS,
    K_NEIGHBORS,
    MIN_ITEM_INTERACTIONS,
    MIN_KNOWN_INTERACTIONS,
    MIN_USER_INTERACTIONS,
)
from persistence import save_model
from recommender import HybridRecommender
from s3_data import fetch

SRC_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SRC_DIR.parent / "models" / "recommender.pkl"


def train(events_path=None):
    # No explicit path -> pull events.csv from S3 (cached locally after the
    # first run). An explicit path is used as-is, so you can still train on a
    # local file if you have one.
    if events_path is None:
        events_path = fetch("raw/events.csv")
    print(f"Loading events from {events_path} ...")
    events = load_events(events_path)
    print(f"Total events: {len(events):,}")

    print("Fitting on the full event log (no holdout — this is the deployed model) ...")
    started = time.time()
    rec = HybridRecommender(
        min_known_interactions=MIN_KNOWN_INTERACTIONS,
        k_neighbors=K_NEIGHBORS,
        **CF_PARAMS,
    )
    rec.fit(
        events,
        min_user_interactions=MIN_USER_INTERACTIONS,
        min_item_interactions=MIN_ITEM_INTERACTIONS,
    )
    print(f"  fitted in {time.time() - started:.0f}s")
    print(f"  catalog          : {len(rec.item_cf.item_id_map):,} items")
    print(f"  similarity edges : {rec.item_cf.sim_matrix.nnz:,}")
    print(f"  transition edges : {rec.sequence_rules.transitions.nnz:,}")
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="also embed the per-customer history index (much larger file; only "
             "useful for a self-contained offline artifact — a live service "
             "should read history from its own event store)",
    )
    parser.add_argument(
        "--events",
        default=None,
        help="path to events.csv; omit to fetch it from S3 (see src/s3_data.py)",
    )
    args = parser.parse_args()

    rec = train(args.events)

    print(f"\nSaving to {args.output} ...")
    path = save_model(rec, args.output, include_history=args.include_history)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  wrote {size_mb:.1f} MB")

    # Load it straight back and serve one recommendation. A model that saves
    # but cannot be loaded and used is worse than no model, and that failure
    # should surface here rather than in the API at 3am.
    print("\nVerifying the saved file round-trips ...")
    from persistence import load_model

    reloaded = load_model(path)
    sample_item = next(iter(reloaded.item_cf.item_id_map))
    probe = [(sample_item, 1.0, 0)]

    original = rec.recommend(user_history=probe, n=10)
    restored = reloaded.recommend(user_history=probe, n=10)

    if [i for i, _, _ in original] != [i for i, _, _ in restored]:
        raise SystemExit("FAILED: reloaded model gives different recommendations")

    print(f"  OK — reloaded model reproduces recommendations for item {sample_item}")
    print(f"\nDone. Load it with:  from persistence import load_model; "
          f"rec = load_model({str(path)!r})")


if __name__ == "__main__":
    main()
