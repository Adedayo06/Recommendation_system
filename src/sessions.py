"""
Session segmentation.

RetailRocket has no session id, so sessions are inferred the usual way: a
visitor's events are split wherever they go quiet for longer than a gap
threshold (30 minutes by convention).

What sessions are worth in *this* dataset, measured rather than assumed:

    sessions                          1,761,675 at a 30-minute gap
    mean distinct items per session   1.33  (p95 = 3)
    multi-item sessions                 263,899  — only 15% of sessions
    active visitors (2+ items)          288,080
      ...with only one session ever         52%
      ...with 2+ items in their last        63%
    median share of a visitor's history
      sitting inside their last session    100%   (mean 74%)

The headline: for over half of the customers the model actually serves, "their
current session" and "their entire history" are the same set of items, so
re-segmenting co-occurrence by session instead of by visitor cannot tell them
apart. That is why `co_occurrence_unit` is offered as a tunable rather than
assumed to be an improvement — on a dataset with longer sessions it usually is,
here it mostly is not.

The parts of sequence-awareness that do pay off here are the two that work
inside a single short session: weighting recent items more heavily than old
ones (position decay, see profile_from_events) and modelling which item tends
to follow which (sequence_rules.py).
"""

import numpy as np
import pandas as pd

SESSION_GAP_MINUTES = 30


def assign_sessions(events_df: pd.DataFrame, gap_minutes: float = SESSION_GAP_MINUTES) -> pd.Series:
    """
    Return a session id per row, aligned to events_df's own index.

    A new session starts on a visitor's first event, or whenever the gap since
    their previous event exceeds gap_minutes.

    Aligned to the caller's index on purpose: evaluate_system.leave_last_out_split
    identifies held-out rows by index, so reindexing here would silently break
    the train/test split.

    Note on leakage: a session id depends only on the events at or before that
    row, so assigning ids before holding out each visitor's *last* event gives
    the same answer as assigning them after. Fit-time callers pass the training
    frame anyway.
    """
    ordered = events_df.sort_values(["visitorid", "timestamp"])
    gap = ordered.groupby("visitorid")["timestamp"].diff()
    starts_session = gap.isna() | (gap > gap_minutes * 60 * 1000)
    session_ids = starts_session.cumsum()
    return session_ids.reindex(events_df.index)


def with_sessions(events_df: pd.DataFrame, gap_minutes: float = SESSION_GAP_MINUTES) -> pd.DataFrame:
    """events_df plus a `sessionid` column."""
    out = events_df.copy()
    out["sessionid"] = assign_sessions(out, gap_minutes=gap_minutes)
    return out


def build_ordered_history_index(events_df: pd.DataFrame, weight_column: str = "weight") -> dict:
    """
    {visitorid: [(itemid, weight, last_timestamp), ...]} with one entry per
    distinct item, ordered oldest first so the customer's most recent interest
    is the last element.

    Order is the whole point: everything downstream that weights recent
    interactions more heavily reads position from this list, so an unordered
    groupby (which sorts by itemid) would silently disable position decay.
    """
    agg = (
        events_df.groupby(["visitorid", "itemid"])
        .agg(weight=(weight_column, "sum"), last_ts=("timestamp", "max"))
        .reset_index()
        .sort_values(["visitorid", "last_ts"])
    )
    return {
        visitorid: list(zip(group["itemid"], group["weight"], group["last_ts"]))
        for visitorid, group in agg.groupby("visitorid", sort=False)
    }


def profile_from_events(user_events, position_decay: float = 1.0, max_items: int = None,
                        current_session_only: bool = False,
                        gap_minutes: float = SESSION_GAP_MINUTES):
    """
    Turn an ordered event list into the [(itemid, weight), ...] profile the CF
    model scores against.

    user_events: [(itemid, weight, timestamp), ...] oldest first, one entry per
    distinct item (i.e. what build_ordered_history_index produces).

    position_decay is applied downstream in ItemBasedCF, *after* the history
    weight scheme, because that scheme may be "binary" — pre-multiplying decay
    here would then be thrown away. So this function only handles truncation
    and session scoping; ordering carries the recency information.

    current_session_only drops everything before the last gap of more than
    gap_minutes, i.e. keeps only what the customer is looking at right now.
    """
    events = list(user_events)
    if not events:
        return []

    if current_session_only and len(events) > 1:
        cutoff = 0
        threshold = gap_minutes * 60 * 1000
        for i in range(len(events) - 1, 0, -1):
            if events[i][2] - events[i - 1][2] > threshold:
                cutoff = i
                break
        events = events[cutoff:]

    if max_items:
        events = events[-max_items:]

    return [(item, weight) for item, weight, _ in events]


def flatten(user_events):
    """Drop timestamps: [(itemid, weight, ts), ...] -> [(itemid, weight), ...],
    preserving order."""
    return [(item, weight) for item, weight, *_ in user_events]
