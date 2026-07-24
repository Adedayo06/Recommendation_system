"""
Data loading and interaction-matrix construction for the recommender.

RetailRocket events.csv schema: timestamp, visitorid, event, itemid, transactionid
event is one of: view, addtocart, transaction.
"""

import math

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Implicit-feedback weighting: a transaction is a much stronger signal than a
# view. Widened relative to the original 1/3/5 because views outnumber
# transactions by roughly 100-to-1 in this data — without a bigger gap, sheer
# view volume drowns out the much rarer, much more meaningful purchase signal.
EVENT_WEIGHTS = {
    "view": 1,
    "addtocart": 4,
    "transaction": 10,
}

# Recency decay half-life in days. NOTE: this now only feeds the *popularity*
# ranking (see `weight_recent` below), where "what's hot right now" is exactly
# what we want. It is deliberately NOT applied to the collaborative-filtering
# matrix — see the comment on load_events().
RECENCY_HALF_LIFE_DAYS = 30

# BM25 parameters (used when weight_scheme="bm25"). K1 controls how fast repeat
# interactions saturate; B controls how hard we normalize for "document" length
# (here, how many users an item has).
BM25_K1 = 1.2
BM25_B = 0.8


def load_events(path: str, half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> pd.DataFrame:
    """
    Load raw events.csv and attach TWO weight columns:

        weight         event-type weight, undecayed. This is what the
                       collaborative-filtering matrix and per-customer
                       histories use.
        weight_recent  the same weight scaled by exponential recency decay.
                       This is what the popularity ranking uses.

    Why the split: this dataset spans 138 days and the half-life is 30 days, so
    a global decay scales the oldest events by 0.5 ** (138/30) ≈ 0.04. Applying
    that to the CF matrix throws away ~60% of the co-occurrence signal — the
    similarity between two items ends up determined almost entirely by whoever
    happened to interact in the final few weeks, and *when* two users acted is
    noise as far as "are these items alike?" is concerned. Item-item taste
    correlations are stable over a 4-month window; item popularity is not.
    So: decay popularity, don't decay similarity.
    """
    df = pd.read_csv(path, usecols=["timestamp", "visitorid", "event", "itemid"])
    df["weight"] = df["event"].map(EVENT_WEIGHTS).fillna(1).astype(float)
    df["weight_recent"] = df["weight"] * _recency_factor(df, half_life_days=half_life_days)
    return df


def _recency_factor(df: pd.DataFrame, half_life_days: float) -> np.ndarray:
    """Exponential decay factor in (0, 1]: 0.5 ** (age_in_days / half_life_days).
    'Now' is the most recent timestamp in the dataset (this is a historical
    snapshot, not a live feed), so the newest events get no discount and
    everything else is scaled down by how old it is relative to that point."""
    reference_time = df["timestamp"].max()
    age_days = (reference_time - df["timestamp"]) / (1000 * 60 * 60 * 24)  # timestamp is epoch ms
    decay_lambda = math.log(2) / half_life_days
    return np.exp(-decay_lambda * age_days)


def _apply_weight_scheme(matrix: csr_matrix, scheme: str) -> csr_matrix:
    """
    Saturate raw summed weights so a single obsessive user can't dominate.

    A customer who viewed one item 40 times contributes a raw weight of 40,
    which under cosine similarity swamps the 39 other customers who viewed it
    once. Every scheme here except "raw" compresses that.

        raw     leave summed weights alone
        log     1 + log(w) — gentle saturation, keeps the view/cart/buy ordering
        binary  did they interact at all — discards strength entirely
        bm25    log-ish saturation plus per-item length normalization
    """
    matrix = matrix.copy()
    if scheme == "raw":
        return matrix
    if scheme == "binary":
        matrix.data = np.ones_like(matrix.data)
        return matrix
    if scheme == "log":
        matrix.data = 1.0 + np.log(matrix.data)
        return matrix
    if scheme == "bm25":
        return _bm25_weight(matrix)
    raise ValueError(f"unknown weight_scheme: {scheme!r}")


def _bm25_weight(matrix: csr_matrix, k1: float = BM25_K1, b: float = BM25_B) -> csr_matrix:
    """
    BM25 over an (items x users) matrix: items are 'documents', users are
    'terms'. Combines saturation of repeat interactions with normalization for
    how many users an item has, so a blockbuster item doesn't get an unfair
    similarity advantage over a niche one.
    """
    matrix = matrix.tocsr().astype(np.float64)
    n_items = matrix.shape[0]

    # Per-item "document length" = total interaction weight on that item.
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    avg_len = row_sums.mean() if row_sums.size else 1.0
    avg_len = avg_len if avg_len > 0 else 1.0
    length_norm = (1.0 - b) + b * (row_sums / avg_len)

    # Expand the per-row length norm to align with matrix.data.
    row_index = np.repeat(np.arange(n_items), np.diff(matrix.indptr))
    denom = matrix.data + k1 * length_norm[row_index]
    matrix.data = matrix.data * (k1 + 1.0) / np.where(denom == 0, 1.0, denom)
    return matrix


def _apply_user_idf(matrix: csr_matrix) -> csr_matrix:
    """
    Inverse-user-frequency on an (items x users) matrix.

    One visitor in this dataset touched 3,814 distinct items. A user like that
    co-occurs with almost everything, so they manufacture spurious similarity
    between completely unrelated items. Weighting each user column by
    log(n_items / (1 + items_touched)) makes a shared *selective* customer
    count for much more than a shared indiscriminate one.
    """
    matrix = matrix.tocsc().astype(np.float64)
    n_items = matrix.shape[0]

    items_per_user = np.diff(matrix.indptr)  # nnz per column = items touched
    idf = np.log(1.0 + n_items / (1.0 + items_per_user))

    col_index = np.repeat(np.arange(matrix.shape[1]), items_per_user)
    matrix.data = matrix.data * idf[col_index]
    return matrix.tocsr()


def build_interaction_matrix(
    events_df: pd.DataFrame,
    min_user_interactions: int = 3,
    min_item_interactions: int = 3,
    weight_column: str = "weight",
    max_filter_passes: int = 10,
    unit: str = "visitorid",
):
    """
    Collapse raw events into a (unit x item) sparse matrix of summed weights,
    after dropping units/items that are too sparse to model reliably (those
    are handled by the popularity fallback instead).

    `unit` chooses what counts as "co-occurrence": "visitorid" treats two items
    as related if the same customer touched both ever; "sessionid" requires
    them to appear in the same browsing session (events_df must already carry
    a sessionid column — see sessions.with_sessions). With unit="sessionid",
    min_user_interactions means "minimum distinct items per session".

    See the measurements in sessions.py before reaching for "sessionid" here:
    on this dataset 52% of active visitors only ever have one session, so the
    two settings are identical for most customers and the session variant
    mainly just discards the cross-session links of the rest.

    Filtering runs to a fixed point rather than in a single pass: dropping
    sparse items pushes some users below the user threshold, which drops more
    items, and so on. A single pass leaves a tail of under-support rows behind
    that go on to generate unreliable similarities.

    Returns:
        matrix: csr_matrix, shape (n_users, n_items)
        user_id_map: dict {visitorid -> row index}
        item_id_map: dict {itemid -> col index}
        item_id_reverse: dict {col index -> itemid}
        user_id_reverse: dict {row index -> visitorid}
    """
    if unit not in ("visitorid", "sessionid"):
        raise ValueError(f"unknown unit: {unit!r}")
    if unit == "sessionid" and "sessionid" not in events_df.columns:
        raise ValueError("unit='sessionid' requires a sessionid column (see sessions.with_sessions)")

    # Aggregate duplicate (unit, item) events into one weighted score
    agg = (
        events_df.groupby([unit, "itemid"])[weight_column]
        .sum()
        .reset_index()
        .rename(columns={weight_column: "weight"})
    )

    # Iterate the co-support filter to a fixed point.
    for _ in range(max_filter_passes):
        item_counts = agg.groupby("itemid")[unit].transform("count")
        user_counts = agg.groupby(unit)["itemid"].transform("count")
        keep = (item_counts >= min_item_interactions) & (user_counts >= min_user_interactions)
        if keep.all():
            break
        agg = agg[keep]
        if agg.empty:
            break

    user_ids = agg[unit].unique()
    item_ids = agg["itemid"].unique()

    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}
    item_id_reverse = {idx: iid for iid, idx in item_id_map.items()}
    user_id_reverse = {idx: uid for uid, idx in user_id_map.items()}

    rows = agg[unit].map(user_id_map)
    cols = agg["itemid"].map(item_id_map)
    vals = agg["weight"].astype(float)

    matrix = csr_matrix(
        (vals, (rows, cols)), shape=(len(user_ids), len(item_ids))
    )

    return matrix, user_id_map, item_id_map, item_id_reverse, user_id_reverse


# NOTE: the per-customer history index lives in sessions.build_ordered_history_index,
# not here. It has to emit items in chronological order for position decay to
# mean anything, and a plain groupby sorts by itemid instead — which would
# disable recency weighting silently rather than loudly.


def build_recent_item_index(events_df: pd.DataFrame) -> dict:
    """{visitorid: itemid of their single most recent event}, built in one pass.
    Same motivation as build_history_index — avoids a full scan per lookup."""
    idx = events_df.groupby("visitorid")["timestamp"].idxmax()
    recent = events_df.loc[idx, ["visitorid", "itemid"]]
    return dict(zip(recent["visitorid"], recent["itemid"]))


def get_user_history(events_df: pd.DataFrame, visitorid: int, weight_column: str = "weight") -> list:
    """Return [(itemid, weight), ...] of everything a given user has interacted with.
    Scans the full log — prefer build_history_index() when serving many lookups."""
    sub = events_df[events_df["visitorid"] == visitorid]
    if sub.empty:
        return []
    agg = sub.groupby("itemid")[weight_column].sum().reset_index()
    return list(agg.itertuples(index=False, name=None))


def get_most_recent_item(events_df: pd.DataFrame, visitorid: int):
    """
    The item from a customer's single most recent event (by timestamp).
    Used to infer what category a customer is currently interested in, so
    the popularity fallback can be scoped instead of falling back to pure
    global popularity. Returns None if the customer has no events.
    """
    sub = events_df[events_df["visitorid"] == visitorid]
    if sub.empty:
        return None
    return sub.loc[sub["timestamp"].idxmax(), "itemid"]
