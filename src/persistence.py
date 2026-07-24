"""
Saving and loading a fitted recommender.

What gets saved is the *model*: the item-item similarity matrix, the transition
matrix, popularity rankings, the category map, and the configuration that
produced them. Fitting these takes several minutes; loading them takes seconds.

What deliberately does NOT get saved by default is the per-customer history
index. That is not model state — it is a snapshot of the event log, it is by
far the largest object in memory (one entry per visitor, 1.4M of them), and in
production it would be stale the moment you wrote it. A live service should
read customer history from its own event store and hand it to `recommend()` via
the `user_history` argument. Pass include_history=True only if you specifically
want a self-contained offline artifact.

The saved file also drops user_id_map / user_id_reverse from the CF model.
Those map training rows to matrix columns and are needed only while fitting;
nothing at serving time reads them, and they are ~259k entries.

Usage:
    from persistence import save_model, load_model
    save_model(rec, "../models/recommender.pkl")
    rec = load_model("../models/recommender.pkl")
"""

import pickle
import time
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


def _sparse_state(matrix, dtype=np.float32):
    """Serialize a CSR matrix as plain arrays, downcasting the values.

    float32 halves the file size and is far more precision than a similarity
    score needs — these values only ever get compared against each other for
    ranking, and float32 carries ~7 significant digits.
    """
    if matrix is None:
        return None
    matrix = matrix.tocsr()
    return {
        "data": matrix.data.astype(dtype),
        "indices": matrix.indices,
        "indptr": matrix.indptr,
        "shape": matrix.shape,
    }


def _restore_sparse(state):
    from scipy.sparse import csr_matrix

    if state is None:
        return None
    return csr_matrix(
        (state["data"], state["indices"], state["indptr"]), shape=state["shape"]
    )


def save_model(rec, path, include_history: bool = False) -> Path:
    """Write a fitted HybridRecommender to `path`. Returns the path written."""
    if not rec.fitted:
        raise RuntimeError("refusing to save an unfitted recommender — call fit() first")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cf = rec.item_cf
    state = {
        "format_version": FORMAT_VERSION,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "min_known_interactions": rec.min_known_interactions,
            "k_neighbors": cf.k_neighbors,
            "weight_scheme": cf.weight_scheme,
            "use_idf": cf.use_idf,
            "shrinkage": cf.shrinkage,
            "similarity_power": cf.similarity_power,
            "history_weight_scheme": cf.history_weight_scheme,
            "max_user_items": cf.max_user_items,
            "position_decay": cf.position_decay,
            "co_occurrence_unit": cf.co_occurrence_unit,
            "sequence_weight": rec.sequence_weight,
            "sequence_position_decay": rec.sequence_position_decay,
            "sequence_max_items": rec.sequence_max_items,
            "current_session_only": rec.current_session_only,
            "max_history_items": rec.max_history_items,
            "session_gap_minutes": rec.session_gap_minutes,
        },
        "item_cf": {
            "sim_matrix": _sparse_state(cf.sim_matrix),
            "item_vectors": _sparse_state(cf.item_vectors),
            "item_id_map": cf.item_id_map,
        },
        "popularity": {
            "ranking": rec.popularity.ranking,
            "scores": rec.popularity.scores,
            "category_ranking": rec.popularity.category_ranking,
            "weight_column": rec.popularity.weight_column,
        },
        "sequence_rules": {
            "transitions": _sparse_state(rec.sequence_rules.transitions),
            "window": rec.sequence_rules.window,
            "fitted": rec.sequence_rules.fitted,
        },
        "item_category_map": rec.item_category_map,
        "history_index": rec.history_index if include_history else None,
        "recent_item_index": rec.recent_item_index if include_history else None,
    }

    with open(path, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def load_model(path):
    """Rebuild a HybridRecommender from disk, ready to serve recommendations."""
    from recommender import HybridRecommender

    with open(path, "rb") as handle:
        state = pickle.load(handle)

    version = state.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path} was written by format version {version}, this code expects "
            f"{FORMAT_VERSION}. Re-run train_model.py to regenerate it."
        )

    config = dict(state["config"])
    rec = HybridRecommender(
        min_known_interactions=config.pop("min_known_interactions"),
        k_neighbors=config.pop("k_neighbors"),
        sequence_weight=config.pop("sequence_weight"),
        sequence_window=state["sequence_rules"]["window"],
        sequence_position_decay=config.pop("sequence_position_decay"),
        sequence_max_items=config.pop("sequence_max_items"),
        current_session_only=config.pop("current_session_only"),
        max_history_items=config.pop("max_history_items"),
        session_gap_minutes=config.pop("session_gap_minutes"),
        **config,
    )

    cf = rec.item_cf
    cf.sim_matrix = _restore_sparse(state["item_cf"]["sim_matrix"])
    cf.item_vectors = _restore_sparse(state["item_cf"]["item_vectors"])
    cf.item_id_map = state["item_cf"]["item_id_map"]
    cf.item_id_reverse = {idx: iid for iid, idx in cf.item_id_map.items()}
    cf.fitted = True

    rec.popularity.ranking = state["popularity"]["ranking"]
    rec.popularity.scores = state["popularity"]["scores"]
    rec.popularity.category_ranking = state["popularity"]["category_ranking"]
    rec.popularity.weight_column = state["popularity"]["weight_column"]

    rec.sequence_rules.transitions = _restore_sparse(state["sequence_rules"]["transitions"])
    rec.sequence_rules.item_id_map = cf.item_id_map
    rec.sequence_rules.item_id_reverse = cf.item_id_reverse
    rec.sequence_rules.fitted = state["sequence_rules"]["fitted"]

    rec.item_category_map = state["item_category_map"]
    rec.history_index = state.get("history_index") or {}
    rec.recent_item_index = state.get("recent_item_index") or {}
    rec.fitted = True

    return rec


def attach_history(rec, events_df):
    """
    Rebuild the per-customer lookup indexes on a loaded model.

    Only needed if you want to call recommend(user_id=...) and let the model
    look the customer up itself. Serving from a live event store instead means
    passing recommend(user_history=...) directly, and this is unnecessary.
    """
    from data_loader import build_recent_item_index
    from sessions import build_ordered_history_index

    rec.events_df = events_df
    rec.history_index = build_ordered_history_index(events_df)
    rec.recent_item_index = build_recent_item_index(events_df)
    return rec
