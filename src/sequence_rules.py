"""
Sequential rules: "customers who viewed A next viewed B".

Item-based CF asks a symmetric question — do A and B attract the same people?
It has no way to express direction. But a phone case follows a phone far more
often than a phone follows a case, and on a homepage that asymmetry is most of
the useful signal: knowing what someone just looked at tells you what comes
next, not merely what else is in the same neighbourhood.

This builds a directed transition matrix over item pairs that occur in the same
session, weighting each pair by how close together they were:

    count[a, b] += 1 / distance      for a occurring `distance` steps before b

so an immediately-following item counts full, one two steps back counts half,
and anything beyond `window` steps is ignored. Rows are then normalised so a
popular predecessor doesn't dominate purely by volume.

This is the "sequential rules" (SR) baseline from the session-based
recommendation literature, which is consistently hard to beat on short-session
e-commerce data — exactly what this dataset has (mean 1.33 items per session,
63% of active visitors with 2+ items in their last session).
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sessions import with_sessions


class SequentialRules:
    def __init__(self, window: int = 3, max_session_length: int = 50, normalize_rows: bool = True):
        """
        window              how many preceding items in the session can predict
                            the current one
        max_session_length  ignore pathological sessions (one here has 389
                            items) — they are crawlers, and the pair count
                            grows with the square of session length
        normalize_rows      scale each row to sum to 1, so "what usually
                            follows A" is a distribution rather than a raw
                            popularity count
        """
        self.window = window
        self.max_session_length = max_session_length
        self.normalize_rows = normalize_rows
        self.transitions = None       # csr_matrix (n_items, n_items)
        self.item_id_map = {}
        self.item_id_reverse = {}
        self.fitted = False

    def fit(self, events_df, item_id_map: dict, gap_minutes: float = 30):
        """
        item_id_map is passed in rather than derived so the transition matrix
        shares column indices with the CF similarity matrix — the two get
        blended per-item downstream, which requires a common index space.
        """
        self.item_id_map = item_id_map
        self.item_id_reverse = {idx: iid for iid, idx in item_id_map.items()}
        n_items = len(item_id_map)

        df = with_sessions(events_df[["timestamp", "visitorid", "itemid"]], gap_minutes=gap_minutes)
        df["col"] = df["itemid"].map(item_id_map)
        df = df.dropna(subset=["col"])
        df["col"] = df["col"].astype(np.int64)

        df = df.sort_values(["sessionid", "timestamp"])

        if self.max_session_length:
            size = df.groupby("sessionid")["col"].transform("size")
            df = df[size <= self.max_session_length]

        session = df["sessionid"].to_numpy()
        col = df["col"].to_numpy()

        rows, cols, vals = [], [], []
        for distance in range(1, self.window + 1):
            if len(col) <= distance:
                break
            # Pair each event with the one `distance` positions earlier, keeping
            # only pairs that fall inside the same session.
            same_session = session[distance:] == session[:-distance]
            predecessor = col[:-distance][same_session]
            successor = col[distance:][same_session]

            # Self-transitions (re-viewing the same item) carry no information
            # about what to recommend next.
            keep = predecessor != successor
            rows.append(predecessor[keep])
            cols.append(successor[keep])
            vals.append(np.full(keep.sum(), 1.0 / distance))

        if not rows:
            self.transitions = csr_matrix((n_items, n_items))
            self.fitted = True
            return self

        transitions = csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items),
        )
        transitions.sum_duplicates()

        if self.normalize_rows:
            row_sums = np.asarray(transitions.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            scale = 1.0 / row_sums
            transitions = transitions.multiply(scale[:, None]).tocsr()

        self.transitions = transitions
        self.fitted = True
        return self

    def score(self, user_history, position_decay: float = 0.7, max_items: int = 5) -> dict:
        """
        Score candidate next-items given an ordered history (oldest first, so
        the last entry is what the customer looked at most recently).

        Only the last `max_items` are consulted and each is discounted by
        position_decay ** steps_back: what someone viewed a minute ago predicts
        the next click far better than what they viewed in May.

        Returns {item_index: score} in the shared index space.
        """
        if not self.fitted or self.transitions.nnz == 0:
            return {}

        recent = [item for item, *_ in user_history][-max_items:]
        if not recent:
            return {}

        cols, weights = [], []
        for steps_back, item in enumerate(reversed(recent)):
            idx = self.item_id_map.get(item)
            if idx is None:
                continue
            cols.append(idx)
            weights.append(position_decay ** steps_back)

        if not cols:
            return {}

        profile = csr_matrix(
            (weights, (np.zeros(len(cols), dtype=np.int64), np.asarray(cols, dtype=np.int64))),
            shape=(1, self.transitions.shape[0]),
        )
        scores = (profile @ self.transitions).tocoo()
        return dict(zip(scores.col.tolist(), scores.data.tolist()))
