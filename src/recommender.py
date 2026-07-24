"""
Hybrid recommender: item-based CF for users/items with enough signal,
popularity-based fallback for cold-start cases, and an optional sequential
("what tends to come next") signal blended on top.

This is the single entry point the API layer (api/services) should call.
"""

import numpy as np

from data_loader import build_recent_item_index
from item_cf import ItemBasedCF
from popularity import PopularityRecommender
from sequence_rules import SequentialRules
from sessions import (
    SESSION_GAP_MINUTES,
    build_ordered_history_index,
    profile_from_events,
    with_sessions,
)


def _normalize(values):
    """Scale to a 0..1 range so two signals on different scales can be mixed.
    CF scores are sums of similarities; sequential-rule scores are transition
    probabilities. Blending them raw would just mean whichever happens to be
    numerically larger wins."""
    if values.size == 0:
        return values
    peak = values.max()
    return values / peak if peak > 0 else values


class HybridRecommender:
    def __init__(self, min_known_interactions: int = 1, k_neighbors: int = 20,
                 sequence_weight: float = 0.0, sequence_window: int = 3,
                 sequence_position_decay: float = 0.7, sequence_max_items: int = 5,
                 current_session_only: bool = False, max_history_items: int = None,
                 session_gap_minutes: float = SESSION_GAP_MINUTES,
                 **cf_params):
        """
        min_known_interactions  minimum number of *model-known* items a user
            must have interacted with before we trust item-based CF for them.
            Below that (or zero), we fall back to popularity. Default is 1: a
            single known item still supports a real "customers who viewed this
            also viewed…" list, which beats global popularity by a wide margin.

        sequence_weight  how much of the final score comes from the directed
            "what tends to follow what" model rather than symmetric item
            similarity. 0.0 disables it entirely (and skips fitting it).

        current_session_only  score against only what the customer is looking
            at right now, discarding everything before their last 30-minute
            gap. Sharpens intent but throws away long-term taste.

        max_history_items  consider only this many of the customer's most
            recent items. None uses everything.

        Remaining keyword arguments (weight_scheme, shrinkage, use_idf,
        similarity_power, history_weight_scheme, max_user_items,
        position_decay, co_occurrence_unit) pass through to ItemBasedCF.
        """
        self.min_known_interactions = min_known_interactions
        self.item_cf = ItemBasedCF(k_neighbors=k_neighbors, **cf_params)
        self.popularity = PopularityRecommender()

        self.sequence_weight = sequence_weight
        self.sequence_position_decay = sequence_position_decay
        self.sequence_max_items = sequence_max_items
        self.sequence_rules = SequentialRules(window=sequence_window)

        self.current_session_only = current_session_only
        self.max_history_items = max_history_items
        self.session_gap_minutes = session_gap_minutes

        self.events_df = None
        self.item_category_map = None
        self.history_index = {}
        self.recent_item_index = {}
        self.fitted = False

    def fit(self, events_df, item_category_map: dict = None,
            min_user_interactions: int = 3, min_item_interactions: int = 3):
        self.events_df = events_df

        # Category scoping is on by default: if no map is passed in, load the
        # cached itemid -> categoryid lookup built in item_metadata.py.
        if item_category_map is None:
            from item_metadata import load_category_map
            item_category_map = load_category_map()
        self.item_category_map = item_category_map

        # Built once here instead of scanning the full event log per lookup.
        # Ordered chronologically — position decay depends on it.
        self.history_index = build_ordered_history_index(events_df)
        self.recent_item_index = build_recent_item_index(events_df)

        self.popularity.fit(events_df)
        if item_category_map:
            self.popularity.fit_categories(events_df, item_category_map)

        cf_events = events_df
        if self.item_cf.co_occurrence_unit == "sessionid":
            cf_events = with_sessions(events_df, gap_minutes=self.session_gap_minutes)

        self.item_cf.fit(
            cf_events,
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
        )

        if self.sequence_weight > 0:
            self.sequence_rules.fit(
                events_df, self.item_cf.item_id_map, gap_minutes=self.session_gap_minutes
            )

        self.fitted = True
        return self

    def _infer_category(self, user_id, user_history):
        """
        Best-effort guess at what category a customer currently cares about,
        used to scope the popularity fallback. Prefers the item from the
        customer's single most recent event; otherwise the highest-weight item
        in whatever history was handed in. Returns None if neither is
        available, which just means the fallback stays global.
        """
        if not self.item_category_map:
            return None

        if user_id is not None:
            recent_item = self.recent_item_index.get(user_id)
            if recent_item is not None:
                return self.item_category_map.get(recent_item)

        if user_history:
            top_item = max(user_history, key=lambda entry: entry[1])[0]
            return self.item_category_map.get(top_item)

        return None

    def _blended_scores(self, profile, n):
        """Combine symmetric item similarity with the directed sequence model."""
        cf_idx, cf_val = self.item_cf.score_items(profile)

        if self.sequence_weight <= 0 or not self.sequence_rules.fitted:
            return cf_idx, cf_val

        seq = self.sequence_rules.score(
            profile,
            position_decay=self.sequence_position_decay,
            max_items=self.sequence_max_items,
        )
        if not seq:
            return cf_idx, cf_val

        seen = {self.item_cf.item_id_map[item] for item, *_ in profile
                if item in self.item_cf.item_id_map}
        seq = {idx: score for idx, score in seq.items() if idx not in seen}
        if not seq:
            return cf_idx, cf_val

        cf_val = _normalize(cf_val)
        seq_idx = np.fromiter(seq.keys(), dtype=np.int64, count=len(seq))
        seq_val = _normalize(np.fromiter(seq.values(), dtype=np.float64, count=len(seq)))

        # Union the two candidate sets — the sequence model can surface items
        # that are not in any history item's top-k neighbourhood at all.
        merged = {}
        for idx, val in zip(cf_idx.tolist(), cf_val.tolist()):
            merged[idx] = (1.0 - self.sequence_weight) * val
        for idx, val in zip(seq_idx.tolist(), seq_val.tolist()):
            merged[idx] = merged.get(idx, 0.0) + self.sequence_weight * val

        idx = np.fromiter(merged.keys(), dtype=np.int64, count=len(merged))
        val = np.fromiter(merged.values(), dtype=np.float64, count=len(merged))
        return idx, val

    def recommend(self, user_id=None, user_history: list = None, n: int = 10, category: int = None):
        """
        Returns [(itemid, score, source), ...] where source is 'item_cf',
        'popularity_category', or 'popularity_global', so callers (and
        analytics/logging) can see which strategy actually served each
        recommendation.

        user_history entries may be (itemid, weight) or (itemid, weight,
        timestamp), and MUST be ordered oldest first — recency weighting reads
        position from that ordering. Pass it for a live/new customer whose
        events aren't in the training set yet; otherwise a known user_id is
        looked up in the prebuilt index.
        """
        if not self.fitted:
            raise RuntimeError("HybridRecommender.fit() must be called before recommend().")

        if user_history is None and user_id is not None:
            user_history = self.history_index.get(user_id, [])
        user_history = user_history or []

        profile = profile_from_events(
            [(entry[0], entry[1], entry[2] if len(entry) > 2 else 0) for entry in user_history],
            max_items=self.max_history_items,
            current_session_only=self.current_session_only,
            gap_minutes=self.session_gap_minutes,
        )

        seen_items = {item for item, *_ in user_history}
        known_count = self.item_cf.known_item_count(profile)

        results = []
        if known_count >= self.min_known_interactions:
            idx, val = self._blended_scores(profile, n)

            # Never recommend anything the customer has already interacted with
            # — the WHOLE history, not just the current-session profile. The
            # scoring functions only exclude what's in the profile, so with
            # current_session_only=True an item viewed in an earlier session
            # would otherwise be scored and handed straight back. Discovery
            # means showing new things; the "recently viewed" job belongs to a
            # separate backend row.
            seen_idx = np.fromiter(
                (self.item_cf.item_id_map[i] for i in seen_items if i in self.item_cf.item_id_map),
                dtype=np.int64,
            )
            if seen_idx.size and idx.size:
                keep = ~np.isin(idx, seen_idx)
                idx, val = idx[keep], val[keep]

            if idx.size:
                take = min(n, idx.size)
                top = np.argpartition(-val, take - 1)[:take]
                top = top[np.argsort(-val[top])]
                results = [
                    (self.item_cf.item_id_reverse[int(idx[i])], float(val[i]), "item_cf")
                    for i in top
                ]

        # Fill any remaining slots with popularity picks (covers cold-start
        # users entirely, AND tops up CF results that came up short because
        # the neighborhood was small).
        if len(results) < n:
            if category is None:
                category = self._infer_category(user_id, user_history)
            used_category = category is not None and category in self.popularity.category_ranking
            source_label = "popularity_category" if used_category else "popularity_global"

            already = seen_items | {item for item, _, _ in results}
            pop_fill = self.popularity.recommend(n=n - len(results), exclude=already, category=category)
            results += [(item, self.popularity.scores.get(item, 0.0), source_label) for item in pop_fill]

        return results[:n]
