"""
Item-based collaborative filtering.

Each item is represented as a sparse vector over users (who interacted with
it, and how strongly). Similarity between items is shrunk cosine similarity
over those vectors.

Rather than calling k-NN per lookup, the top-k neighbor list for every item is
precomputed once at fit time via a blocked sparse matrix product and stored as
a sparse (n_items x n_items) matrix with only k entries per row. Scoring a
customer is then a single sparse vector-matrix product instead of one
brute-force scan of the catalog per item in their history.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from data_loader import (
    build_interaction_matrix,
    _apply_weight_scheme,
    _apply_user_idf,
)


class ItemBasedCF:
    def __init__(
        self,
        k_neighbors: int = 20,
        weight_scheme: str = "bm25",
        use_idf: bool = True,
        shrinkage: float = 25.0,
        similarity_power: float = 1.0,
        history_weight_scheme: str = "log",
        max_user_items: int = 200,
        position_decay: float = 1.0,
        co_occurrence_unit: str = "visitorid",
        block_size: int = 1024,
    ):
        """
        k_neighbors        how many neighbors to keep per item
        weight_scheme      saturation applied to the item x user matrix before
                           similarity: raw | log | binary | bm25
        use_idf            downweight indiscriminate users (see _apply_user_idf)
        shrinkage          co-occurrence shrinkage strength; see _build_similarity.
                           0 disables it.
        similarity_power   exponent applied to similarities before scoring;
                           >1 sharpens (trust strong neighbors more)
        history_weight_scheme  saturation applied to the customer's own history
                           weights at scoring time: raw | log | binary
        max_user_items     drop users touching more than this many distinct
                           items from the similarity computation (bot filter);
                           None disables
        position_decay     how fast a history item's influence falls off as it
                           gets older: weight *= position_decay ** steps_back,
                           counting back from the customer's most recent item.
                           1.0 means "treat the history as an unordered bag"
                           (the original behaviour)
        co_occurrence_unit what makes two items related — "visitorid" (same
                           customer, ever) or "sessionid" (same browsing
                           session). See sessions.py for why the session
                           variant is not the free win it looks like here.
        block_size         rows per block in the similarity matmul (memory knob)
        """
        self.k_neighbors = k_neighbors
        self.weight_scheme = weight_scheme
        self.use_idf = use_idf
        self.shrinkage = shrinkage
        self.similarity_power = similarity_power
        self.history_weight_scheme = history_weight_scheme
        self.max_user_items = max_user_items
        self.position_decay = position_decay
        self.co_occurrence_unit = co_occurrence_unit
        self.block_size = block_size

        self.item_vectors = None       # csr_matrix, shape (n_items, n_users), raw weights
        self.sim_matrix = None         # csr_matrix, shape (n_items, n_items), top-k per row
        self.item_id_map = {}
        self.item_id_reverse = {}
        self.user_id_map = {}
        self.user_id_reverse = {}
        self.fitted = False

    # ------------------------------------------------------------------ fit

    def fit(self, events_df, min_user_interactions: int = 3, min_item_interactions: int = 3):
        matrix, user_id_map, item_id_map, item_id_reverse, user_id_reverse = build_interaction_matrix(
            events_df,
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
            unit=self.co_occurrence_unit,
        )

        self.user_id_map = user_id_map
        self.user_id_reverse = user_id_reverse
        self.item_id_map = item_id_map
        self.item_id_reverse = item_id_reverse

        # Transpose to item x user so each row is one item's vector.
        self.item_vectors = matrix.transpose().tocsr()
        self._build_similarity()
        self.fitted = True
        return self

    def _signal_matrix(self) -> csr_matrix:
        """The item x user matrix actually used for similarity, after bot
        filtering and weight transforms."""
        signal = self.item_vectors

        if self.max_user_items:
            # Users touching hundreds of distinct items are crawlers and
            # comparison bots, not shoppers. They co-occur with everything, so
            # they invent similarity between unrelated items. Drop their
            # columns from the similarity computation only — their history is
            # still served normally at recommend time.
            items_per_user = np.asarray((signal > 0).sum(axis=0)).ravel()
            drop = items_per_user > self.max_user_items
            if drop.any():
                # Mask through .indices rather than .multiply() with a
                # broadcast row — same result, ~5x faster and no densification.
                signal = signal.copy()
                signal.data[drop[signal.indices]] = 0.0
                signal.eliminate_zeros()

        signal = _apply_weight_scheme(signal, self.weight_scheme)
        if self.use_idf:
            signal = _apply_user_idf(signal)
        return signal

    def _build_similarity(self):
        """
        Precompute the top-k neighbor list for every item.

        Plain cosine on data this sparse is dominated by noise: the median item
        here has only a couple of users, so any two items that happen to share
        one single customer score a cosine near 1.0 and crowd out genuinely
        related items. Shrinkage fixes that by scaling each similarity by

            co / (co + shrinkage)

        where co is how many customers the two items actually share. A pair
        with 1 shared customer and shrinkage=25 keeps 4% of its score; a pair
        with 200 shared customers keeps 89%. Confidence, not just angle.
        """
        signal = self._signal_matrix()
        normalized = normalize(signal, norm="l2", axis=1).tocsr()

        binary = signal.copy()
        binary.data = np.ones_like(binary.data)
        binary = binary.tocsr()

        # Hoist the transposes out of the loop. `X.T` on a csr gives a csc,
        # which scipy re-converts to csr on every single matmul — that
        # conversion alone was a third of the total build time.
        normalized_t = normalized.T.tocsr()
        binary_t = binary.T.tocsr() if self.shrinkage else None

        n_items = normalized.shape[0]
        k = self.k_neighbors

        rows, cols, vals = [], [], []
        for start in range(0, n_items, self.block_size):
            stop = min(start + self.block_size, n_items)

            sim_block = (normalized[start:stop] @ normalized_t).tocsr()
            sim_block.sort_indices()

            if self.shrinkage:
                co_block = (binary[start:stop] @ binary_t).tocsr()
                co_block.sort_indices()
                # Both products come from non-negative operands over identical
                # sparsity patterns, so after sort_indices their data arrays
                # align element-for-element.
                if co_block.nnz == sim_block.nnz:
                    sim_block.data *= co_block.data / (co_block.data + self.shrinkage)
                else:  # pragma: no cover - defensive, patterns should match
                    sim_block = sim_block.multiply(
                        co_block / (co_block + self.shrinkage)
                    ).tocsr()

            # Drop self-similarity before taking the top k.
            self._zero_diagonal(sim_block, offset=start)
            sim_block.eliminate_zeros()

            r, c, v = self._topk_coo(sim_block, k, row_offset=start)
            rows.append(r)
            cols.append(c)
            vals.append(v)

        rows = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
        cols = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
        vals = np.concatenate(vals) if vals else np.array([], dtype=np.float64)

        if self.similarity_power != 1.0:
            vals = np.power(vals, self.similarity_power)

        self.sim_matrix = csr_matrix(
            (vals, (rows, cols)), shape=(n_items, n_items)
        )
        self.sim_matrix.sort_indices()

    @staticmethod
    def _zero_diagonal(block: csr_matrix, offset: int):
        """Zero out the (i, i) entry of a row-block of a square matrix. An item
        is always its own nearest neighbor, which would otherwise eat a slot."""
        row_of_entry = np.repeat(np.arange(block.shape[0]), np.diff(block.indptr)) + offset
        block.data[block.indices == row_of_entry] = 0.0

    @staticmethod
    def _topk_coo(block: csr_matrix, k: int, row_offset: int):
        """Keep only the k largest entries of each row; return COO triplets."""
        rows, cols, vals = [], [], []
        for local_row in range(block.shape[0]):
            start, stop = block.indptr[local_row], block.indptr[local_row + 1]
            if stop == start:
                continue
            data = block.data[start:stop]
            indices = block.indices[start:stop]
            if data.size > k:
                top = np.argpartition(-data, k)[:k]
                data = data[top]
                indices = indices[top]
            rows.append(np.full(indices.size, local_row + row_offset, dtype=np.int64))
            cols.append(indices.astype(np.int64))
            vals.append(data)

        if not rows:
            empty_i = np.array([], dtype=np.int64)
            return empty_i, empty_i, np.array([], dtype=np.float64)
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

    # -------------------------------------------------------------- queries

    def is_known_item(self, itemid) -> bool:
        return itemid in self.item_id_map

    def known_item_count(self, user_history: list) -> int:
        """How many items in this user's history does the model actually know?"""
        # Entries may be (itemid, weight) or (itemid, weight, timestamp).
        return sum(1 for entry in user_history if entry[0] in self.item_id_map)

    def similar_items(self, itemid, n: int = 10) -> list:
        """Return [(itemid, similarity_score), ...] for items similar to the given item."""
        if not self.fitted or itemid not in self.item_id_map:
            return []

        idx = self.item_id_map[itemid]
        start, stop = self.sim_matrix.indptr[idx], self.sim_matrix.indptr[idx + 1]
        data = self.sim_matrix.data[start:stop]
        indices = self.sim_matrix.indices[start:stop]

        order = np.argsort(-data)[:n]
        return [(self.item_id_reverse[int(indices[i])], float(data[i])) for i in order]

    def _history_weights(self, user_history: list):
        """
        Map a customer's history onto (column indices, transformed weights),
        dropping items the model doesn't know.

        user_history must be ordered oldest first, so the last entry is what
        the customer looked at most recently — position_decay reads recency
        from that ordering. Entries may be (itemid, weight) or
        (itemid, weight, timestamp); the timestamp is ignored here.
        """
        cols, weights = [], []
        for entry in user_history:
            item, weight = entry[0], entry[1]
            idx = self.item_id_map.get(item)
            if idx is None:
                continue
            cols.append(idx)
            weights.append(float(weight))

        if not cols:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        weights = np.asarray(weights, dtype=np.float64)
        if self.history_weight_scheme == "log":
            # Without this, one purchase (weight 10) outvotes nine views, so a
            # customer's single strongest signal effectively becomes their
            # whole profile.
            weights = 1.0 + np.log(np.maximum(weights, 1e-9))
            weights = np.maximum(weights, 1e-6)
        elif self.history_weight_scheme == "binary":
            weights = np.ones_like(weights)
        elif self.history_weight_scheme != "raw":
            raise ValueError(f"unknown history_weight_scheme: {self.history_weight_scheme!r}")

        if self.position_decay != 1.0:
            # Applied AFTER the weight scheme, not before: history_weight_scheme
            # may be "binary", which overwrites every weight with 1.0 and would
            # therefore erase any decay folded in upstream.
            steps_back = np.arange(len(weights) - 1, -1, -1, dtype=np.float64)
            weights = weights * (self.position_decay ** steps_back)

        return np.asarray(cols, dtype=np.int64), weights

    def score_items(self, user_history: list):
        """
        Raw candidate scores for a customer, as (item_indices, scores) in the
        model's internal index space, with already-seen items removed.

        Split out from recommend_for_history so callers that blend these with
        another signal (see sequence_rules.SequentialRules) can work with the
        full score vector instead of a truncated top-N list.

        Scored as one sparse product: profile (1 x n_items) @ sim_matrix
        (n_items x n_items). Equivalent to summing similarity * weight over
        every history item's neighbors, but done in one pass over only the
        stored top-k entries.
        """
        empty = (np.array([], dtype=np.int64), np.array([], dtype=np.float64))
        if not self.fitted:
            return empty

        cols, weights = self._history_weights(user_history)
        if cols.size == 0:
            return empty

        profile = csr_matrix(
            (weights, (np.zeros_like(cols), cols)), shape=(1, self.sim_matrix.shape[0])
        )
        scores = (profile @ self.sim_matrix).tocoo()
        if scores.nnz == 0:
            return empty

        mask = ~np.isin(scores.col, cols)
        return scores.col[mask], scores.data[mask]

    def recommend_for_history(self, user_history: list, n: int = 10) -> list:
        """
        user_history: [(itemid, weight), ...] ordered oldest first — the items
        a user has interacted with and how strongly. Returns [(itemid, score),
        ...] ranked recommendations, excluding items already in the history.
        """
        candidate_idx, candidate_val = self.score_items(user_history)
        if candidate_idx.size == 0:
            return []

        take = min(n, candidate_idx.size)
        top = np.argpartition(-candidate_val, take - 1)[:take]
        top = top[np.argsort(-candidate_val[top])]

        return [
            (self.item_id_reverse[int(candidate_idx[i])], float(candidate_val[i]))
            for i in top
        ]

    def similarity(self, item_a, item_b) -> float:
        """
        The stored similarity between two known items, read straight off the
        precomputed matrix.

        Note this is the *model's* similarity, not raw cosine: it carries
        whatever weight scheme, IDF, shrinkage and similarity_power the model
        was built with. It also returns 0.0 when the pair isn't in each
        other's top-k, since only those entries are kept.
        """
        if item_a not in self.item_id_map or item_b not in self.item_id_map:
            return 0.0
        idx_a = self.item_id_map[item_a]
        idx_b = self.item_id_map[item_b]
        return float(self.sim_matrix[idx_a, idx_b])

    def predict_score(self, user_history: list, target_item):
        """
        Predict how strongly a user would interact with target_item, using
        the classic item-based CF formula: a similarity-weighted average of
        the weights the user gave to items similar to the target.

            predicted = sum(sim(target, j) * weight(j)) / sum(|sim(target, j)|)

        over items j the user has interacted with that are also neighbors
        of target_item. Returns None if the target is unknown to the model
        or there's no overlap between the user's history and its neighbors
        (i.e. the prediction is undefined, not just zero).
        """
        if not self.fitted or target_item not in self.item_id_map:
            return None

        neighbors = dict(self.similar_items(target_item, n=self.k_neighbors))
        if not neighbors:
            return None

        numerator = 0.0
        denominator = 0.0
        for entry in user_history:
            # Entries may be (itemid, weight) or (itemid, weight, timestamp).
            item, weight = entry[0], entry[1]
            if item in neighbors:
                sim = neighbors[item]
                numerator += sim * weight
                denominator += abs(sim)

        if denominator == 0:
            return None
        return numerator / denominator
