"""
Popularity-based recommender.

Used as the cold-start fallback: new clients, new items, and users with too
little history for item-based CF to say anything reliable about them.

This is where recency decay belongs. "What is selling right now" is genuinely
time-sensitive in a way that "which items are alike" is not, so this ranker
reads the decayed `weight_recent` column while the CF model reads the raw
`weight` column.
"""

import pandas as pd


class PopularityRecommender:
    def __init__(self, weight_column: str = "weight_recent"):
        self.weight_column = weight_column
        self.ranking = []          # list of itemids, most popular first
        self.scores = {}           # itemid -> weighted popularity score
        self.category_ranking = {} # categoryid -> list of itemids (optional)

    def _column(self, events_df: pd.DataFrame) -> str:
        """Prefer the decayed column, but stay usable on a frame that only has
        a plain `weight` (e.g. a caller building events by hand)."""
        return self.weight_column if self.weight_column in events_df.columns else "weight"

    def fit(self, events_df: pd.DataFrame):
        """Compute a global recency-weighted popularity score per item."""
        column = self._column(events_df)
        pop = (
            events_df.groupby("itemid")[column]
            .sum()
            .sort_values(ascending=False)
        )
        self.scores = pop.to_dict()
        self.ranking = pop.index.tolist()
        return self

    def fit_categories(self, events_df: pd.DataFrame, item_category_map: dict):
        """
        Optional: build per-category popularity rankings so cold-start
        recommendations can at least be relevant to a category, not just
        globally popular. item_category_map: {itemid -> categoryid}.
        """
        column = self._column(events_df)
        scoped = events_df[["itemid", column]].copy()
        scoped["categoryid"] = scoped["itemid"].map(item_category_map)
        scoped = scoped.dropna(subset=["categoryid"])

        pop = (
            scoped.groupby(["categoryid", "itemid"])[column]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        self.category_ranking = {
            cat: group["itemid"].tolist()
            for cat, group in pop.groupby("categoryid", sort=False)
        }
        return self

    def recommend(self, n: int = 10, exclude: set = None, category: int = None) -> list:
        """Top-n itemids, optionally scoped to a category, excluding seen items."""
        exclude = exclude or set()

        if category is not None and category in self.category_ranking:
            candidates = self.category_ranking[category]
        else:
            candidates = self.ranking

        out = []
        for item in candidates:
            if item in exclude:
                continue
            out.append(item)
            if len(out) == n:
                break
        return out
