"""
Model-serving layer for the API.

Loads the trained recommender once at startup and turns HTTP-friendly input
(a list of interactions) into the ordered history the model expects, then into
enriched recommendation output. Everything model-specific lives here so the
routers stay thin and the FastAPI layer never touches the pickle or the src/
modules directly.

The saved model deliberately does NOT carry a per-customer history index (that
would be a stale snapshot of the event log), so recommendations are served from
the history the caller supplies — exactly how a live service works, reading a
customer's recent activity from its own event store and passing it in.
"""

import sys
import time
from pathlib import Path

# The model modules live in src/; put them on the path before importing the
# persistence loader (which reconstructs the recommender from those classes).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import EVENT_WEIGHTS  # noqa: E402
from persistence import load_model  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "recommender.pkl"


class ModelService:
    """Holds the loaded model and serves recommendations. One instance per
    process, loaded on startup (see api/main.py)."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.rec = None
        self.loaded_at = None
        self.load_seconds = None

    # ---------------------------------------------------------------- lifecycle

    def load(self):
        started = time.time()
        self.rec = load_model(self.model_path)
        self.load_seconds = time.time() - started
        self.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return self

    @property
    def ready(self) -> bool:
        return self.rec is not None and self.rec.fitted

    @property
    def catalog_size(self) -> int:
        return len(self.rec.item_cf.item_id_map) if self.ready else 0

    def is_known_item(self, item_id: int) -> bool:
        return self.ready and item_id in self.rec.item_cf.item_id_map

    # ------------------------------------------------------------------ helpers

    def _category_of(self, item_id):
        cat = (self.rec.item_category_map or {}).get(item_id)
        return int(cat) if cat is not None else None

    def _to_history(self, interactions) -> list:
        """
        Convert API interactions into the model's [(item_id, weight, timestamp)]
        history, oldest first.

        Each interaction may carry an explicit `weight`, or an `event`
        (view/addtocart/transaction) that maps to one via the same table the
        model was trained with, defaulting to a view. If timestamps are given
        we sort by them; otherwise the caller's order is trusted as oldest-first.
        """
        rows = []
        for i, it in enumerate(interactions):
            if it.weight is not None:
                weight = float(it.weight)
            else:
                weight = float(EVENT_WEIGHTS.get(it.event, 1.0)) if it.event else 1.0
            ts = it.timestamp if it.timestamp is not None else i
            rows.append((int(it.item_id), weight, ts))

        rows.sort(key=lambda r: r[2])
        return rows

    # -------------------------------------------------------------- public API

    def recommend(self, interactions, n: int = 10, category: int = None) -> dict:
        history = self._to_history(interactions)
        raw = self.rec.recommend(user_history=history, n=n, category=category)

        items = [
            {
                "item_id": int(item_id),
                "score": float(score),
                "source": source,
                "category": self._category_of(item_id),
            }
            for item_id, score, source in raw
        ]
        # If any personalized (item_cf) result came back, the customer was warm;
        # otherwise this was a pure cold-start / popularity response.
        strategy = "personalized" if any(i["source"] == "item_cf" for i in items) else "cold_start"
        return {
            "recommendations": items,
            "count": len(items),
            "strategy": strategy,
            "known_history_items": self.rec.item_cf.known_item_count(
                [(item_id, w) for item_id, w, _ in history]
            ),
        }

    def similar_items(self, item_id: int, n: int = 10) -> dict:
        pairs = self.rec.item_cf.similar_items(int(item_id), n=n)
        return {
            "item_id": int(item_id),
            "category": self._category_of(int(item_id)),
            "similar": [
                {
                    "item_id": int(sim_id),
                    "similarity": float(sim),
                    "category": self._category_of(sim_id),
                }
                for sim_id, sim in pairs
            ],
            "count": len(pairs),
        }


# Process-wide singleton; api/main.py calls .load() on startup.
model_service = ModelService()
