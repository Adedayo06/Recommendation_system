"""
Human-readable labels for items and customers.

RetailRocket is an anonymized dataset: there are no real item names, brands,
or customer names anywhere in it (item_properties.csv has ~1000 hashed
numeric property codes, and visitorid is a bare id with nothing attached).
The one real, non-hashed piece of item metadata is `categoryid`, joined
against category_tree.csv for a bit of hierarchy. That's what we use to
label items here. Customers stay as plain ids since there's genuinely
nothing else on them.

In production, an actual client's catalog/customer DB would supply real
product names and customer names/emails — this module is where that lookup
would plug in instead.
"""

import os
from pathlib import Path

import pandas as pd

# Anchored to this file's location (not the current working directory), so
# these defaults work whether you run a script from src/, the project root,
# or anywhere else.
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / "data"

CATEGORY_PROPERTY_FILES_DEFAULT = [
    str(DATA_DIR / "raw" / "item_properties_part1.csv"),
    str(DATA_DIR / "raw" / "item_properties_part2.csv"),
]
CATEGORY_MAP_CACHE_DEFAULT = str(DATA_DIR / "processed" / "item_category_map.csv")


def load_category_map(paths=None, chunksize: int = 500_000, cache_path: str = None) -> dict:
    """
    Build {itemid: categoryid}. Reads from the cached
    data/processed/item_category_map.csv if present (built once via a fast
    grep-then-pandas pass over the raw files, since scanning 850MB+ of
    item_properties directly with pandas chunking is much slower). Falls
    back to a full pandas scan of the raw files if no cache exists yet.

    With no explicit cache_path, the cached map is fetched from S3 (and cached
    locally). Import is lazy so this module — which is only pulled in during
    training — never forces boto3 on the serving path.
    """
    if cache_path is None:
        from s3_data import fetch
        cache_path = fetch("processed/item_category_map.csv")

    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        return dict(zip(cached["itemid"], cached["categoryid"]))

    paths = paths or CATEGORY_PROPERTY_FILES_DEFAULT
    frames = []
    for path in paths:
        for chunk in pd.read_csv(path, usecols=["timestamp", "itemid", "property", "value"], chunksize=chunksize):
            cat_rows = chunk[chunk["property"] == "categoryid"]
            if not cat_rows.empty:
                frames.append(cat_rows[["timestamp", "itemid", "value"]])

    if not frames:
        return {}

    all_cats = pd.concat(frames, ignore_index=True)
    # Keep the most recent categoryid per item
    latest = all_cats.sort_values("timestamp").drop_duplicates("itemid", keep="last")
    cat_map = dict(zip(latest["itemid"], latest["value"].astype(int)))

    if cache_path:
        pd.DataFrame(list(cat_map.items()), columns=["itemid", "categoryid"]).to_csv(cache_path, index=False)

    return cat_map


def load_category_tree(path: str = str(DATA_DIR / "raw" / "category_tree.csv")) -> dict:
    """Build {categoryid: parentid} for optional hierarchy display."""
    df = pd.read_csv(path)
    return dict(zip(df["categoryid"], df["parentid"]))


def category_path(categoryid, tree: dict, max_depth: int = 5) -> str:
    """Walk up the category tree to show a bit of hierarchy, e.g. '1338 -> 213 -> 9'."""
    path = [str(categoryid)]
    current = categoryid
    for _ in range(max_depth):
        parent = tree.get(current)
        if parent is None or pd.isna(parent):
            break
        path.append(str(int(parent)))
        current = int(parent)
    return " -> ".join(path)


def item_label(itemid, category_map: dict, tree: dict = None) -> str:
    """e.g. 'Item 57245 (Category 1338 -> 213 -> 9)' or 'Item 12345 (Category unknown)'."""
    cat = category_map.get(itemid)
    if cat is None:
        return f"Item {itemid} (Category unknown)"
    if tree:
        return f"Item {itemid} (Category {category_path(cat, tree)})"
    return f"Item {itemid} (Category {cat})"


def customer_label(visitorid) -> str:
    """No real customer data exists in RetailRocket, so this is just a formatted id."""
    return f"Customer #{visitorid}"
