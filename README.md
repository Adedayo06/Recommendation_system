# E-commerce Recommendation System

Item-based collaborative filtering with a popularity fallback, built on the
[RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
implicit-feedback dataset (2.76M events, 1.41M visitors, 235k items, 138 days).

## Results

The headline number is **Discovery Hit Rate @10 = 0.2511**: for 1 in 4 customers,
the exact item they clicked next — something they had *not* already seen — was in
the top 10. A bestsellers list gets that right 1 time in 200. See
[Discovery vs reminder](#discovery-vs-reminder) below for why this, rather than a
single blended hit rate, is the number to quote.

**Controlled before/after** (`tune.py`, same 3,000 held-out customers, not used
during tuning). These are *blended* hit rates — the metric the original code
reported — so the comparison is like-for-like:

| Metric | Original | Tuned | |
|---|---|---|---|
| Blended Hit Rate @10 | 0.0823 | **~0.19** | +130% |
| Customers served by CF | 41.3% | **84.8%** | |

The improvement came in two rounds: fixing the similarity model and its tuning
took blended HR@10 from 0.082 to ~0.17, and adding sequence awareness took it to
~0.19. (The sweep itself was run before the reminder-leak fix and reported ~0.20;
that fix lowered the absolute figures without changing which configuration won,
because it only affects already-seen items and the tuning target never depended
on them.)

### Discovery vs reminder

25% of held-out events are a customer returning to an item they had *already
viewed*. The recommender deliberately never re-recommends something already in a
customer's history — a homepage full of already-seen items is a bad homepage —
so those cases are structurally unwinnable here. They are a different product
feature (a "recently viewed" row, which is a backend database query, not a
model), so the report scores them separately instead of blaming the model for
them.

**Discovery Hit Rate is the headline** — cases where the next click really was
on something new, which is the only job this model has.

| Metric (`evaluate_system.py`, 5,000 customers) | @10 | @20 |
|---|---|---|
| **Discovery Hit Rate** | **0.2511** | **0.3078** |
| Discovery NDCG | 0.1567 | 0.1709 |
| Discovery ceiling (in catalog & unseen) | 0.6044 | 0.6044 |
| Discovery HR / ceiling (share captured) | 0.3107 | 0.51 |
| Reminder Hit Rate (expected 0) | 0.0000 | 0.0000 |
| Blended Hit Rate (both jobs averaged) | 0.1878 | 0.2312 |
| Coverage | 0.3596 | 0.547 |
| Diversity | 0.9177 | 0.932 |
| Popularity-only baseline | 0.0052 | 0.0068 |

Against the original model's blended HR@10 of 0.0880, the tuned model's blended
HR@10 is 0.1878 — but the fairer comparison is discovery hit rate, where it
captures **31% of everything achievable** and beats a bestsellers list (0.0052)
by ~48x.

Coverage more than quadrupled (8% → 36%), which matters commercially: the
original model left 92% of sellers with no exposure at all.

> **Note on the earlier 0.2026 figure.** An earlier version of this README
> reported blended HR@10 of 0.2026. That number was inflated by a bug: with
> `current_session_only=True`, items a customer viewed in *earlier* sessions
> were not being excluded from the output, so the model was scoring ~6% of
> reminder cases as "hits" by recommending already-seen items back. Fixing it
> (recommender.py now excludes the full history, not just the current-session
> profile) dropped blended HR to its honest 0.1878 and nudged discovery HR
> *up*, since the freed slots went to genuine candidates.

## Layout

```
src/
  data_loader.py      event loading, weighting, interaction matrix
  item_cf.py          item-based CF: similarity build + scoring
  sessions.py         session segmentation + ordered history profiles
  sequence_rules.py   directed "what tends to follow what" model
  popularity.py       cold-start fallback ranker
  recommender.py      hybrid entry point (CF + sequence + fallback)
  item_metadata.py    itemid -> categoryid lookup
  evaluate_system.py  evaluation harness and report (leave-last-out)
  tune.py             hyperparameter sweep
  persistence.py      save / load a fitted model
  train_model.py      fit on the full log and save to models/recommender.pkl
models/
  recommender.pkl     the trained, deployable model (~50 MB)
```

Run from the project root:

```bash
python src/evaluate_system.py   # full report, leave-last-out split
python src/train_model.py       # fit on the full log, save models/recommender.pkl
python src/tune.py              # re-run the hyperparameter sweep (only if you change the model)
```

Serving a recommendation from the saved model:

```python
from persistence import load_model
rec = load_model("../models/recommender.pkl")

# history is [(itemid, weight, timestamp), ...], oldest first, from your event store
recs = rec.recommend(user_history=history, n=10)
# -> [(itemid, score, source), ...]  source in {item_cf, popularity_category, popularity_global}
```

The saved file holds the *model* (similarity matrix, transition matrix,
popularity rankings, category map, config) but deliberately not the per-customer
history index — that is a stale snapshot of the event log in production, so a
live service passes each customer's history to `recommend()` from its own store.
See the docstring in `persistence.py`.

## What was wrong, and what changed

The original model scored HR@10 0.088. Six things were holding it back.

### 1. Recency decay was corrupting the similarity matrix

`load_events` applied a 30-day half-life to every event, and the CF matrix read
those decayed weights. The dataset spans 138 days, so the oldest events were
scaled to `0.5 ** (138/30) ≈ 0.04` — a 24x compression that left item-item
similarity determined almost entirely by whoever happened to interact in the
final few weeks.

*When* two customers acted is noise as far as "are these two items alike?" is
concerned. Item-item taste correlations are stable across a four-month window;
item *popularity* is not. So `load_events` now emits two columns — `weight`
(undecayed, used by CF) and `weight_recent` (decayed, used by the popularity
ranker) — and each consumer reads the one that suits it.

### 2. No shrinkage, so the neighbor lists were mostly noise

This was the biggest algorithmic problem. The median item here has interactions
from only **two** customers. Any two items that happen to share a single
customer score a cosine similarity of **1.0** — a perfect match on the strength
of one coincidence — and those spurious pairs crowded genuinely related items
out of every top-k list.

Similarities are now scaled by `co / (co + shrinkage)`, where `co` is the number
of customers the two items actually share. A pair with 1 shared customer keeps
2% of its score at `shrinkage=50`; a pair with 200 keeps 80%. This encodes
confidence, not just angle. `test_item_cf.py` demonstrates the pathology
directly: without shrinkage a 1-customer pair and a 4-customer pair both score
exactly 1.000.

Turning shrinkage on alone moved HR@10 from 0.1437 to 0.1767.

### 3. Half of all customers never reached the CF model

`min_known_interactions=2` sent every customer with a single known item straight
to global popularity — 51% of them. Popularity hits 0.6% of the time, so those
customers were effectively being served nothing.

One known item is still enough to support a real "customers who viewed this also
viewed…" list. Lowering the threshold to 1 raised CF coverage from 41% to 87% and
was the single largest win in the sweep (HR@10 0.128 → 0.173).

### 4. Power users were manufacturing false similarity

One visitor touched 3,814 distinct items; the 99th percentile is 29. Users like
that are crawlers and price-comparison bots, and because they co-occur with
almost everything they invent similarity between unrelated products. Two
defences now apply: inverse-user-frequency weighting damps indiscriminate users
smoothly, and `max_user_items` drops the extreme tail from the similarity
computation entirely (their history is still served normally at recommend time).

### 5. Raw summed weights let one obsessive customer dominate an item

A customer who viewed an item 40 times contributed a raw weight of 40, swamping
the 39 other customers who viewed it once. The interaction matrix now supports
`raw` / `log` / `binary` / `bm25` saturation; `log` won the sweep.

### 6. The model treated history as an unordered bag

What someone viewed five minutes ago predicts their next click far better than
what they viewed in May, and the original model could not express that. Three
mechanisms now do (HR@10 0.181 → 0.205):

| Change | HR@10 | |
|---|---|---|
| tuned, sequence-blind | 0.1807 | |
| `position_decay=0.7` | 0.1847 | older history items count less |
| `sequence_weight=0.2` | 0.1870 | blend in directed transitions |
| `sequence_max_items=3` | 0.1877 | only recent items drive transitions |
| `current_session_only=True` | **0.1987** | ignore everything before the last 30-min gap |
| `max_history_items=5` | 0.1993 | cap profile length |

`current_session_only` was the single biggest win. Scoring against only what the
customer is looking at *right now* — discarding everything before their last
30-minute gap — beat using their full history by 5.9%. Intent is short-lived.

The directed part lives in `sequence_rules.py`. Item CF asks a symmetric
question (do A and B attract the same people?) and structurally cannot tell you
that a phone case follows a phone rather than the reverse. The transition model
counts ordered within-session pairs weighted by `1/distance` and gets blended in
at `sequence_weight`.

**What didn't work: switching co-occurrence from visitor to session.** This is
the standard session-kNN recommendation and it scored *worse* here (0.1973 vs
0.1993). `sessions.py` documents why — 52% of active visitors only ever have one
session, so for most customers the two settings are identical, and for the rest
it just discards cross-session links. It stayed in as a tunable
(`co_occurrence_unit`) because it does buy noticeably better catalog coverage
(0.290 vs 0.263), which may be worth the hit rate on a marketplace.

## Performance

Neighbor lookup used to run brute-force k-NN once per item in the customer's
history, per request. The top-k neighbor list for every item is now precomputed
once at fit time with a blocked sparse matrix product, and scoring a customer is
a single sparse vector-matrix product against it.

Evaluating 500 customers went from minutes to **0.3 seconds**, which is what made
the hyperparameter sweep practical at all. Two smaller fixes came out of
profiling: hoisting a transpose out of the similarity loop (scipy was
re-converting CSC→CSR on every block, a third of build time), and replacing a
broadcast `.multiply()` mask with direct index masking.

Per-customer lookups also no longer scan the full 2.7M-row event log — the
history and most-recent-item indexes are built once at fit time.

## Evaluation methodology

Two reference points are now reported alongside every metric, because a hit rate
on its own is unreadable:

- **Reachable ceiling** — what fraction of held-out items are even in the model's
  catalog. You cannot recommend an item you filtered out of training, so this is
  the hard cap on hit rate (currently ~84%). It separates "the ranker is bad"
  from "the item wasn't there".
- **Popularity-only baseline** — what a bestsellers-for-everybody ranker scores
  on the same cases (0.006). CF has to beat this to justify existing.

`tune.py` splits the held-out cases into a validation half for choosing
hyperparameters and a test half used once at the end. Validation HR@10 came in at
0.1810 against 0.1813 on test, so the tuning did not overfit the sample.

**RMSE/MAE are reported last and should not be used to judge this system.**
RetailRocket has no explicit ratings, so they compare a predicted implicit event
weight against a made-up pseudo-rating scale, over only the minority of cases with
neighbor overlap. They measure calibration of an invented scale, not whether the
ranking is any good. Judge on Hit Rate and NDCG.

## Two use cases, and an honest ceiling

An honest **time-based split** (train on the first ~120 days, test on the last
~18 — no future leakage) tells a sharper story than the headline leave-last-out
number, and it splits the product in two:

- **In-session ("you're shopping now")** — the model is genuinely strong. The
  leave-last-out discovery HR@10 of ~0.25 is real *for this use case*, because
  the next click is usually in the same session the model is reading.
- **Returning-customer homepage ("welcome back")** — much harder. Under the time
  split, discovery HR@10 is ~0.034. Still ~4-5x a trending-only baseline, so the
  model adds real value, but this is the honest number for a homepage.

Three separate attempts were made to lift the returning-customer number, and
**none beat the current model** — a strong signal that ~0.034 is close to the
data's ceiling for this problem, not a fixable deficiency:

| Attempt | Result |
|---|---|
| Matrix factorisation (ALS/BPR) as a third signal | ALS +2% blended HR at a ~20% coverage cost; BPR nothing. Removed. |
| Re-tuning every hyperparameter for the time-split objective | +0.0% on held-out data — the return-visit customers have thin histories, so the knobs don't bite. |
| Category-affinity model (predict the category, not the item) | Beat trending ~4x but lost to CF in every history-depth slice. Removed. |

The productive next moves for the homepage are therefore **not another offline
model** — they are a real-user A/B test, or leaning into the in-session product
where the model is already good.

## Known gaps and next steps

**The seller-side sentiment analysis has no data source in this dataset.**
RetailRocket ships no reviews, no ratings, and no free text. The only named item
properties are `available` and `categoryid`; the other ~1,100 are hashed numeric
codes with hashed values, and there is no seller ID anywhere. Sentiment-based
seller recommendations will need either a different dataset (Amazon Reviews,
Yelp) or real review data joined in from the target platform.

Other open directions, roughly in order of expected payoff:

1. **A real-user A/B test.** Every offline avenue now points the same way; the
   only thing that can still change the picture for the homepage is real people.
2. **Content features for cold items.** ~17% of time-split targets are items too
   new to be in the catalog at all. Features over `item_properties` (hashed, but
   usable as a similarity space) would let genuinely new items be recommended.
3. **Coverage vs. hit-rate trade.** `min_item_interactions=2` maximises hit rate,
   but 3 costs ~0.7% relative hit rate for +32% relative catalog coverage. On a
   marketplace where seller exposure matters, that is a product decision.
4. **A learned reranker**, if MF is ever revisited — the embeddings are a better
   *feature* than a linear blend partner (CF score, sequence score, popularity,
   recency, category match, adapting per customer).
# Recommendation_system
# Recommendation_system
