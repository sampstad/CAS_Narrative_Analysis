"""Search-driven article scoping.

When an entity and/or thematic query is active, we can restrict a narrative's
articles to those that match:

  * entity   -> articles whose slot-fillers include the matched entity labels
                (via entity_index → article ids)
  * thematic -> articles whose e5-small narrative embedding is similar enough to
                the query vector (per-article cosine, threshold)

Both yield a set of article ids. If both are active we intersect them.

Given a scoped article-id set, the stats row and the narrative card recompute
their numbers on just those articles instead of using the precomputed full
aggregates.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st

import data as D
import search_core as SC

THEMATIC_ARTICLE_SIM = float(os.environ.get("THEMATIC_ARTICLE_SIM", "0.85"))


# ---------------------------------------------------------------------------
# Building the scoped article-id set
# ---------------------------------------------------------------------------
def entity_article_ids(matched_entities: pd.DataFrame, slots: list | None = None) -> set | None:
    """Article ids whose extractions include any matched entity label, optionally
    restricted to specific slots. slots=None or [] means any slot.
    Returns None if the bridge is unavailable (no scoping)."""
    if matched_entities is None or matched_entities.empty:
        return None
    bridge = D.load_entity_article_bridge()
    if bridge is None:
        return None
    labels = set(matched_entities["label"])
    sub = bridge[bridge["label"].isin(labels)]
    if slots and "slot" in sub.columns:
        sub = sub[sub["slot"].isin(slots)]
    ids = sub["id"].astype(str)
    return set(ids) if len(ids) else set()


def thematic_article_ids(query_vec: np.ndarray, threshold: float = None) -> set | None:
    """Article ids whose e5-small narrative embedding is >= threshold similar to
    the query. Returns None if the article embeddings aren't available."""
    threshold = THEMATIC_ARTICLE_SIM if threshold is None else threshold
    pack = D.load_article_embeddings()
    if pack is None:
        return None
    emb, ids = pack
    sims = emb @ query_vec                    # both normalised → cosine
    keep = ids[sims >= threshold]
    return set(map(str, keep))