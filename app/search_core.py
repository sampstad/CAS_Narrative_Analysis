"""Query-time search logic, app-local (no pipeline dependencies).

These mirror the search functions in precompute_lib but live in the app so it
stays self-contained and importable from the app/ working directory. The
build-time counterparts (that created the embeddings) remain in precompute_lib.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

# e5 prefixes — must match how precompute embedded entities/narratives.
E5_QUERY_PREFIX   = "query: "
E5_PASSAGE_PREFIX = "passage: "

SEARCH_SLOTS = ["subject", "object", "sender", "receiver", "helper", "opponent"]
ENTITY_SIM_THRESHOLD = float(os.environ.get("SEARCH_ENTITY_THRESHOLD", "0.90"))
MAX_ENTITIES = int(os.environ.get("SEARCH_MAX_ENTITIES", "100"))


def embed_query_entity(query: str, model) -> np.ndarray:
    """Entity search: prefix-free, matching how entity labels were embedded."""
    return model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]


def embed_query_thematic(query: str, model) -> np.ndarray:
    """Thematic search: e5 query prefix, matching passage-prefixed centroids."""
    return model.encode([E5_QUERY_PREFIX + query],
                        convert_to_numpy=True, normalize_embeddings=True)[0]


def semantic_entity_search(query_vec, entity_embeddings, entity_vocab,
                           threshold=None, max_entities=None) -> pd.DataFrame:
    threshold = ENTITY_SIM_THRESHOLD if threshold is None else threshold
    max_entities = MAX_ENTITIES if max_entities is None else max_entities
    sims = entity_embeddings @ query_vec
    order = np.argsort(-sims)
    hits = []
    for i in order[:max_entities]:
        if sims[i] < threshold:
            break
        row = entity_vocab.iloc[i]
        hits.append({
            "label": row["label"], "similarity": float(sims[i]),
            "total_articles": int(row["total_articles"]),
            "n_clusters": int(row["n_clusters"]), "slots": row["slots"],
        })
    return pd.DataFrame(hits)
