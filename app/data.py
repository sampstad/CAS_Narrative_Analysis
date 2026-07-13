"""Cached data-loading layer.

Every view reads through here. Tables load once via st.cache_data; the embedding
model loads once via st.cache_resource (persists across Streamlit reruns).

Paths point at the parquet/npy artifacts produced by 07_precompute.ipynb.
Set APP_DATA_DIR env var to override the default location.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st

# Resolve data location robustly regardless of the working directory.
# Priority: (1) APP_DATA_DIR env var, (2) <repo_root>/data/viz where repo_root is
# the parent of the app/ folder this file lives in, (3) ./data/viz as a fallback.
_APP_DIR   = os.path.dirname(os.path.abspath(__file__))      # .../app
_REPO_ROOT = os.path.dirname(_APP_DIR)                        # repo root
_DEFAULT   = os.path.join(_REPO_ROOT, "data", "viz")

DATA_DIR = os.environ.get("APP_DATA_DIR")
if not DATA_DIR:
    DATA_DIR = _DEFAULT if os.path.isdir(_DEFAULT) else os.path.join("data", "viz")


def _p(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def _read_table(stem: str) -> pd.DataFrame:
    """Read a table by stem, preferring parquet, falling back to pickle.
    Real deployments use parquet (see requirements). The pickle fallback lets
    the app run in environments without a parquet engine."""
    pq = _p(f"{stem}.parquet")
    pk = _p(f"{stem}.pkl")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    return pd.read_pickle(pk)


def _table_exists(stem: str) -> bool:
    return os.path.exists(_p(f"{stem}.parquet")) or os.path.exists(_p(f"{stem}.pkl"))


# ---------------------------------------------------------------------------
# Display tables
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_article_table() -> pd.DataFrame:
    return _read_table("article_table")


@st.cache_data(show_spinner=False)
def load_cluster_meta() -> pd.DataFrame:
    return _read_table("cluster_meta")


@st.cache_data(show_spinner=False)
def load_cluster_publisher() -> pd.DataFrame:
    return _read_table("cluster_publisher")


@st.cache_data(show_spinner=False)
def load_cluster_week() -> pd.DataFrame:
    return _read_table("cluster_week")


# ---------------------------------------------------------------------------
# Search artifacts
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_entity_vocab() -> pd.DataFrame:
    return _read_table("entity_vocab")


@st.cache_data(show_spinner=False)
def load_entity_embeddings() -> np.ndarray:
    # Stored int8-quantized (unit vectors x127) to stay under GitHub's size
    # limit; dequantize and renormalize back to unit vectors for search.
    # Older float32 files are returned as-is for backward compatibility.
    raw = np.load(_p("entity_embeddings.npy"))
    if raw.dtype == np.int8:
        emb = raw.astype(np.float32) / 127.0
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        np.divide(emb, norms, out=emb, where=norms > 0)
        return emb
    return raw


@st.cache_resource(show_spinner="Loading search model…")
def load_search_model():
    """The embedding model for query-time semantic search.
    Loaded once and cached across reruns. Must match the model that built the
    entity embeddings — otherwise query and entity vectors live in different
    spaces and search results are meaningless.

    To avoid mismatches, we prefer the model name recorded by precompute in
    search_meta.json. An explicit SEARCH_MODEL env var overrides it (and is then
    checked against the recorded model, failing loud on conflict)."""
    import json
    from sentence_transformers import SentenceTransformer

    recorded = None
    meta_path = _p("search_meta.json")
    if os.path.exists(meta_path):
        recorded = json.load(open(meta_path)).get("model")

    env_model = os.environ.get("SEARCH_MODEL")
    # Priority: recorded model (what the vectors were actually built with),
    # then env override, then a sensible default matching the search pipeline.
    model_name = recorded or env_model or "intfloat/multilingual-e5-small"

    if env_model and recorded and env_model != recorded:
        raise RuntimeError(
            f"Model mismatch: entity embeddings were built with '{recorded}' "
            f"but SEARCH_MODEL is set to '{env_model}'. Unset SEARCH_MODEL to use "
            f"'{recorded}', or re-run 07_precompute.ipynb with "
            f"SEARCH_MODEL='{env_model}'."
        )
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def load_map_xy() -> pd.DataFrame:
    return _read_table("map_xy")


@st.cache_data(show_spinner=False)
def load_map_centroids() -> pd.DataFrame:
    return _read_table("map_centroids")


def map_available() -> bool:
    return _table_exists("map_xy") and _table_exists("map_centroids")


@st.cache_data(show_spinner=False)
def load_article_embeddings():
    """(emb, ids) e5-small per-article narrative embeddings for thematic article
    scoping. Returns None if not present (thematic card-scoping then disabled)."""
    ep = _p("narrative_emb_small.npy")
    ip = _p("narrative_emb_ids.npy")
    if not (os.path.exists(ep) and os.path.exists(ip)):
        return None
    emb = np.load(ep)
    ids = np.load(ip, allow_pickle=True).astype(str)
    return emb, ids


@st.cache_data(show_spinner=False)
def load_entity_article_bridge():
    """(label, id) rows: which article contains which entity label as a slot
    filler. Enables entity search to scope to specific articles. None if absent."""
    if not _table_exists("entity_article_bridge"):
        return None
    return _read_table("entity_article_bridge")


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------
LEVELS = ["macro", "meso", "micro"]

# Metric display names and their kind (sequential vs diverging colour scale)
METRICS = {
    "sentiment":  {"label": "Sentiment",      "kind": "diverging"},
    "anger":      {"label": "Anger",          "kind": "sequential"},
    "fear":       {"label": "Fear",           "kind": "sequential"},
    "joy":        {"label": "Joy",            "kind": "sequential"},
    "sadness":    {"label": "Sadness",        "kind": "sequential"},
}

FREQ_METRICS = {
    "n_articles":       "Absolute count",
    "rel_by_publisher": "Share of publisher",
    "rel_by_narrative": "Share of narrative",
}


def cluster_meta_for(meta: pd.DataFrame, level: str) -> pd.DataFrame:
    return meta[meta["level"] == level].copy()


def get_cluster_row(meta: pd.DataFrame, level: str, cluster: float) -> pd.Series | None:
    m = meta[(meta["level"] == level) & (meta["cluster"] == cluster)]
    return m.iloc[0] if len(m) else None


def data_available() -> bool:
    """True if the precompute artifacts exist on disk."""
    required = ["article_table", "cluster_meta", "cluster_publisher", "cluster_week"]
    return all(_table_exists(f) for f in required)


def search_available() -> bool:
    ok = _table_exists("entity_index") and _table_exists("entity_vocab") \
         and _table_exists("cluster_centroid_keys") \
         and os.path.exists(_p("entity_embeddings.npy")) \
         and os.path.exists(_p("cluster_centroids.npy"))
    return ok


@st.cache_data(show_spinner=False)
def load_narrative_extractions() -> pd.DataFrame:
    """Slim per-filler slot table used to aggregate slot-type groups at query time.

    Prefers the precomputed `slot_fillers` table (data/viz), which carries only the
    six card slots and the columns the card needs. Falls back to the full raw
    extractions CSV for older data dirs that predate the slim table.
    Returns an empty DataFrame if neither is present."""
    if _table_exists("slot_fillers"):
        return _read_table("slot_fillers")
    path = os.path.join(_REPO_ROOT, "data", "narrative_extractions_normalized.csv")
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_entity_ftype_map() -> dict:
    """Map from canonical entity label to its most common filler_type
    (actor | abstract_force | ideology). Returns {} if data not available.

    Prefers the precomputed `entity_ftype` table; falls back to deriving the map
    from the extractions table at runtime."""
    if _table_exists("entity_ftype"):
        t = _read_table("entity_ftype")
        if not t.empty and {"label", "filler_type"} <= set(t.columns):
            return dict(zip(t["label"], t["filler_type"]))
    ext = load_narrative_extractions()
    if ext.empty or "filler_type" not in ext.columns or "label" not in ext.columns:
        return {}
    return (
        ext.dropna(subset=["label", "filler_type"])
        .groupby("label")["filler_type"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )