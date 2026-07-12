"""Compare — side-by-side narrative cards with saved filters."""
from __future__ import annotations
import pandas as pd
import streamlit as st

import data as D
from components import card
from components import tray as T


# ---------------------------------------------------------------------------
def _open_in_inspect(item: dict):
    """Pre-populate saved filters and navigate to Inspect view."""
    from components.tray import _navigate_to_inspect
    _navigate_to_inspect(item)


# ---------------------------------------------------------------------------
def render():
    cs = st.session_state.get("comparison_set", [])

    st.title("Compare narratives")

    if not cs:
        st.info(
            "No narratives selected yet. "
            "Go to **Explore** to open narrative cards and add them to your "
            "comparison set using the **＋ Add to comparison** button."
        )
        return

    meta = D.load_cluster_meta()
    art  = D.load_article_table()
    ext  = D.load_narrative_extractions()

    st.caption(f"{len(cs)} selected \u00b7 up to {T.MAX_COMPARE}")
    st.divider()

    _narrative_comparison(cs, meta, art, ext)



# ---------------------------------------------------------------------------
# Mode A: Narrative comparison
# ---------------------------------------------------------------------------
def _narrative_comparison(cs, meta, art, ext):
    cw = D.load_cluster_week()
    cp = D.load_cluster_publisher()

    cols = st.columns(len(cs))
    for col, item in zip(cols, cs):
        with col:
            _narrative_column(item, meta, art, cw, cp, ext)


# ---------------------------------------------------------------------------
def _narrative_column(item, meta, art, cw_full, cp_full, ext):
    level = item.get("level", "search")
    k     = T._key(item)
    saved_pub  = item.get("pub")
    saved_week = item.get("week")

    # Header: inspect + remove
    _hc = st.columns([3, 1])
    if _hc[0].button("↗ Inspect", key=f"cmp_ins_{k}", width='stretch'):
        _open_in_inspect(item)
    if _hc[1].button("✕", key=f"cmp_rm_{k}", help="Remove"):
        T.toggle(item)
        st.rerun()

    if item.get("type") == "cluster":
        _cluster_card(item, meta, art, cw_full, cp_full, ext, k)
    elif item.get("type") == "search":
        _search_card(item, art, ext, saved_pub, k)


# ---------------------------------------------------------------------------
def _cluster_card(item, meta, art, cw_full, cp_full, ext, ks):
    level        = item["level"]
    cluster      = item["cluster"]
    saved_pub    = item.get("pub")
    saved_week   = item.get("week")
    row = D.get_cluster_row(meta, level, cluster)
    if row is None:
        st.error("Cluster not found.")
        return

    lm         = meta[meta["level"] == level]
    corpus_avg = {
        col: float(lm[col].mean())
        for col in ("sentiment", "anger", "fear", "joy", "sadness")
        if col in lm.columns
    }

    card.render_card(
        meta_row=row, article_table=art,
        cluster_week=cw_full, cluster_publisher=cp_full,
        level=level, corpus_avg=corpus_avg,
        ext=ext,
        key_suffix=f"_cmp_{ks}",
        initial_pub=saved_pub,
        initial_week=saved_week,
        filters_expanded=False,
    )


# ---------------------------------------------------------------------------
def _search_card(item, art, ext, selected_pub, ks):
    import search_core as SC
    import scope as SCP_
    from views.search_view import _pseudo_card

    query     = item.get("query", "")
    method    = item.get("method", "thematic")
    threshold = item.get("threshold", 0.85)

    if method != "corpus" and not D.search_available():
        st.warning("Search artifacts not found.")
        return

    model = D.load_search_model() if method != "corpus" else None

    if method == "corpus":
        matched_art = art.copy()
    elif method == "entity_labels":
        _labels = [l for l in query.split("|") if l]
        _ent_df = pd.DataFrame({"label": _labels})
        import scope as SCP_
        ids = SCP_.entity_article_ids(_ent_df)
        matched_art = art[art["id"].astype(str).isin(ids)] if ids else pd.DataFrame()
    elif method == "entity":
        entity_emb   = D.load_entity_embeddings()
        entity_vocab = D.load_entity_vocab()
        bridge       = D.load_entity_article_bridge()
        if bridge is None:
            st.warning("Entity\u2013article bridge not found.")
            return
        qv          = SC.embed_query_entity(query, model)
        matched_ent = SC.semantic_entity_search(
            qv, entity_emb, entity_vocab,
            threshold=threshold, max_entities=len(entity_vocab),
        )
        scope_ids   = SCP_.entity_article_ids(matched_ent)
        matched_art = art[art["id"].isin(scope_ids)] if scope_ids else pd.DataFrame()
    else:
        qv        = SC.embed_query_thematic(query, model)
        scope_ids = SCP_.thematic_article_ids(qv, threshold=threshold)
        matched_art = art[art["id"].isin(scope_ids)] if scope_ids else pd.DataFrame()

    if matched_art.empty:
        st.caption("No articles matched this query.")
        return

    _title = item.get("title") or query
    _pseudo_card(_title, matched_art, ext, art,
                 _ks=f"_cmp_{ks}",
                 initial_pub=item.get("pub"),
                 initial_week=item.get("week"),
                 filters_expanded=False,
                 level=item.get("level", "search"))
