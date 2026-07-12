"""Inspect — full narrative card for a selected cluster or search result.

Navigated to programmatically from the Explore page when the user opens a
cluster card or a search result card.  Never visited directly by the user.
"""
from __future__ import annotations
import pandas as pd
import streamlit as st

import data as D
import search_core as SC
import scope as SCP
from components import card
from components import tray as T


# ---------------------------------------------------------------------------
def render():
    if not D.data_available():
        st.warning("Data not found. Run the precompute pipeline first.")
        return

    meta = D.load_cluster_meta()
    art  = D.load_article_table()

    info = st.session_state.get("explore_card") or {}
    if not info:
        st.info("No narrative selected. Go to **Explore** and click a cluster or run a search.")
        if st.button("← Back to Explore"):
            st.switch_page(st.session_state["_pg_explore"])
        return

    level = info.get("level", "")

    # ── Header: back · level badge · comparison toggle ───────────────────
    hc = st.columns([1, 8])
    _origin = st.session_state.get("_inspect_origin", "explore")
    _back_label = "\u2190 Back to Compare" if _origin == "compare" else "\u2190 Back"
    if hc[0].button(_back_label, key="inspect_back"):
        st.session_state["overview_selected"] = None
        st.session_state["explore_card"] = None
        st.session_state.pop("_inspect_origin", None)
        _dest = (
            st.session_state.get("_pg_compare")
            if _origin == "compare"
            else st.session_state.get("_pg_explore")
        )
        if _dest:
            st.switch_page(_dest)
    with hc[1]:
        # Build a comparison item reflecting the current live filter state.
        # Session state is already updated for all widgets before this script runs.
        _cur_info = dict(info)
        if info.get("type") == "cluster":
            _lv_h = info.get("level", "")
            _cl_h = info.get("cluster")
            if _lv_h and _cl_h is not None:
                _ks_h = f"_{_lv_h}_{int(_cl_h)}"
                _pub_raw_h = st.session_state.get(f"pub_sel{_ks_h}")
                _cur_pub_h = None if (not _pub_raw_h or _pub_raw_h == "All") else _pub_raw_h
                _tl_h = f"card_timeline{_ks_h}_{_cur_pub_h or 'all'}"
                _wk_h = None
                try:
                    _wsh = st.session_state.get(_tl_h)
                    if _wsh and _wsh.selection and _wsh.selection.points:
                        _wk_h = _wsh.selection.points[0].get("x")
                except Exception:
                    pass
                _cur_info["pub"]  = _cur_pub_h
                _cur_info["week"] = _wk_h
        elif info.get("type") == "search":
            # _pseudo_card uses bare key suffix "" in inspect view
            _pub_raw_h = st.session_state.get("pseudo_pub_sel")
            _cur_pub_h = None if (not _pub_raw_h or _pub_raw_h == "All") else _pub_raw_h
            _tl_h = f"pseudo_card_timeline_{_cur_pub_h or 'all'}"
            _wk_h = None
            try:
                _wsh = st.session_state.get(_tl_h)
                if _wsh and _wsh.selection and _wsh.selection.points:
                    _wk_h = _wsh.selection.points[0].get("x")
            except Exception:
                pass
            _cur_info["pub"]  = _cur_pub_h
            _cur_info["week"] = _wk_h

        already      = T.in_set(_cur_info)
        already_base = T.in_set_base(_cur_info)  # same narrative, different filters
        cs_full  = len(st.session_state.get("comparison_set", [])) >= T.MAX_COMPARE
        if already:
            if st.button("✓ In comparison", key="inspect_cmp_toggle", type="secondary"):
                T.toggle(_cur_info)
                st.rerun()
        elif already_base:
            # Same narrative already saved — offer update OR separate new slot
            if st.button("↻ Update filters", key="inspect_cmp_update", type="primary",
                         help="Replace the saved filters for this narrative",
                         width='stretch'):
                T.update_filters(_cur_info)
                st.rerun()
            if not cs_full:
                if st.button("＋ Add new slot", key="inspect_cmp_add", type="secondary",
                             help="Keep existing slot and add this as a second one",
                             width='stretch'):
                    T.toggle(_cur_info)
                    st.rerun()
        elif cs_full:
            st.caption(f"Max {T.MAX_COMPARE} reached")
        else:
            if st.button("＋ Add to comparison", key="inspect_cmp_toggle", type="primary"):
                T.toggle(_cur_info)
                st.rerun()

    st.divider()

    card_type = info.get("type")
    if card_type == "cluster":
        _show_cluster_card(meta, art, info)
    elif card_type == "search":
        _show_search_card(art, info)
    else:
        st.info("No narrative selected.")


# ---------------------------------------------------------------------------
def _show_cluster_card(meta, art, info):
    level   = info["level"]
    cluster = info["cluster"]
    row = D.get_cluster_row(meta, level, cluster)
    if row is None:
        st.error("Cluster not found.")
        return
    corpus_avg  = _level_corpus_avg(meta, level)
    ext         = D.load_narrative_extractions()
    article_ids = art.loc[art[f"{level}_cluster"] == cluster, "id"].tolist()
    card.render_card(
        meta_row=row, article_table=art,
        cluster_week=D.load_cluster_week(),
        cluster_publisher=D.load_cluster_publisher(),
        level=level, corpus_avg=corpus_avg,
        ext=ext, article_ids=article_ids,
    )


def _show_search_card(art, info):
    query     = info.get("query", "")
    method    = info.get("method", "thematic")
    threshold = info.get("threshold", 0.85)
    if method != "corpus" and not D.search_available():
        st.warning("Search artifacts not found. Run `07_precompute.ipynb` first.")
        return
    model = D.load_search_model() if method != "corpus" else None
    if method == "corpus":
        matched_art = art.copy()
    elif method == "entity_labels":
        # Direct lookup: query stores pipe-separated canonical labels
        _labels = [l for l in query.split("|") if l]
        _ent_df = pd.DataFrame({"label": _labels})
        ids = SCP.entity_article_ids(_ent_df)
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
        ids = SCP.entity_article_ids(matched_ent)
        matched_art = art[art["id"].isin(ids)] if ids else pd.DataFrame()
    else:
        qv  = SC.embed_query_thematic(query, model)
        ids = SCP.thematic_article_ids(qv, threshold=threshold)
        matched_art = art[art["id"].isin(ids)] if ids else pd.DataFrame()
    if matched_art.empty:
        st.caption("No articles matched. The threshold may have changed since the preview was opened.")
        return
    ext = D.load_narrative_extractions()
    from views.search_view import _pseudo_card
    _title = info.get("title") or query
    _pseudo_card(_title, matched_art, ext, art, level=info.get("level", "search"))


def _level_corpus_avg(meta, level: str) -> dict:
    lm = meta[meta["level"] == level]
    return {
        col: float(lm[col].mean())
        for col in ("sentiment", "complexity", "anger", "fear", "joy", "sadness")
        if col in lm.columns
    }
