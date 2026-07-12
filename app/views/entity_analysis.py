"""Entity Analysis - explore and inspect entities in the narrative corpus.

Search resolves a query to one or more entity labels, then delegates the full
profile rendering to components.entity_card.render_entity_card (the same card
used by the entity comparison view), so the two entity screens stay in sync.
"""
from __future__ import annotations
import streamlit as st

import data as D
from components.entity_card import render_entity_card


def render():
    st.title("Explore & inspect entities")

    if not D.data_available():
        st.warning("Data not found. Run the precompute pipeline first.")
        return

    art       = D.load_article_table()
    bridge    = D.load_entity_article_bridge()
    vocab     = D.load_entity_vocab()
    ftype_map = D.load_entity_ftype_map()

    if bridge is None:
        st.warning("Entity\u2013article bridge not found. Re-run `07_precompute.ipynb`.")
        return

    # Search (pre-populated from tray navigation when present)
    _pre = st.session_state.pop("ea_inspect_label", "")
    _q = st.text_input(
        "entity_search",
        value=_pre,
        placeholder="e.g. SNB, Thomas Jordan, Klimawandel\u2026",
        label_visibility="collapsed",
        key="ea_query",
    )
    q = _q.strip()
    if not q:
        st.caption("Enter an entity name to explore its role in the narrative corpus.")
        return

    hits = vocab[vocab["label"].str.lower().str.contains(q.lower(), regex=False, na=False)]
    hits = hits.sort_values("total_articles", ascending=False)
    if hits.empty:
        st.caption("No entity labels matched.")
        return

    n_labels = len(hits)
    labels = list(hits["label"])
    st.caption(f"{n_labels:,} entity label{'s' if n_labels != 1 else ''} matched")

    _cols = st.columns([5, 1])
    with _cols[1]:
        if st.button("\u2715 Close", key="ea_close"):
            st.session_state.pop("ea_query", None)
            st.rerun()

    st.divider()

    render_entity_card(
        title=q,
        labels=labels,
        bridge=bridge,
        art=art,
        ftype_map=ftype_map,
        key_suffix="_ea",
    )
