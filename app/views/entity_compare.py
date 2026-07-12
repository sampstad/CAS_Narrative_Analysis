"""Entity Compare — side-by-side entity cards."""
from __future__ import annotations
import streamlit as st

import data as D


def render():
    st.title("Compare entities")

    bridge    = D.load_entity_article_bridge()
    art       = D.load_article_table()
    ftype_map = D.load_entity_ftype_map()

    if bridge is None:
        st.warning("Entity\u2013article bridge not found. Re-run `07_precompute.ipynb`.")
        return

    ecs = st.session_state.get("entity_compare_set", [])

    if not ecs:
        st.info(
            "No entities selected. Use **Explore entities** to add entities "
            "to your comparison set via the sidebar."
        )
        return

    n = len(ecs)
    st.caption(f"{n} selected \u00b7 up to 8")
    st.divider()

    from components.entity_card import render_entity_card
    vocab = D.load_entity_vocab()
    cols = st.columns(n)
    for col, query in zip(cols, ecs):
        with col:
            # Replicate same multi-label lookup as the inspect view
            hits = vocab[vocab["label"].str.lower().str.contains(
                query.lower(), regex=False, na=False)]
            matched_labels = hits["label"].tolist() if not hits.empty else [query]
            render_entity_card(
                title=query, labels=matched_labels,
                bridge=bridge, art=art, ftype_map=ftype_map,
                compact=False, key_suffix=f"_cmp_{abs(hash(query)) % 9999}",
            )
