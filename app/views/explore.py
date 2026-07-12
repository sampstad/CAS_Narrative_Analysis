"""Explore — landing page for narrative analysis.

One scrolling page with two sections:
  search  — three panels (narrative/concept, entity, full corpus) that each
            produce a preview card from a free-form query
  map     — the 2D cluster map at three resolutions (macro / meso / micro)
            plus an "all articles" reference cloud; a text box highlights
            matching clusters

Selecting a cluster dot shows an inline preview; opening a preview or search
result navigates to the Inspect page (st.switch_page) for the full card.
"""
from __future__ import annotations
import pandas as pd
import streamlit as st

import data as D
import search_core as SC
import scope as SCP
import theme
from components import card
from components import tray as T


# ---------------------------------------------------------------------------
def render():
    if not D.data_available():
        st.warning("Data not found. Run the precompute pipeline first.")
        return

    meta = D.load_cluster_meta()
    art  = D.load_article_table()

    _browse_view(meta, art)


# ---------------------------------------------------------------------------
# Cluster preview (inline, shown when a map dot is selected)
# ---------------------------------------------------------------------------
def _cluster_preview(meta, art, sel):
    import json
    from components.slotgrammar import _compute_slot_type_groups
    from components.card import level_badge_html

    level, cluster = sel
    row = D.get_cluster_row(meta, level, cluster)
    if row is None:
        return

    article_ids = art.loc[art[f"{level}_cluster"] == cluster, "id"]
    n     = len(article_ids)
    total = max(len(art), 1)

    st.markdown(
        f"{level_badge_html(level)} &nbsp; **{row['title']}**",
        unsafe_allow_html=True,
    )

    from components.card import _prof_headline_html
    st.markdown(_prof_headline_html(f"{n:,}", f"{n / total:.1%}"),
                unsafe_allow_html=True)

    ext     = D.load_narrative_extractions()
    stg_raw = _compute_slot_type_groups(ext, article_ids)
    stg     = json.loads(stg_raw) if stg_raw and stg_raw != "{}" else {}
    if stg:
        _SLOT_STYLES = {
            "helper":   ("#4a9a4a", "rgba(74,154,74,0.06)",   "1px"),
            "subject":  ("#5a72c0", "rgba(90,114,192,0.30)",  "2px"),
            "opponent": ("#c04040", "rgba(192,64,64,0.06)",   "1px"),
            "sender":   ("#c88a3a", "rgba(200,138,58,0.06)",  "1px"),
            "object":   ("#3a9a9a", "rgba(58,154,154,0.18)",  "2px"),
            "receiver": ("#8a5ac8", "rgba(138,90,200,0.06)",  "1px"),
        }
        from components.card import _fmt_type, _CATEGORY_EMOJI, _NON_SPECIFIC_TYPES
        cells = []
        for slot, (color, bg, bw) in _SLOT_STYLES.items():
            groups = stg.get(slot, [])
            slot_labels = []
            for g in groups[:5]:
                stype = g.get("specific_type") or ""
                if not stype or stype in _NON_SPECIFIC_TYPES:
                    continue
                emoji = _CATEGORY_EMOJI.get(g.get("category") or g.get("filler_type", ""), "")
                lbl   = _fmt_type(stype) or stype.replace("_", " ")
                slot_labels.append(f"{emoji}\u00a0{lbl}" if emoji else lbl)
                if len(slot_labels) >= 3:
                    break
            if slot_labels:
                items_html = "".join(
                    f"<div style='margin-bottom:2px;overflow:hidden;text-overflow:ellipsis;"
                    f"white-space:nowrap;max-width:100%;font-size:0.80rem;'>{sl}</div>"
                    for sl in slot_labels
                )
            else:
                items_html = "<div style='opacity:0.35;font-style:italic;'>—</div>"
            name_weight = "800" if slot in ("subject", "object") else "700"
            name_size   = "0.73rem" if slot in ("subject", "object") else "0.68rem"
            text_color  = "rgba(10,14,40,0.88)" if slot in ("subject", "object") else "inherit"
            cells.append(
                f"<div style='border:{bw} solid {color};border-radius:6px;"
                f"padding:6px 9px;background:{bg};min-width:0;overflow:hidden;'>"
                f"<div style='font-size:{name_size};text-transform:uppercase;"
                f"letter-spacing:0.06em;font-weight:{name_weight};color:{color};"
                f"margin-bottom:4px;'>{slot}</div>"
                f"<div style='font-size:0.81rem;line-height:1.45;color:{text_color};'>{items_html}</div>"
                f"</div>"
            )
        st.markdown(
            "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
            "gap:7px;margin:10px 0 6px 0;overflow:hidden;'>"
            + "".join(cells) + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    from components import tray as T
    _cmp_item = {
        "type": "cluster", "level": level, "cluster": cluster,
        "title": str(row["title"]), "pub": None, "week": None,
    }
    _already  = T.in_set(_cmp_item)
    _cs_full  = len(st.session_state.get("comparison_set", [])) >= T.MAX_COMPARE
    _fc = st.columns(3)
    with _fc[0]:
        if _already:
            if st.button("\u2713 In comparison", key="cluster_preview_cmp",
                         type="secondary", width='stretch'):
                T.toggle(_cmp_item); st.rerun()
        elif _cs_full:
            st.caption(f"Max {T.MAX_COMPARE} reached")
        else:
            if st.button("\uff0b Compare", key="cluster_preview_cmp",
                         type="secondary", width='stretch'):
                T.toggle(_cmp_item); st.rerun()
    with _fc[1]:
        if st.button("Open full card \u2192", type="primary",
                     key="cluster_preview_open",
                     width='stretch'):
            st.session_state["explore_card"] = {
                "type":    "cluster",
                "level":   level,
                "cluster": cluster,
                "title":   str(row["title"]),
            }
            st.session_state["_inspect_origin"] = "explore"
            st.switch_page(st.session_state["_pg_inspect"])
    with _fc[2]:
        if st.button("\u2715 Close", key="cluster_preview_close_btn",
                     width='stretch'):
            st.session_state["overview_selected"] = None
            st.rerun()


# ---------------------------------------------------------------------------
# Browse mode
# ---------------------------------------------------------------------------
def _browse_view(meta, art):

    # ── Search ────────────────────────────────────────────────────────────
    st.markdown("### \U0001f50d Search specific narratives or topics")
    st.markdown(
        "Generate a narrative card from a free-form query, outside the cluster structure. "
        "Three modes are available: search by **narrative proposition** "
        "(a story structure you want to find), by **topic / entity** "
        "(a concept that must appear in the narrative roles), or browse "
        "the **full corpus** — all articles, no filter. "
        "All three produce a card you can add to your comparison set."
    )
    _search_section()

    st.divider()

    # ── Map ───────────────────────────────────────────────────────────────
    st.markdown("### 🗺️ Browse automatically generated clusters")
    st.markdown(
        "The maps below visualise **narrative clusters** — groups of articles that share a common "
        "story structure, not just a topic. Each cluster was derived automatically from the corpus "
        "by embedding every article's extracted narrative roles and then grouping articles whose "
        "storylines converge. The clusters are therefore **data-driven propositions**: they do not "
        "reflect a researcher's prior categories but the inherent narrative logic of the corpus itself.\n\n"
        "The embedding space has three levels of resolution. At the **auto-cluster / macro** level you see the "
        "broadest thematic poles of the corpus. **Meso** reveals distinct story angles "
        "within each macro theme, and **micro** shows the finest framings. "
        "**Click any dot** to open that cluster's full narrative card."
    )

    # ── Cluster text search ───────────────────────────────────────────────
    _cluster_q = st.text_input(
        "Search clusters",
        placeholder="e.g. SNB, climate, migration\u2026",
        key="map_cluster_search",
        help="Searches cluster titles, descriptions, and slot entities across all levels. "
             "Matching clusters are highlighted on the maps.",
    )
    _q = _cluster_q.strip().lower()
    if _q:
        _search_cols = [c for c in ("title", "description", "top_entities") if c in meta.columns]
        _mask = meta[_search_cols[0]].fillna("").str.lower().str.contains(_q, na=False)
        for _col in _search_cols[1:]:
            _mask |= meta[_col].fillna("").str.lower().str.contains(_q, na=False)
        _hits = meta[_mask]
        _mm: dict = {lv: set(_hits.loc[_hits["level"] == lv, "cluster"]) for lv in ("macro", "meso", "micro")}
        st.session_state["_map_matched"] = _mm
        _total_m = sum(len(v) for v in _mm.values())
        if _total_m:
            st.caption(
                f"\u2605 {_total_m} cluster{'s' if _total_m != 1 else ''} matched "
                f"\u2014 macro: {len(_mm['macro'])}, meso: {len(_mm['meso'])}, micro: {len(_mm['micro'])}"
            )
        else:
            st.caption("No clusters matched.")
    else:
        st.session_state.pop("_map_matched", None)
    if D.map_available():
        _map_section(meta, art)
    else:
        st.warning(
            "2D map not found. Run the `cell-2d-map` step in "
            "`07_precompute.ipynb` to generate `map_xy` and `map_centroids`."
        )

    # ── Cluster preview ───────────────────────────────────────────────────
    sel = st.session_state.get("overview_selected")
    if sel:
        st.divider()
        st.markdown("### Selected cluster")
        _cluster_preview(meta, art, sel)


def _map_section(meta, art):
    from views.overview import (
        _macro_colour_map, _cloud_map, _one_map, _colour_legend,
        _selection_highlights, _compose,
    )
    sel       = st.session_state.get("overview_selected")
    highlight = _selection_highlights(meta, sel)
    matched   = st.session_state.get("_map_matched")
    active    = _compose(highlight, matched)
    cmap      = _macro_colour_map(meta)

    xy    = D.load_map_xy()
    xpad  = (xy["x_2d"].max() - xy["x_2d"].min()) * 0.03
    ypad  = (xy["y_2d"].max() - xy["y_2d"].min()) * 0.03
    xrange = [xy["x_2d"].min() - xpad, xy["x_2d"].max() + xpad]
    yrange = [xy["y_2d"].min() - ypad, xy["y_2d"].max() + ypad]

    cloud_hl = None
    if sel:
        _lv, _cl = sel
        cloud_hl = set(art[art[f"{_lv}_cluster"] == _cl]["id"].astype(str))
    elif matched:
        # Highlight articles belonging to any matched cluster
        _search_ids: set = set()
        for _lv, _cls in matched.items():
            for _cl in _cls:
                _col = f"{_lv}_cluster"
                if _col in art.columns:
                    _search_ids.update(art[art[_col] == _cl]["id"].astype(str))
        if _search_ids:
            cloud_hl = _search_ids

    all_cents = D.load_map_centroids()
    size_ref  = (float(all_cents["n_articles"].min()),
                 float(all_cents["n_articles"].max()))

    def _count(lvl):
        return len(meta[meta.level == lvl])

    def _title(lvl, label):
        total = _count(lvl)
        mm = st.session_state.get("_map_matched")
        if mm and mm.get(lvl) is not None:
            n_hl = len(mm[lvl])
            return f"**{label}** ({n_hl} of {total} highlighted) \u00b7 click a dot"
        return f"**{label}** ({total}) \u00b7 click a dot"

    total_art = art["id"].nunique()
    r1 = st.columns(2)
    with r1[0]:
        if cloud_hl:
            art_cap = f"({len(cloud_hl):,} of {total_art:,} highlighted)"
        else:
            art_cap = f"({total_art:,} articles)"
        st.markdown(f"**All articles** {art_cap}")
        _cloud_map(xy, cmap, cloud_hl, xrange, yrange)
        st.caption("Reference \u2014 coloured by macro theme, not interactive.")
    with r1[1]:
        st.markdown(_title("macro", "Macro themes"))
        _one_map(meta, xy, cmap, "macro", active.get("macro"), size_ref, xrange, yrange)
    r2 = st.columns(2)
    with r2[0]:
        st.markdown(_title("meso", "Meso topics"))
        _one_map(meta, xy, cmap, "meso", active.get("meso"), size_ref, xrange, yrange)
    with r2[1]:
        st.markdown(_title("micro", "Micro framings"))
        _one_map(meta, xy, cmap, "micro", active.get("micro"), size_ref, xrange, yrange)
    _colour_legend(cmap, meta)


def _search_section():
    """Three-column search: narrative proposition, entity/topic, full corpus."""
    c_nar, c_top, c_corp = st.columns(3)
    with c_nar:
        _narrative_panel()
    with c_top:
        _entity_panel()
    with c_corp:
        _full_corpus_panel()


# ---------------------------------------------------------------------------
# Narrative search
# ---------------------------------------------------------------------------
def _narrative_panel():
    st.markdown("#### \U0001f50d Narrative / concept search")
    st.markdown(
        "Enter any query \u2014 a single concept (*immigration*, *SNB*, *Klimawandel*) "
        "or a full narrative proposition (*FDP is drifting right under voter pressure*). "
        "The search finds articles whose extracted story structure is semantically "
        "similar to your query. The more proposition-like the query, the more "
        "precise the match."
    )
    query = st.text_input(
        "narrative_q", placeholder="e.g. Russia is threatening European democracy…",
        label_visibility="collapsed", key="explore_nar_query",
    )
    if query.strip():
        _narrative_result(query.strip())


def _narrative_result(query: str):
    if not D.search_available():
        st.warning("Search artifacts not found. Run `07_precompute.ipynb` first.")
        return
    # Only re-embed when the query actually changes
    if st.session_state.get("_nar_last_q") != query:
        with st.spinner("Searching…"):
            art   = D.load_article_table()
            model = D.load_search_model()
            qv    = SC.embed_query_thematic(query, model)
            pack  = D.load_article_embeddings()
        if pack is None:
            st.caption("Article-level matching unavailable (`narrative_emb_small.npy` not found).")
            return
        emb, emb_ids = pack
        sims = emb @ qv
        st.session_state["_nar_sims_cache"]    = sims
        st.session_state["_nar_emb_ids_cache"] = emb_ids
        st.session_state.pop("explore_nar_sim", None)
        st.session_state["_nar_last_q"] = query
    else:
        art     = D.load_article_table()
        sims    = st.session_state["_nar_sims_cache"]
        emb_ids = st.session_state["_nar_emb_ids_cache"]
    import numpy as _np
    _t_vals  = _np.round(_np.arange(0.70, 0.955, 0.01), 2)
    _counts  = [(sims >= t).sum() for t in _t_vals]
    _auto_t, _auto_cands = _auto_threshold(_counts, _t_vals)

    # Use auto-threshold by default; expose slider in expander for advanced use
    threshold = st.session_state.get("explore_nar_sim", _auto_t)
    with st.expander("\u2699\ufe0f Adjust similarity threshold", expanded=False):
        chart_slot = st.empty()
        threshold  = st.slider(
            "Similarity threshold", 0.70, 0.95, _auto_t, 0.01,
            key="explore_nar_sim",
            help="Higher = only very similar stories; lower = broader net",
        )
        with chart_slot:
            _narrative_threshold_curve(sims, threshold, _auto_t, _auto_cands, key="nar_thr_curve")
        st.caption(
            "Green lines mark thresholds where the next lower step would double results; "
            "\u2605 = suggested default."
        )

    ids = set(emb_ids[sims >= threshold].astype(str))
    if ids:
        matched = art[art["id"].isin(ids)]
        _fallback_note = None
    else:
        # No results at auto-threshold — fall back to top 30 most similar
        top_idx = _np.argsort(-sims)[:30]
        ids = set(emb_ids[top_idx].astype(str))
        matched = art[art["id"].isin(ids)]
        _fallback_note = "No natural threshold found \u2014 showing the 30 most similar articles."

    if _fallback_note:
        st.caption(_fallback_note)

    ext = D.load_narrative_extractions()
    _search_preview(query, matched, ext, art, level="narrative", ks="nar", threshold=threshold)


# ---------------------------------------------------------------------------
# Topic / entity search
# ---------------------------------------------------------------------------
def _entity_panel():
    st.markdown("#### \U0001f3f7\ufe0f Entity search")
    st.markdown(
        "Deselecting slots or types filters both the label list and the article counts. "
        "Bars show how many matching labels fall in each category. "
        "See **Glossary** (sidebar) for slot and entity type definitions."
    )
    st.markdown(
        "<style>.ent-panel label p{font-size:0.78rem !important;}</style>",
        unsafe_allow_html=True,
    )

    # Pre-load vocab + ftype map (cached) so we can compute distributions
    _vocab     = D.load_entity_vocab()
    _ftype_map = D.load_entity_ftype_map()

    def _mini_bar(count: int, max_count: int, color: str = "#5a72c0") -> str:
        pct = (count / max_count * 100) if max_count else 0
        return (
            f"<div style='background:rgba(128,128,128,0.12);border-radius:2px;"
            f"height:5px;overflow:hidden;margin:2px 0 1px 0;'>"
            f"<div style='background:{color};height:100%;width:{pct:.0f}%;"
            f"border-radius:2px;opacity:0.70;'></div></div>"
            f"<span style='font-size:0.62rem;opacity:0.50;'>{count:,}</span>"
        )

    _ALL_SLOTS  = ["subject", "object", "helper", "opponent", "sender", "receiver"]
    _SLOT_COLOR = theme.SLOT_COLOR
    _FT_COLOR   = theme.FT_COLOR
    _FT_LABELS  = theme.FT_LABEL

    # ── Search inputs ──────────────────────────────────────────────────────
    with st.container(border=True):
        _ent_q = st.text_input(
            "entity_q", placeholder="e.g. SNB, Thomas Jordan, Asylbewerber\u2026",
            label_visibility="collapsed", key="explore_ent_query",
        )
        _q_lower = _ent_q.strip().lower()

        # Distribution counts from unfiltered text matches (for bar display)
        if _q_lower:
            _dist_hits = _vocab[_vocab["label"].str.lower().str.contains(
                _q_lower, regex=False, na=False)]
            _slot_dist = {
                s: int(_dist_hits["slots"].fillna("").str.contains(s).sum())
                for s in _ALL_SLOTS
            } if "slots" in _dist_hits.columns else {s: 0 for s in _ALL_SLOTS}
            _max_slot = max(_slot_dist.values()) or 1
            _ft_dist = {
                ft: sum(1 for l in _dist_hits["label"] if _ftype_map.get(l) == ft)
                for ft in _FT_LABELS
            }
            _max_ft = max(_ft_dist.values()) or 1
        else:
            _slot_dist = {s: 0 for s in _ALL_SLOTS}
            _ft_dist   = {ft: 0 for ft in _FT_LABELS}
            _max_slot  = 1
            _max_ft    = 1

        st.markdown(
            "<span style='font-size:0.72rem;color:rgba(120,120,120,0.80);'>Slots</span>",
            unsafe_allow_html=True,
        )
        _r1 = st.columns(3)
        _r2 = st.columns(3)
        _slot_vals = {}
        for _i, _s in enumerate(["subject", "object", "helper"]):
            with _r1[_i]:
                _slot_vals[_s] = st.checkbox(_s.capitalize(), value=True,
                                              key=f"explore_ent_slot_{_s}")
                st.markdown(_mini_bar(_slot_dist[_s], _max_slot, _SLOT_COLOR[_s]),
                            unsafe_allow_html=True)
        for _i, _s in enumerate(["opponent", "sender", "receiver"]):
            with _r2[_i]:
                _slot_vals[_s] = st.checkbox(_s.capitalize(), value=True,
                                              key=f"explore_ent_slot_{_s}")
                st.markdown(_mini_bar(_slot_dist[_s], _max_slot, _SLOT_COLOR[_s]),
                            unsafe_allow_html=True)

        st.markdown(
            "<span style='font-size:0.72rem;color:rgba(120,120,120,0.80);'>Filler type</span>",
            unsafe_allow_html=True,
        )
        _ft_cols = st.columns(3)
        _ft_vals = {}
        for _i, (ft, lbl) in enumerate(_FT_LABELS.items()):
            with _ft_cols[_i]:
                _ft_vals[ft] = st.checkbox(lbl, value=True, key=f"explore_ent_ft_{ft}")
                st.markdown(_mini_bar(_ft_dist[ft], _max_ft, _FT_COLOR[ft]),
                            unsafe_allow_html=True)
    _slot_filter_list = [s for s, v in _slot_vals.items() if v]
    _slot_filter      = _slot_filter_list if len(_slot_filter_list) < 6 else None
    _ft_filter_list   = [ft for ft, v in _ft_vals.items() if v]
    _ft_filter        = _ft_filter_list if len(_ft_filter_list) < 3 else None

    _q = _ent_q.strip()
    if not _q or not _slot_filter_list or not _ft_filter_list:
        if not _slot_filter_list:
            st.caption("Select at least one slot to search.")
        elif not _ft_filter_list:
            st.caption("Select at least one filler type to search.")
        return

    _hits = _vocab[_vocab["label"].str.lower().str.contains(_q.lower(), regex=False, na=False)]

    # Filter by filler type
    if _ft_filter:
        _hits = _hits[_hits["label"].map(
            lambda l: _ftype_map.get(l, "actor") in _ft_filter
        )]

    # Filter by selected slots
    if _slot_filter and "slots" in _hits.columns:
        def _slot_match(s):
            return bool(set(str(s).split(",")) & set(_slot_filter)) if pd.notna(s) else False
        _hits = _hits[_hits["slots"].apply(_slot_match)]

    _total_matches = len(_hits)
    _hits = _hits.sort_values("total_articles", ascending=False).head(20)

    if _hits.empty:
        st.caption("No entity labels matched.")
        return

    # Compute article counts filtered by selected slots
    _bridge = D.load_entity_article_bridge()
    if _bridge is not None and "slot" in _bridge.columns and not _hits.empty:
        _match_labels = set(_hits["label"])
        _bsub = _bridge[_bridge["label"].isin(_match_labels)]
        if _slot_filter:
            _bsub = _bsub[_bsub["slot"].isin(_slot_filter)]
        _slot_counts = _bsub.groupby("label")["id"].nunique()
        _hits = _hits.copy()
        _hits["display_n"] = _hits["label"].map(_slot_counts).fillna(0).astype(int)
    else:
        _hits = _hits.copy()
        _hits["display_n"] = _hits["total_articles"].astype(int)

    _hits = _hits[_hits["display_n"] > 0].sort_values("display_n", ascending=False)
    if _hits.empty:
        st.caption("No entity labels matched for the selected slots.")
        return

    # ── Results ────────────────────────────────────────────────────────────
    with st.container(border=True):
        _n_shown  = len(_hits)
        _cap_note = (
            f"showing {_n_shown} of {_total_matches} matches"
            if _total_matches > _n_shown else
            f"{_n_shown} match{'es' if _n_shown != 1 else ''}"
        )
        st.markdown(
            f"<span style='font-size:0.72rem;color:rgba(120,120,120,0.80);'>{_cap_note}</span>",
            unsafe_allow_html=True,
        )
        _n_hits = _n_shown

        # Select-all / clear-all buttons
        _btn_c = st.columns(2)
        if _btn_c[0].button("Select all", key="explore_ent_sel_all"):
            for _j in range(_n_hits):
                st.session_state[f"explore_ent_lbl_{_j}"] = True
            st.rerun()
        if _btn_c[1].button("Clear all", key="explore_ent_clr_all"):
            for _j in range(_n_hits):
                st.session_state[f"explore_ent_lbl_{_j}"] = False
            st.rerun()
        _selected = []
        for _i, (_, _row) in enumerate(_hits.iterrows()):
            _lbl = _row["label"]
            _n   = int(_row["display_n"])
            if st.checkbox(f"{_lbl}  \u00b7  {_n:,}", value=(_i < 3),
                           key=f"explore_ent_lbl_{_i}"):
                _selected.append(_lbl)

        if _selected:
            _entity_result(_selected, _slot_filter, search_query=_q)


def _entity_result(selected_labels: list, slot_filter: list | None = None,
                   search_query: str = ""):
    art     = D.load_article_table()
    _ent_df = pd.DataFrame({"label": selected_labels})
    ids     = SCP.entity_article_ids(_ent_df, slots=slot_filter)
    if not ids:
        st.caption("No articles found for the selected entities.")
        if slot_filter:
            st.caption(f"Try removing the slot restriction (currently: {', '.join(slot_filter)}).")
        return
    matched = art[art["id"].astype(str).isin(ids)]
    ext     = D.load_narrative_extractions()
    _base   = search_query or selected_labels[0]
    if slot_filter:
        _slot_str = ", ".join(s.capitalize() for s in slot_filter)
        _display  = f"{_base} [{_slot_str}]"
    else:
        _display  = _base
    _search_preview(_display, matched, ext, art, level="topic", ks="ent",
                    threshold=None, query_override="|".join(selected_labels),
                    method_override="entity_labels")


# ---------------------------------------------------------------------------
# Threshold sensitivity curves
# ---------------------------------------------------------------------------
def _auto_threshold(counts: list, t_vals: "list",
                    min_count: int = 100, max_count: int = 5000) -> "tuple[float, list]":
    """Find every threshold where stepping to the next lower value at least doubles results.
    Returns (selected_threshold, all_candidate_thresholds).
    Candidates are sorted high-to-low.  The selected threshold is the middle
    candidate (rounded toward the higher value when the count is even).
    Returns (lowest_with_results, []) when no doubling thresholds are found.
    """
    candidates = []
    for i in range(len(t_vals) - 1, 0, -1):
        c_here  = int(counts[i])
        c_lower = int(counts[i - 1])
        if min_count <= c_here <= max_count and c_lower >= 2 * c_here:
            candidates.append(float(t_vals[i]))
    if candidates:
        selected = candidates[(len(candidates) - 1) // 2]
        return selected, candidates
    # Fallback: lowest threshold that returns at least one result
    for i in range(len(t_vals) - 1, -1, -1):
        if int(counts[i]) > 0:
            return float(t_vals[i]), []
    return 0.85, []


def _narrative_threshold_curve(sims: "np.ndarray", current_threshold: float,
                                auto_threshold: float = None,
                                candidates: list = None, key: str = None):
    import numpy as np
    import plotly.graph_objects as go

    t_vals = np.round(np.arange(0.70, 0.955, 0.01), 2)
    counts = [(sims >= t).sum() for t in t_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_vals, y=counts, mode="lines", fill="tozeroy",
        line=dict(width=2, color="#8a5ac8"),
        hovertemplate="threshold %{x:.2f} → %{y:,} articles<extra></extra>",
    ))
    for ct in (candidates or []):
        is_sel = (auto_threshold is not None and abs(ct - auto_threshold) < 0.005)
        if is_sel and abs(ct - current_threshold) > 0.005:
            fig.add_vline(
                x=ct, line_dash="dot",
                line=dict(color="#2d9f2d", width=2.0),
                annotation_text=f"★ {ct:.2f}",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="#2d9f2d",
            )
        elif not is_sel:
            fig.add_vline(x=ct, line_dash="dot",
                          line=dict(color="#2d9f2d", width=1.0))
    fig.add_vline(
        x=current_threshold, line_dash="dash",
        line=dict(color="#8a5ac8", width=1.5),
        annotation_text=f"{current_threshold:.2f}",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_xanchor="left",
    )
    fig.update_layout(
        height=150, margin=dict(l=0, r=0, t=18, b=0),
        xaxis=dict(title="Similarity threshold", range=[0.70, 0.95]),
        yaxis=dict(title="Articles"),
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch',
                    config={"displayModeBar": False}, key=key)


# ---------------------------------------------------------------------------
# Shared inline preview
# ---------------------------------------------------------------------------
def _search_preview(
    query: str, matched_art: pd.DataFrame, ext: pd.DataFrame,
    art: pd.DataFrame, level: str, ks: str, threshold: float = 0.85,
    query_override: str = None, method_override: str = None,
):
    """Lightweight card preview — stats + top slot entities + Open button."""
    import json
    from components.card import level_badge_html
    from components.slotgrammar import _compute_slot_type_groups

    n     = len(matched_art)
    total = max(len(art), 1)

    st.markdown(level_badge_html(level), unsafe_allow_html=True)

    if n == 0:
        st.caption("No articles matched. Try lowering the threshold.")
        return

    pc = st.columns(2)
    pc[0].metric("Matched articles", f"{n:,}")
    pc[1].metric("Corpus share", f"{n / total:.1%}")

    # Top entities per slot — mini slot cards matching card.py color scheme
    if not matched_art.empty:
        stg_raw = _compute_slot_type_groups(ext, matched_art["id"])
        stg = json.loads(stg_raw) if stg_raw and stg_raw != "{}" else {}
        _SLOT_STYLES = {
            "helper":   ("#4a9a4a", "rgba(74,154,74,0.06)",   "1px"),
            "subject":  ("#5a72c0", "rgba(90,114,192,0.30)",  "2px"),
            "opponent": ("#c04040", "rgba(192,64,64,0.06)",   "1px"),
            "sender":   ("#c88a3a", "rgba(200,138,58,0.06)",  "1px"),
            "object":   ("#3a9a9a", "rgba(58,154,154,0.18)",  "2px"),
            "receiver": ("#8a5ac8", "rgba(138,90,200,0.06)",  "1px"),
        }
        cells = []
        for slot, (color, bg, bw) in _SLOT_STYLES.items():
            groups = stg.get(slot, [])
            stypes = [
                card._fmt_type(g.get("specific_type"))
                for g in groups
                if g.get("specific_type") and g["specific_type"] not in card._NON_SPECIFIC_TYPES
            ][:4]
            if stypes:
                items_html = "".join(
                    "<div style='margin-bottom:1px;overflow:hidden;text-overflow:ellipsis;"
                    "white-space:nowrap;max-width:100%;'>" + (_s[:40] + "\u2026" if len(_s) > 40 else _s) + "</div>"
                    for _s in stypes
                )
            else:
                items_html = "<div style='opacity:0.35;font-style:italic;'>—</div>"
            name_weight = "800" if slot in ("subject", "object") else "700"
            name_size   = "0.73rem" if slot in ("subject", "object") else "0.68rem"
            text_color = "rgba(10,14,40,0.88)" if slot in ("subject", "object") else "inherit"
            cells.append(
                f"<div style='border:{bw} solid {color};border-radius:6px;"
                f"padding:6px 9px;background:{bg};min-width:0;overflow:hidden;'>"
                f"<div style='font-size:{name_size};text-transform:uppercase;"
                f"letter-spacing:0.06em;font-weight:{name_weight};color:{color};"
                f"margin-bottom:4px;'>{slot}</div>"
                f"<div style='font-size:0.81rem;line-height:1.45;color:{text_color};'>{items_html}</div>"
                f"</div>"
            )
        grid = (
            "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
            "gap:7px;margin:10px 0 6px 0;overflow:hidden;'>"
            + "".join(cells)
            + "</div>"
        )
        st.markdown(grid, unsafe_allow_html=True)

    st.markdown("")
    from components import tray as _T
    _scmp = {
        "type": "search", "level": level, "title": query,
        "query": query_override if query_override is not None else query,
        "method": method_override if method_override is not None else (
            "entity" if level == "topic" else ("corpus" if level == "corpus" else "thematic")
        ),
        "threshold": threshold, "pub": None, "week": None,
    }
    _salready = _T.in_set(_scmp)
    _sfc = st.columns(3)
    with _sfc[0]:
        if _salready:
            if st.button("✓ In comparison", key=f"explore_{ks}_cmp",
                         type="secondary", width='stretch'):
                _T.toggle(_scmp); st.rerun()
        elif len(st.session_state.get("comparison_set", [])) < _T.MAX_COMPARE:
            if st.button("＋ Compare", key=f"explore_{ks}_cmp",
                         type="secondary", width='stretch'):
                _T.toggle(_scmp); st.rerun()
        else:
            st.caption(f"Max {_T.MAX_COMPARE} reached")
    with _sfc[1]:
        if st.button("Open full card →", type="primary", key=f"explore_{ks}_open",
                     width='stretch'):
            st.session_state["explore_card"] = {
                "type":      "search",
                "level":     level,
                "title":     query,
                "query":     query_override if query_override is not None else query,
                "method":    method_override if method_override is not None else (
                    "entity" if level == "topic" else ("corpus" if level == "corpus" else "thematic")
                ),
                "threshold": threshold,
            }
            st.session_state["_inspect_origin"] = "explore"
            st.switch_page(st.session_state["_pg_inspect"])
    with _sfc[2]:
        if st.button("\u2715 Close", key=f"explore_{ks}_close",
                     width='stretch'):
            _close_keys = {"nar": "explore_nar_query", "top": "explore_top_query",
                           "ent": "explore_ent_query"}
            _qk = _close_keys.get(ks)
            if _qk:
                st.session_state.pop(_qk, None)
            if ks == "ent":
                st.session_state.pop("explore_ent_sel", None)
            elif ks not in ("nar", "top", "ent"):
                st.session_state.pop("explore_corpus_loaded", None)
            st.rerun()


# ---------------------------------------------------------------------------
# Full corpus panel
# ---------------------------------------------------------------------------
def _full_corpus_panel():
    st.markdown("#### \U0001f310 Full corpus")
    st.markdown(
        "No query needed \u2014 explore the **entire dataset**. "
        "Useful for comparing how publishers cover the corpus as a whole, "
        "independent of any specific narrative cluster."
    )
    if st.button(
        "Load full corpus \u203a",
        key="explore_corpus_btn",
        help="Load all articles into a narrative card you can filter by publisher and week",
    ):
        st.session_state["explore_corpus_loaded"] = True

    if not st.session_state.get("explore_corpus_loaded"):
        return

    art = D.load_article_table()
    ext = D.load_narrative_extractions()
    _search_preview("Full corpus", art, ext, art, level="corpus", ks="corpus", threshold=None)




