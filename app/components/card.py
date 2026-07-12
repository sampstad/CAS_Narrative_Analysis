"""
Narrative card component (hybrid: native Streamlit + one styled-HTML block).

The slot-grammar section is rendered as styled HTML because the Greimas grammar
benefits from a designed layout. Everything interactive (timeline, publisher
charts, click-throughs) uses native Streamlit / Plotly.

"""
from __future__ import annotations
import pandas as pd
import streamlit as st

import theme

from components.htmlutil import _esc
from components.slotgrammar import (
    parse_top_entities,
    _CORE_SLOTS,
    _AUX_SLOTS,
    _ROW1_SLOTS,
    _ROW2_SLOTS,
    _SLOT_DISPLAY,
    _NON_SPECIFIC_TYPES,
    _SLOT_CSS,
    _slot_html_block,
    render_slot_grammar,
    parse_slot_type_groups,
    SPECIFIC_TYPE_LABELS,
    _fmt_type,
    _FTYPE_COLOR,
    _FTYPE_COLOR_DEFAULT,
    _CATEGORY_EMOJI,
    _type_group_html,
    _slot_type_card_html,
    render_slot_grammar_compact,
    render_slot_grammar_typed,
    _SGI_SLOT_COLOR,
    _slot_groups_from_ext,
    _compute_slot_type_groups,
)
from components.metrics import (
    _METRIC_CSS,
    _pct_dev,
    _metric_bar_html,
    _METRIC_COLORS,
    _bar_for_metric,
    _prof_row_html,
    _prof_headline_html,
    _DEV_METRIC_COLS,
    _DEV_BAR_CSS,
    _DEV_BAR_METRIC_COLS,
    _DEV_BAR_RGB,
    _DEV_BAR_FULL,
    _dev_bars_html,
)
from components.cardviz import (
    _FLOW_PALETTE,
    _FLOW_BOX,
    _FLOW_SLOT_LINE,
    _FLOW_SLOT_FILL,
    _FLOW_CONNECTIONS,
    _FLOW_EDGE_PAD,
    _flow_rgba,
    _flow_banded_segs,
    _flow_midpt,
    _flow_half,
    _flow_ctrl,
    _flow_draw_ribbon,
    _build_actantial_fig,
    render_timeline,
    _render_publisher_timeline,
)


_LEVEL_COLORS = theme.LEVEL_COLORS


_LEVEL_BG = theme.LEVEL_BG


_LEVEL_DISPLAY = theme.LEVEL_DISPLAY


def level_badge_html(level: str) -> str:
    color   = _LEVEL_COLORS.get(level, "#888888")
    bg      = _LEVEL_BG.get(level, "rgba(128,128,128,0.10)")
    display = _LEVEL_DISPLAY.get(level, level)
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'border:1px solid {color};background:{bg};color:{color};'
        f'font-size:0.72rem;font-weight:700;letter-spacing:0.04em;'
        f'vertical-align:middle;">{display}</span>'
    )


def centroid_top_n(
    candidate_ids,
    emb: "np.ndarray",
    emb_ids: "np.ndarray",
    n: int = 5,
) -> list:
    """Return up to n article IDs from candidate_ids nearest to their centroid.

    Uses the preloaded narrative_emb_small embeddings (emb / emb_ids).
    Falls back to the original order when embeddings are unavailable.
    """
    import numpy as np
    cand = [str(i) for i in candidate_ids]
    id_to_row = {str(eid): idx for idx, eid in enumerate(emb_ids)}
    rows = [id_to_row[c] for c in cand if c in id_to_row]
    if not rows:
        return cand[:n]
    sub = emb[rows]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sub_n = sub / norms
    centroid = sub_n.mean(axis=0)
    cnorm = np.linalg.norm(centroid)
    if cnorm == 0:
        return cand[:n]
    centroid /= cnorm
    sims = sub_n @ centroid
    order = np.argsort(-sims)[:n]
    cand_indexed = [c for c in cand if c in id_to_row]
    return [cand_indexed[i] for i in order]


def _art_snippet(row, max_chars: int = 240) -> str:
    """Extract a brief text preview from an article row (up to ~3 sentences)."""
    for col in ("snippet", "content", "subhead", "body", "text", "lead"):
        try:
            val = row[col]
        except (KeyError, AttributeError):
            continue
        if val and isinstance(val, str):
            s = val.strip()
            if not s:
                continue
            if len(s) <= max_chars:
                return s
            cut = s[:max_chars]
            for p in (". ", "! ", "? "):
                last = cut.rfind(p)
                if last > max_chars // 2:
                    return cut[:last + 1] + " …"
            last_space = cut.rfind(" ")
            return (cut[:last_space] if last_space > 0 else cut) + " …"
    return ""


def render_card(
    meta_row: pd.Series,
    article_table: pd.DataFrame,
    cluster_week: pd.DataFrame,
    cluster_publisher: pd.DataFrame,
    level: str,
    relative: bool = True,
    show_examples: bool = True,
    scoped_note: str = None,
    corpus_avg: dict | None = None,
    ext: pd.DataFrame | None = None,
    article_ids: "list | None" = None,
    key_suffix: str = "",
    initial_pub: "str | None" = None,
    initial_week: "str | None" = None,
    filters_expanded: bool = True,
):
    """Redesigned narrative card, top to bottom:
      1. title + scoped note
      2. description
      3. key metrics (articles, share) + sentiment/emotion deviation bars
      4. story roles (Greimas actantial slots)
      5. timeline | publisher mix (two columns)
      6. representative articles
    """
    cluster = meta_row["cluster"]
    _ks = f"_{level}_{int(cluster)}{key_suffix}"

    # 1. Title + badge + description — fixed-height block in comparison, native widgets in inspect
    if not filters_expanded:
        _desc_raw = str(meta_row.get("description", "") or "").strip()
        _desc_html = (
            f"<div style='font-size:0.85rem;line-height:1.45;margin-top:6px;'>{_esc(_desc_raw)}</div>"
            if _desc_raw else
            "<div style='font-size:0.78rem;opacity:0.42;font-style:italic;margin-top:6px;'>"
            "No description available \u2014 not an automatically generated cluster.</div>"
        )
        _note_html = (
            f"<div style='font-size:0.72rem;opacity:0.55;margin-top:2px;'>\u29d7 {_esc(scoped_note)}</div>"
            if scoped_note else ""
        )
        if initial_pub:
            _bp_html = (
                f"<span style='background:rgba(90,114,192,0.15);border-radius:3px;"
                f"padding:1px 7px;font-size:0.68rem;font-weight:600;'>{_esc(initial_pub)}</span>"
            )
        else:
            _bp_html = (
                "<span style='background:rgba(128,128,128,0.10);border-radius:3px;"
                "padding:1px 7px;font-size:0.68rem;font-weight:600;opacity:0.60;'>All publishers</span>"
            )
        if initial_week:
            try:
                _wt = pd.Timestamp(initial_week)
                _wlbl = f"w/{_wt.strftime('%d %b %y')}"
            except Exception:
                _wlbl = str(initial_week)[:10]
            _bw_html = (
                f"<span style='background:rgba(74,154,74,0.15);border-radius:3px;"
                f"padding:1px 7px;font-size:0.68rem;font-weight:600;'>{_wlbl}</span>"
            )
        else:
            _bw_html = (
                "<span style='background:rgba(128,128,128,0.10);border-radius:3px;"
                "padding:1px 7px;font-size:0.68rem;font-weight:600;opacity:0.60;'>All weeks</span>"
            )
        st.markdown(
            f"<div style='height:210px;overflow:hidden;padding-bottom:4px;'>"
            f"<p style='font-size:1.15rem;font-weight:700;margin:0 0 3px 0;"
            f"line-height:1.25;'>{_esc(str(meta_row['title']))}</p>"
            f"{level_badge_html(level)}"
            f"{_note_html}"
            f"<div style='margin:4px 0 0 0;'>{_bp_html}&nbsp;{_bw_html}</div>"
            f"{_desc_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.subheader(str(meta_row["title"]))
        st.markdown(level_badge_html(level), unsafe_allow_html=True)
        if scoped_note:
            st.caption(f"\u29d7 {scoped_note}")
        if isinstance(meta_row["description"], str) and meta_row["description"]:
            st.markdown(str(meta_row["description"]))
        else:
            st.caption("No description available \u2014 not an automatically generated cluster.")

    # ── Publisher filter (applies to whole card) ────────────────────────────
    cp_cluster = cluster_publisher[
        (cluster_publisher["level"] == level) &
        (cluster_publisher["cluster"] == cluster)
    ].sort_values("n_articles", ascending=False).reset_index(drop=True)

    if not filters_expanded:
        # Comparison view: read-only — use the saved publisher filter
        sel_pub = initial_pub
    elif not cp_cluster.empty:
        st.markdown("**Publisher (click to filter)**")
        _pub_key = f"pub_sel{_ks}"
        sel_pub_raw = st.radio(
            "Publisher",
            options=["All"] + cp_cluster["publisher"].tolist(),
            key=_pub_key,
            horizontal=True,
            label_visibility="collapsed",
        )
        sel_pub = None if sel_pub_raw == "All" else sel_pub_raw
    else:
        sel_pub = None

    # ── Base article IDs for this cluster ─────────────────────────────────
    if article_ids is not None:
        _base_ids = article_ids
    else:
        _clcol = f"{level}_cluster"
        _base_ids = (
            article_table[article_table[_clcol] == cluster]["id"].tolist()
            if _clcol in article_table.columns else []
        )

    # ── Scoped article IDs ────────────────────────────────────────────────
    if sel_pub:
        _pub_id_set = set(article_table.loc[article_table["publisher"] == sel_pub, "id"].astype(str))
        scoped_ids  = [i for i in _base_ids if str(i) in _pub_id_set]
    else:
        scoped_ids = article_ids  # preserve None for downstream slot logic

    # ── Timeline + week filter ───────────────────────────────────────
    if not filters_expanded:
        # Comparison view: read-only — use the saved week filter
        _clicked_week = initial_week
        _tl_key = f"card_timeline{_ks}_{sel_pub or 'all'}"  # for metric key consistency
        # week filter is already shown as a badge above; no chart in comparison
    else:
        _tl_key = f"card_timeline{_ks}_{sel_pub or 'all'}"
        _clicked_week = None
        try:
            _ws = st.session_state.get(_tl_key)
            if _ws and _ws.selection and _ws.selection.points:
                _clicked_week = _ws.selection.points[0].get("x")
        except Exception:
            pass
        st.markdown("**Time Series (click bar to filter)**")
        if sel_pub and scoped_ids is not None:
            scoped_str = set(str(i) for i in scoped_ids)
            art_sub  = article_table[article_table["id"].astype(str).isin(scoped_str)]
            pub_all  = article_table[article_table["publisher"] == sel_pub]
            _render_publisher_timeline(art_sub, cluster_week, level, cluster,
                                       key=_tl_key, pub_all=pub_all)
        else:
            render_timeline(cluster_week, level, cluster, relative, key=_tl_key)
        if _clicked_week:
            st.caption("↕ Click the selected bar again to deselect")

    st.divider()

    # ── 3. Stats (1/5) + Articles (4/5) ──────────────────────────────────────
    _n_narr       = len(_base_ids) if _base_ids else int(meta_row.get("n_articles") or 0)
    _total_corpus = max(len(article_table), 1)
    _share_narr   = meta_row.get("corpus_share")

    # ── Metric deviation table ────────────────────────────────────────────
    _METRIC_KEYS = ["sentiment", "joy", "anger", "fear", "sadness"]
    _corpus_means = {k: corpus_avg.get(k) for k in _METRIC_KEYS} if corpus_avg else {}

    # Row 1: narrative overall vs corpus
    _narr_means = {}
    for _k in _METRIC_KEYS:
        _r = meta_row.get(_k)
        _narr_means[_k] = float(_r) if _r is not None and pd.notna(_r) else None
    _dev_rows = [("Narrative", _narr_means, _corpus_means)]

    # Row 2: publisher-filtered vs that publisher's overall mean
    if sel_pub:
        _pub_all_dt = article_table[article_table["publisher"] == sel_pub]
        _pub_means_d = {
            k: (float(_pub_all_dt[k].mean())
                if k in _pub_all_dt.columns and not _pub_all_dt.empty else None)
            for k in _METRIC_KEYS
        }
        if scoped_ids is not None:
            _scoped_s_dt = set(str(i) for i in scoped_ids)
            _sdf_dt = article_table[article_table["id"].astype(str).isin(_scoped_s_dt)]
            _smeans_d = {
                k: (float(_sdf_dt[k].mean())
                    if k in _sdf_dt.columns and not _sdf_dt.empty else None)
                for k in _METRIC_KEYS
            }
            _dev_rows.append((sel_pub[:22], _smeans_d, _pub_means_d))

    # Row 3: week-filtered vs appropriate baseline
    if _clicked_week:
        _wbase_ids = (scoped_ids if (sel_pub and scoped_ids) else _base_ids) or []
        _wbase_src = set(str(i) for i in _wbase_ids)
        _wref_dt   = article_table[article_table["id"].astype(str).isin(_wbase_src)]
        _wdf_dt    = (
            _wref_dt[_wref_dt["iso_week"] == _clicked_week]
            if "iso_week" in _wref_dt.columns else _wref_dt
        )
        _wmeans_d  = {
            k: (float(_wdf_dt[k].mean())
                if k in _wdf_dt.columns and not _wdf_dt.empty else None)
            for k in _METRIC_KEYS
        }
        _wbase_d   = _pub_means_d if (sel_pub and scoped_ids is not None) else _corpus_means
        _dev_rows.append(("Week " + str(_clicked_week)[:7], _wmeans_d, _wbase_d))

    # Headline (articles + corpus share)
    if _clicked_week:
        if sel_pub and scoped_ids:
            _hl_nw_ids = set(str(i) for i in scoped_ids)
            _hl_week_denom = article_table[
                (article_table["publisher"] == sel_pub) &
                (article_table["iso_week"] == _clicked_week)
            ].shape[0]
            _hl_share_ref = len(scoped_ids) / max(len(_pub_id_set), 1)
        else:
            _hl_nw_ids = set(str(i) for i in _base_ids) if _base_ids else set()
            _hl_week_denom = article_table[article_table["iso_week"] == _clicked_week].shape[0]
            _hl_share_ref = float(_share_narr) if _share_narr is not None and pd.notna(_share_narr) else None
        _hl_ref_df  = article_table[article_table["id"].astype(str).isin(_hl_nw_ids)]
        _hl_week_df = (
            _hl_ref_df[_hl_ref_df["iso_week"] == _clicked_week]
            if "iso_week" in _hl_ref_df.columns else _hl_ref_df
        )
        _hl_nw = len(_hl_week_df)
        _hl_share = _hl_nw / max(_hl_week_denom, 1)
        _hl = _prof_headline_html(f"{_hl_nw:,}", f"{_hl_share:.1%}")
    elif sel_pub and scoped_ids is not None:
        _hl_pub_total = len(_pub_id_set)
        _hl_pub_share = len(scoped_ids) / max(_hl_pub_total, 1)
        _hl = _prof_headline_html(f"{len(scoped_ids):,}", f"{_hl_pub_share:.1%}")
    else:
        _hl_share_str = (
            f"{_share_narr:.1%}"
            if _share_narr is not None and pd.notna(_share_narr) else "\u2014"
        )
        _hl = _prof_headline_html(f"{_n_narr:,}" if _n_narr else "\u2014", _hl_share_str)


    # Build representative articles list
    _ex = pd.DataFrame()
    if show_examples:
        import data as _D
        _art_emb = _D.load_article_embeddings()  # (emb, ids) or None

        def _centroid_pool(pool: pd.DataFrame, n: int = 5) -> pd.DataFrame:
            if pool.empty:
                return pool
            if _art_emb is not None:
                emb, emb_ids = _art_emb
                top_ids = centroid_top_n(pool["id"].tolist(), emb, emb_ids, n)
                id_order = {rid: i for i, rid in enumerate(top_ids)}
                res = pool[pool["id"].astype(str).isin(top_ids)].copy()
                res["_rank"] = res["id"].astype(str).map(id_order)
                return res.sort_values("_rank").drop(columns="_rank").head(n)
            if "pubtime" in pool.columns:
                return pool.sort_values("pubtime", ascending=False).head(n)
            return pool.head(n)

        if _clicked_week:
            _src_ids = scoped_ids if (sel_pub and scoped_ids) else _base_ids
            _src_str = set(str(i) for i in _src_ids) if _src_ids else set()
            _pool = article_table[article_table["id"].astype(str).isin(_src_str)].copy()
            if "iso_week" in _pool.columns:
                _pool = _pool[_pool["iso_week"] == _clicked_week]
            _ex = _centroid_pool(_pool)
        elif sel_pub and scoped_ids:
            _scoped_str2 = set(str(i) for i in scoped_ids)
            _pool = article_table[article_table["id"].astype(str).isin(_scoped_str2)].copy()
            _ex = _centroid_pool(_pool)
        else:
            _rep_ids = str(meta_row.get("representative_ids", "")).split(",")
            _rep_ids = [r for r in _rep_ids if r]
            if _rep_ids:
                _ex = article_table[article_table["id"].isin(_rep_ids)].copy()
                _id_order = {rid: i for i, rid in enumerate(_rep_ids)}
                _ex["_rank"] = _ex["id"].map(_id_order)
                _ex = _ex.sort_values("_rank").head(5)

    _section_title = (
        "Week" if _clicked_week else
        sel_pub if sel_pub else
        "Narrative Profile"
    )
    _hl_html = (
        _METRIC_CSS
        + '<div class="prof-section">'
        + f'<div class="prof-title">{_section_title}</div>'
        + _hl
        + '</div>'
    )
    # Current context values (first row = current filter state)
    _cur_vals_rc  = _dev_rows[0][1]  # means for current filter state
    _cur_base_rc  = _dev_rows[0][2]  # corpus baseline
    # If publisher active, use publisher baseline (row 1 of dev_rows if present)
    if len(_dev_rows) > 1 and sel_pub:
        _cur_vals_rc = _dev_rows[1][1]
        _cur_base_rc = _dev_rows[1][2]
    if _clicked_week and len(_dev_rows) > 0:
        _cur_vals_rc = _dev_rows[-1][1]
        _cur_base_rc = _dev_rows[-1][2]
    _dev_bars_rc = _dev_bars_html(_cur_vals_rc, _cur_base_rc)
    _metrics_html = (
        _DEV_BAR_CSS + _METRIC_CSS
        + '<div class="prof-section">'
        + f'<div class="prof-title">{_section_title}</div>'
        + _hl
        + ('<hr class="prof-sep">' if _dev_bars_rc else "")
        + _dev_bars_rc
        + '</div>'
    )
    if not filters_expanded:
        st.markdown(_metrics_html, unsafe_allow_html=True)
    else:
        _stat_col, _art_col = st.columns([1, 4])
        with _stat_col:
            st.markdown(_metrics_html, unsafe_allow_html=True)

        with _art_col:
            if show_examples and not _ex.empty:
                _art_lbl = (
                    f"**Articles \u2014 week {_clicked_week}**" if _clicked_week
                    else "**Representative articles**"
                )
                st.markdown(_art_lbl)
                _head_col = "head" if "head" in _ex.columns else None
                for _i, (_, _a) in enumerate(_ex.iterrows()):
                    _atitle = _a[_head_col] if _head_col else _a["id"]
                    _snip = _art_snippet(_a)
                    _ahtml = (
                        f"<div style='margin-bottom:5px;line-height:1.3;'>"
                        f"<span style='font-size:0.70rem;font-weight:700;opacity:0.35;"
                        f"margin-right:4px;'>#{_i + 1}</span>"
                        f"<span style='font-size:0.80rem;font-weight:600;'>{_esc(str(_atitle))}</span>"
                    )
                    if _snip:
                        _ahtml += (
                            f"<div style='font-size:0.75rem;opacity:0.65;margin:1px 0;'>"
                            f"{_esc(_snip)}</div>"
                        )
                    _ahtml += (
                        f"<div style='font-size:0.68rem;opacity:0.45;'>"
                        f"{_esc(str(_a['publisher']))} \u00b7 "
                        f"{pd.to_datetime(_a['pubtime']).date()}</div>"
                        f"</div>"
                    )
                    st.markdown(_ahtml, unsafe_allow_html=True)

        st.divider()

    # 4. Story roles (publisher / week filtered)
    # For the default (unfiltered) card we reuse the precomputed slot_type_groups
    # from cluster_meta. The uncapped live aggregation is only needed as a live
    # reference when a publisher or week filter is active — or as a fallback when
    # the precomputed groups are missing.
    stg_all = str(meta_row.get("slot_type_groups", "") or "")
    _need_live = bool(
        ext is not None and _base_ids
        and (_clicked_week or sel_pub or not stg_all or stg_all in ("{}", ""))
    )
    if _need_live:
        stg_full = _compute_slot_type_groups(ext, _base_ids, max_groups=None)
    else:
        stg_full = None

    _slot_lbl = f"**Narrative Slots \u2014 week {_clicked_week}**" if _clicked_week else "**Narrative Slots**"
    st.markdown(_slot_lbl)

    if _clicked_week and ext is not None:
        _week_base = scoped_ids if (sel_pub and scoped_ids) else _base_ids
        _week_src_str = set(str(i) for i in _week_base) if _week_base else set()
        _wdf = article_table[article_table["id"].astype(str).isin(_week_src_str)]
        if "iso_week" in _wdf.columns:
            _wdf = _wdf[_wdf["iso_week"] == _clicked_week]
        _week_ids = _wdf["id"].tolist()
        if _week_ids:
            stg_week = _compute_slot_type_groups(ext, _week_ids)
            _ref_ids = scoped_ids if (sel_pub and scoped_ids) else _base_ids
            stg_ref_w = (
                _compute_slot_type_groups(ext, _ref_ids, max_groups=None)
                if _ref_ids else (stg_full or stg_all)
            )
            if stg_week and stg_week not in ("{}", ""):
                render_slot_grammar_typed(stg_week,
                    ref_json=stg_ref_w if stg_ref_w not in ("{}", "") else None,
                    compact=not filters_expanded)
            else:
                st.caption(f"No slot data for week {_clicked_week}.")
        else:
            st.caption(f"No articles found for week {_clicked_week}.")
    elif sel_pub and ext is not None and scoped_ids is not None:
        stg_pub = _compute_slot_type_groups(ext, scoped_ids)
        stg_ref = stg_full if stg_full and stg_full not in ("{}", "") else stg_all
        if stg_pub and stg_pub not in ("{}", ""):
            render_slot_grammar_typed(stg_pub,
                ref_json=stg_ref if stg_ref not in ("{}", "") else None,
                compact=not filters_expanded)
        else:
            st.caption(f"No slot data for {_esc(sel_pub)}.")
    else:
        stg_display = stg_full if stg_full and stg_full not in ("{}", "") else stg_all
        if stg_display and stg_display not in ("{}", ""):
            render_slot_grammar_typed(stg_display, compact=not filters_expanded)
        else:
            render_slot_grammar(meta_row.get("top_entities", ""))

    # ── Top entities in this narrative ────────────────────────────────────────
    import data as _D_card
    _bridge_n = _D_card.load_entity_article_bridge()
    _ids_for_ent = scoped_ids if (sel_pub and scoped_ids) else _base_ids
    if _bridge_n is not None and _ids_for_ent:
        _bid_set_n = set(str(i) for i in _ids_for_ent)
        _ent_sub_n = _bridge_n[_bridge_n["id"].astype(str).isin(_bid_set_n)]
        if not _ent_sub_n.empty:
            _ent_top_n = (_ent_sub_n.groupby("label")["id"].nunique()
                          .sort_values(ascending=False).head(12))
            _ent_slot_n = (
                _ent_sub_n.groupby("label")["slot"]
                .agg(lambda x: x.value_counts().index[0])
            )
            _badges = []
            for _lbl_n, _n_n in _ent_top_n.items():
                _sl_n = _ent_slot_n.get(_lbl_n, "subject")
                _sc_n = _SGI_SLOT_COLOR.get(_sl_n, "#888")
                _badges.append(
                    f"<span style='display:inline-block;padding:2px 7px;margin:2px 2px;"
                    f"border-radius:12px;border:1px solid {_sc_n};color:{_sc_n};"
                    f"font-size:0.70rem;'>{_lbl_n}</span>"
                )
            with st.expander(f"Top entities ({len(_ent_top_n)})",
                             expanded=not filters_expanded):
                st.markdown(" ".join(_badges), unsafe_allow_html=True)
