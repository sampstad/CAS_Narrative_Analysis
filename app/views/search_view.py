"""Pseudo-narrative card for free-text search results.

`_pseudo_card` renders a card for an arbitrary set of matched articles (metrics,
slot grammar, timeline, sample articles), mirroring `render_card`. Imported by
the Inspect and Compare views; not a standalone page.
"""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as D
import search_core as SC
import scope as SCP
from components import card
from components.slotgrammar import _compute_slot_type_groups


# ---------------------------------------------------------------------------
def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pseudo_card(query: str, matched_art: pd.DataFrame, ext: pd.DataFrame,
                 art: pd.DataFrame, _ks: str = "",
                 initial_pub: "str | None" = None,
                 initial_week: "str | None" = None,
                 filters_expanded: bool = True,
                 level: str = "search"):
    """Pseudo-narrative card for matched articles.
    Mirrors render_card: title, metrics+publisher rows, publisher filter,
    slots (with delta badges when filtered), timeline, representative articles.
    """
    head_col = "head" if "head" in matched_art.columns else None
    n        = len(matched_art)
    total    = max(len(art), 1)

    emo_keys = ["anger", "fear", "joy", "sadness"]
    corpus_avg = {
        "sentiment":  float(art["sentiment"].mean()) if "sentiment"  in art.columns else None,
        "complexity": float(art["complexity"].mean()) if "complexity" in art.columns else None,
        **{k: float(art[k].mean()) for k in emo_keys if k in art.columns},
    }

    # ── 1. Title + level badge ─────────────────────────────────────────────
    from components.card import level_badge_html, _esc
    from components.card import (
        _pct_dev, _METRIC_CSS,
        _metric_bar_html, _bar_for_metric,
        _prof_row_html, _prof_headline_html,
    )

    if not filters_expanded:
        # Fixed-height header block for alignment across comparison columns
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
            f"line-height:1.25;'>{_esc(query)}</p>"
            f"{level_badge_html(level)}"
            f"<div style='font-size:0.72rem;opacity:0.55;margin-top:2px;'>"
            f"\u29d7 {n:,} matched articles</div>"
            f"<div style='margin:4px 0 0 0;'>{_bp_html}&nbsp;{_bw_html}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.subheader(query)
        st.markdown(level_badge_html(level), unsafe_allow_html=True)
        st.caption(f"\u29d7 {n:,} matched articles")

    # ── 3. Publisher filter + scoped metrics row ────────────────────────────
    if not filters_expanded:
        sel_pub = initial_pub
        _clicked_week = initial_week
        _tl_key = f"pseudo_card_timeline{_ks}_{sel_pub or 'all'}"
    else:
        # Inspect view: interactive publisher radio
        if "publisher" in matched_art.columns:
            _pubs = (
                matched_art.groupby("publisher")["id"].count()
                .sort_values(ascending=False).index.tolist()
            )
            st.markdown("**Publisher (click to filter)**")
            _pub_key = f"pseudo_pub_sel{_ks}"
            sel_pub_raw = st.radio(
                "Publisher",
                options=["All"] + _pubs,
                key=_pub_key,
                horizontal=True,
                label_visibility="collapsed",
            )
            sel_pub = None if sel_pub_raw == "All" else sel_pub_raw
        else:
            sel_pub = None
        _tl_key = f"pseudo_card_timeline{_ks}_{sel_pub or 'all'}"
        _clicked_week = None
        try:
            _ws = st.session_state.get(_tl_key)
            if _ws and _ws.selection and _ws.selection.points:
                _clicked_week = _ws.selection.points[0].get("x")
        except Exception:
            pass

    # Derive scoped articles
    if sel_pub:
        scoped_art = matched_art[matched_art["publisher"] == sel_pub].copy()
    else:
        scoped_art = matched_art

    # ── Time Series ─────────────────────────────────────────────────
    if not filters_expanded:
        pass  # week filter shown as badge above; no timeline chart in comparison
    else:
        st.markdown("**Time Series (click bar to filter)**")
        if "iso_week" in matched_art.columns and "iso_week" in art.columns:
            _corp_weekly = (
                art.groupby("iso_week")["id"].count().reset_index(name="n_total")
            )
            if sel_pub:
                _pub_corp_weekly = (
                    art[art["publisher"] == sel_pub]
                    .groupby("iso_week")["id"].count()
                    .reset_index(name="n_total")
                )
                weekly_pub = (
                    scoped_art.groupby("iso_week")["id"].nunique()
                    .reset_index(name="n").sort_values("iso_week")
                )
                weekly_pub = weekly_pub.merge(_pub_corp_weekly, on="iso_week", how="left")
                weekly_pub["share"] = weekly_pub["n"] / weekly_pub["n_total"].clip(lower=1)
                weekly_ref = (
                    matched_art.groupby("iso_week")["id"].nunique()
                    .reset_index(name="n").sort_values("iso_week")
                )
                weekly_ref = weekly_ref.merge(_corp_weekly, on="iso_week", how="left")
                weekly_ref["share"] = weekly_ref["n"] / weekly_ref["n_total"].clip(lower=1)
                if not weekly_pub.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=weekly_ref["iso_week"], y=weekly_ref["share"],
                        mode="lines", name="All publishers",
                        line=dict(width=1.5, color="rgba(160,160,160,0.45)", dash="dot"),
                    ))
                    fig.add_trace(go.Bar(
                        x=weekly_pub["iso_week"], y=weekly_pub["share"],
                        name=sel_pub,
                    ))
                    fig.update_layout(
                        height=200, margin=dict(l=0, r=0, t=6, b=0),
                        yaxis_title="Share of publisher corpus", xaxis_title=None,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    fig.update_yaxes(tickformat=".1%")
                    st.plotly_chart(fig, width='stretch',
                                    config={"displayModeBar": False},
                                    key=_tl_key, on_select="rerun",
                                    selection_mode="points")
            else:
                weekly = (
                    matched_art.groupby("iso_week")["id"].nunique()
                    .reset_index(name="n").sort_values("iso_week")
                )
                weekly = weekly.merge(_corp_weekly, on="iso_week", how="left")
                weekly["share"] = weekly["n"] / weekly["n_total"].clip(lower=1)
                if not weekly.empty:
                    fig = go.Figure(go.Bar(
                        x=weekly["iso_week"], y=weekly["share"],
                    ))
                    fig.update_layout(
                        height=180, margin=dict(l=0, r=0, t=6, b=0),
                        yaxis_title="Share of corpus", xaxis_title=None, showlegend=False,
                    )
                    fig.update_yaxes(tickformat=".1%")
                    st.plotly_chart(fig, width='stretch',
                                    config={"displayModeBar": False},
                                    key=_tl_key, on_select="rerun",
                                    selection_mode="points")
        if _clicked_week:
            st.caption("\u2195 Click the selected bar again to deselect")

    st.divider()

    _METRIC_KEYS_P = ["sentiment", "joy", "anger", "fear", "sadness"]

    # Deviation baseline: publisher's OVERALL corpus mean (cluster- and narrative-independent)
    if sel_pub and "publisher" in art.columns:
        _pub_all_p = art[art["publisher"] == sel_pub]
        _pub_mean_p = {
            k: (float(_pub_all_p[k].mean())
                if k in _pub_all_p.columns and not _pub_all_p.empty else None)
            for k in _METRIC_KEYS_P
        }
    else:
        _pub_mean_p = {k: corpus_avg.get(k) for k in _METRIC_KEYS_P} if corpus_avg else {}

    from components.card import _dev_bars_html, _DEV_BAR_CSS

    _share_denom   = (
        int((art["publisher"] == sel_pub).sum()) if sel_pub and "publisher" in art.columns
        else total
    )
    _scoped_share  = len(scoped_art) / max(_share_denom, 1)
    _overall_share = len(matched_art) / total if sel_pub else None

    # Build metric HTML items (vertical list for left column)
    if _clicked_week and "iso_week" in scoped_art.columns:
        _week_art = scoped_art[scoped_art["iso_week"] == _clicked_week]
        _nw = len(_week_art)
        # Denominator: ALL articles that week (publisher-scoped if active)
        if sel_pub and "publisher" in art.columns:
            _week_denom = int(((art["publisher"] == sel_pub) & (art["iso_week"] == _clicked_week)).sum())
        else:
            _week_denom = int((art["iso_week"] == _clicked_week).sum()) if "iso_week" in art.columns else total
        _corpus_share_week = _nw / max(_week_denom, 1)

        def _wmean(df, col):
            try:
                return float(df[col].mean()) if col in df.columns and not df.empty else None
            except Exception:
                return None

        _hl_p = _prof_headline_html(f"{_nw:,}", f"{_corpus_share_week:.1%}")
        _cur_vals_p = {k: _wmean(_week_art, k) for k in _METRIC_KEYS_P}
        _metric_rows_p = _dev_bars_html(_cur_vals_p, _pub_mean_p)
    else:
        def _smean_p(df, col):
            try:
                return float(df[col].mean()) if col in df.columns and not df.empty else None
            except Exception:
                return None
        _hl_p = _prof_headline_html(f"{len(scoped_art):,}", f"{_scoped_share:.1%}")
        _metric_rows_p = _dev_bars_html(
            {k: _smean_p(scoped_art, k) for k in _METRIC_KEYS_P}, _pub_mean_p)

    # Build articles list
    from components.card import _art_snippet, centroid_top_n
    _art_emb = D.load_article_embeddings()  # (emb, ids) or None

    def _centroid_articles(pool: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Return up to n rows from pool sorted by proximity to pool centroid."""
        if pool.empty:
            return pool
        if _art_emb is not None:
            emb, emb_ids = _art_emb
            top_ids = centroid_top_n(pool["id"].tolist(), emb, emb_ids, n)
            id_order = {rid: i for i, rid in enumerate(top_ids)}
            result = pool[pool["id"].astype(str).isin(top_ids)].copy()
            result["_rank"] = result["id"].astype(str).map(id_order)
            return result.sort_values("_rank").drop(columns="_rank").head(n)
        # fallback: most recent
        if "pubtime" in pool.columns:
            return pool.sort_values("pubtime", ascending=False).head(n)
        return pool.head(n)

    if _clicked_week and "iso_week" in scoped_art.columns:
        _src_art = scoped_art if sel_pub else matched_art
        _ex_art = _centroid_articles(_src_art[_src_art["iso_week"] == _clicked_week].copy())
    elif sel_pub:
        _ex_art = _centroid_articles(scoped_art.copy())
    else:
        _ex_art = _centroid_articles(matched_art)

    _ps_title = (
        f"Week {str(_clicked_week)[:10]}" if _clicked_week else
        sel_pub if sel_pub else
        "Corpus Profile"
    )
    _metrics_html = (
        _DEV_BAR_CSS + _METRIC_CSS
        + '<div class="prof-section">'
        + f'<div class="prof-title">{_ps_title}</div>'
        + _hl_p
        + ('<hr class="prof-sep">' if _metric_rows_p else '')
        + _metric_rows_p
        + '</div>'
    )
    if not filters_expanded:
        st.markdown(_metrics_html, unsafe_allow_html=True)
    else:
        _stat_col, _art_col = st.columns([1, 4])
        with _stat_col:
            st.markdown(_metrics_html, unsafe_allow_html=True)

        with _art_col:
            if not _ex_art.empty:
                _art_lbl = (
                    f"**Articles \u2014 week {_clicked_week}**" if _clicked_week
                    else "**Representative articles**"
                )
                st.markdown(_art_lbl)
                for _i, (_, _a) in enumerate(_ex_art.iterrows()):
                    _atitle = _a[head_col] if head_col else str(_a.get("id", ""))
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

        if filters_expanded:
            st.divider()

    # ── 4. Narrative slots (publisher / week filtered) ─────────────────────
    stg_full = _compute_slot_type_groups(ext, matched_art["id"], max_groups=None)
    _slot_lbl = f"**Narrative Slots \u2014 week {_clicked_week}**" if _clicked_week else "**Narrative Slots**"
    st.markdown(_slot_lbl)

    if _clicked_week:
        _week_src = scoped_art if sel_pub else matched_art
        _wdf = (
            _week_src[_week_src["iso_week"] == _clicked_week]
            if "iso_week" in _week_src.columns else pd.DataFrame()
        )
        if not _wdf.empty:
            stg_week = _compute_slot_type_groups(ext, _wdf["id"])
            _ref_w = stg_full if stg_full and stg_full != "{}" else None
            if stg_week and stg_week != "{}":
                card.render_slot_grammar_typed(stg_week, ref_json=_ref_w, compact=not filters_expanded)
            else:
                st.caption(f"No slot data for week {_clicked_week}.")
        else:
            st.caption(f"No articles found for week {_clicked_week}.")
    elif sel_pub:
        stg_pub = _compute_slot_type_groups(ext, scoped_art["id"])
        if stg_pub and stg_pub != "{}":
            card.render_slot_grammar_typed(
                stg_pub,
                ref_json=stg_full if stg_full and stg_full != "{}" else None,
                compact=not filters_expanded,
            )
        else:
            st.caption(f"No slot data for {_esc(sel_pub)}.")
    else:
        stg_display = stg_full if stg_full and stg_full != "{}" else \
                      _compute_slot_type_groups(ext, matched_art["id"])
        if stg_display and stg_display != "{}":
            card.render_slot_grammar_typed(stg_display, compact=not filters_expanded)
        else:
            st.caption("No slot data available for this query.")
