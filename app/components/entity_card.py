"""Entity card component — full profile for a set of entity labels."""
from __future__ import annotations
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import theme

_ALL_SLOTS  = ["subject", "object", "helper", "opponent", "sender", "receiver"]
_SLOT_COLOR = theme.SLOT_COLOR
_FT_COLOR   = theme.FT_COLOR
_FT_LABEL   = theme.FT_LABEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tg_bar(label: str, count: int, max_count: int, color: str) -> str:
    """Compact overlay-label bar matching the narrative card slot type style."""
    pct = (count / max_count * 100) if max_count else 0
    return (
        f'<div style="position:relative;height:22px;border-radius:4px;overflow:hidden;'
        f'background:rgba(128,128,128,0.10);margin-bottom:2px;">'
        f'<div style="position:absolute;top:0;left:0;height:100%;width:{pct:.1f}%;'
        f'background:{color};border-radius:4px;opacity:0.70;"></div>'
        f'<div style="position:absolute;top:0;left:7px;right:5px;height:100%;'
        f'display:flex;align-items:center;justify-content:space-between;'
        f'font-size:0.78rem;font-weight:600;white-space:nowrap;">'
        f'<span>{label}</span>'
        f'<span style="opacity:0.60;font-size:0.72rem;font-weight:normal;">{count:,}</span>'
        f'</div>'
        f'</div>'
    )


def _slot_row(slot: str, count: int, max_count: int) -> str:
    return _tg_bar(slot.capitalize(), count, max_count, _SLOT_COLOR.get(slot, "#888"))


def _ft_row(ft: str, count: int, max_count: int) -> str:
    return _tg_bar(_FT_LABEL.get(ft, ft), count, max_count, _FT_COLOR.get(ft, "#888"))


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render_entity_card(
    title: str,
    labels: list[str],
    bridge: pd.DataFrame,
    art: pd.DataFrame,
    ftype_map: dict,
    compact: bool = False,
    key_suffix: str = "",
):
    """Render a full entity card.

    title    — display title (search query or single label)
    labels   — list of canonical entity labels to aggregate
    bridge   — entity_article_bridge (label, id, slot)
    art      — article_table
    ftype_map— label → filler_type dict
    compact  — abbreviated layout for comparison view
    """
    from components.card import _METRIC_CSS
    from components import entity_tray as ET

    labels_set = set(labels)

    # ── Header ────────────────────────────────────────────────────────────
    if not compact:
        st.subheader(title)
    else:
        st.markdown(
            f"<div style='height:60px;overflow:hidden;padding-bottom:4px;'>"
            f"<p style='font-size:1.05rem;font-weight:700;margin:0 0 4px 0;"
            f"line-height:1.25;'>{title}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Comparison toggle ─────────────────────────────────────────────────
    _ecs = st.session_state.get("entity_compare_set", [])
    _in_set = title in _ecs or (len(labels) == 1 and labels[0] in _ecs)
    _cs_full = len(_ecs) >= ET.MAX_ENTITY_COMPARE
    _cmp_label = labels[0] if len(labels) == 1 else title
    if _in_set:
        if st.button("\u2713 In comparison", key=f"ec_toggle{key_suffix}",
                     type="secondary", width='stretch'):
            if _cmp_label in _ecs:
                _ecs.remove(_cmp_label)
            st.rerun()
    elif not _cs_full:
        if st.button("\uff0b Add to comparison", key=f"ec_toggle{key_suffix}",
                     type="primary", width='stretch'):
            ET.toggle(_cmp_label)
            st.rerun()
    else:
        st.caption(f"Max {ET.MAX_ENTITY_COMPARE} reached")

    # ── Scope bridge ──────────────────────────────────────────────────────
    bsub = bridge[bridge["label"].isin(labels_set)].copy()
    if bsub.empty:
        st.caption("No data found for this entity.")
        return

    has_slot = "slot" in bsub.columns
    art_ids  = set(bsub["id"].astype(str))
    art_match = art[art["id"].astype(str).isin(art_ids)].copy()

    # ── Summary stats ─────────────────────────────────────────────────────
    from components.card import _METRIC_CSS, _prof_row_html
    _sm = st.columns(2)
    _sm[0].metric("Articles",   f"{len(art_match):,}")
    _sm[1].metric("Publishers",
                  f"{art_match['publisher'].nunique():,}"
                  if "publisher" in art_match.columns else "\u2014")

    # ── Temporal coverage ────────────────────────────────────────────────────
    if "iso_week" in art_match.columns and len(art_match) > 1:
        _wk_ec = (art_match.groupby("iso_week")["id"].nunique()
                  .sort_index().reset_index())
        _wk_ec.columns = ["week", "n"]
        if len(_wk_ec) > 1:
            import plotly.graph_objects as go
            _fig_wk = go.Figure(go.Bar(
                x=_wk_ec["week"], y=_wk_ec["n"],
                marker_color="rgba(90,114,192,0.50)",
            ))
            _fig_wk.update_layout(
                height=70, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(_fig_wk, width='stretch',
                            config={"displayModeBar": False})

    # ── Slot + filler type profile (Role / Class bars) ────────────────────
    from components.card import _METRIC_CSS
    profile_rows = []
    if has_slot:
        profile_rows.append('<div class="prof-title" style="margin-top:2px;">Role</div>')
        slot_counts = {s: int((bsub["slot"] == s).sum()) for s in _ALL_SLOTS}
        max_slot = max(slot_counts.values()) or 1
        for s in _ALL_SLOTS:
            profile_rows.append(_slot_row(s, slot_counts[s], max_slot))
    profile_rows.append('<div style="height:4px;"></div>')
    profile_rows.append('<div class="prof-title" style="margin-top:2px;">Class</div>')
    ft_col_counts: dict[str, int] = {}
    for l in labels_set:
        ft = ftype_map.get(l, "actor")
        ft_col_counts[ft] = ft_col_counts.get(ft, 0) + int(len(bsub[bsub["label"] == l]))
    max_ft = max(ft_col_counts.values()) if ft_col_counts else 1
    for ft in ["actor", "abstract_force", "ideology"]:
        profile_rows.append(_ft_row(ft, ft_col_counts.get(ft, 0), max_ft))
    st.markdown(
        _METRIC_CSS
        + '<div style="margin:4px 0 8px 0;">'
        + "".join(profile_rows)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Overall deviation bars (entity vs corpus, diverging from centre) ──
    from components.card import _DEV_METRIC_COLS, _dev_bars_html, _DEV_BAR_CSS
    _avail_m = [c for c in _DEV_METRIC_COLS if c in art_match.columns
                and not art_match[c].isna().all()]
    _corpus_means_e = {k: (float(art[k].mean()) if k in art.columns else None)
                       for k in _avail_m}
    _ent_means_e = {k: float(art_match[k].mean()) if not art_match.empty else None
                    for k in _avail_m}

    _overall_bars = _dev_bars_html(_ent_means_e, _corpus_means_e, _avail_m)
    if _overall_bars:
        st.markdown(
            _DEV_BAR_CSS
            + "<div style='margin:4px 0 6px 0;'>"
            + "<div style='font-size:0.67rem;text-transform:uppercase;letter-spacing:0.07em;"
            + "opacity:0.40;font-weight:700;margin:0 0 3px 0;'>vs.\u00a0corpus\u00a0average</div>"
            + _overall_bars + "</div>",
            unsafe_allow_html=True,
        )

    # ── Top clusters for this entity ───────────────────────────────────────
    import data as D_ec
    _cmeta_ec = D_ec.load_cluster_meta()
    _cl_hits_ec = []
    for _lv_ec, _col_ec in [("micro", "micro_cluster"), ("meso", "meso_cluster"), ("macro", "macro_cluster")]:
        if _col_ec not in art_match.columns:
            continue
        _top_cl_ec = art_match[_col_ec].dropna().value_counts().head(3)
        for _cid_ec, _cnt_ec in _top_cl_ec.items():
            _crow_ec = D_ec.get_cluster_row(_cmeta_ec, _lv_ec, float(_cid_ec))
            if _crow_ec is not None:
                _cl_hits_ec.append((_lv_ec, str(_crow_ec["title"])[:55], int(_cnt_ec)))
    if _cl_hits_ec:
        from components.card import level_badge_html
        with st.expander(f"Top clusters ({len(_cl_hits_ec)})",
                         expanded=False):
            for _lv_ec2, _ctitle_ec, _cnt_ec2 in _cl_hits_ec:
                st.markdown(
                    f"{level_badge_html(_lv_ec2)}&nbsp;"
                    f"<span style='font-size:0.82rem;'>{_ctitle_ec}</span>"
                    f"<span style='font-size:0.70rem;opacity:0.5;'> \u00b7 {_cnt_ec2:,} articles</span>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # Pre-compute publisher order + deviation rows (single canonical order)
    _pub_dev_rows_e = []
    _ordered_pubs_e = []
    if _avail_m and "publisher" in art_match.columns:
        # Global corpus order so all entities show same publisher sequence
        _global_order_e = (
            art.groupby("publisher")["id"].nunique()
            .sort_values(ascending=False).index.tolist()
        )
        _entity_pubs_e = set(art_match["publisher"].dropna())
        _ordered_pubs_e = [p for p in _global_order_e if p in _entity_pubs_e]
        for _pub_e in _ordered_pubs_e:
            _pub_art_e  = art_match[art_match["publisher"] == _pub_e]
            _pub_all_e  = art[art["publisher"] == _pub_e]
            _pub_base_e = {k: (float(_pub_all_e[k].mean())
                               if k in _pub_all_e.columns and not _pub_all_e.empty else None)
                           for k in _avail_m}
            _pub_ent_e  = {k: (float(_pub_art_e[k].mean())
                               if not _pub_art_e.empty else None)
                           for k in _avail_m}
            _pub_dev_rows_e.append((_pub_e[:22], _pub_ent_e, _pub_base_e))



    if has_slot and "publisher" in art_match.columns:
        st.markdown("**Publisher breakdown**")

        top_pubs = _ordered_pubs_e or (
            art_match.groupby("publisher")["id"].nunique()
            .sort_values(ascending=False).index.tolist()
        )

        # Join bridge -> art publisher
        bsub_pub = bsub.copy()
        bsub_pub["id"] = bsub_pub["id"].astype(str)
        art_pub = art_match[["id", "publisher"]].copy()
        art_pub["id"] = art_pub["id"].astype(str)
        merged = bsub_pub.merge(art_pub, on="id", how="left")
        merged = merged[merged["publisher"].isin(top_pubs)]

        # Article count per publisher for axis labels
        _pac_e = art_match.groupby("publisher")["id"].nunique()
        _y_labels_e = [f"{p} ({_pac_e.get(p, 0)}" + ")" for p in top_pubs]

        # Role share per publisher (horizontal 100% stacked)
        st.markdown("**Role share per publisher**")
        slot_pub = (
            merged.groupby(["publisher", "slot"])["id"].count()
            .unstack(fill_value=0).reindex(index=top_pubs, fill_value=0)
        )
        slot_pub_norm = slot_pub.div(slot_pub.sum(axis=1).clip(lower=1), axis=0) * 100
        fig_slot = go.Figure()
        for slot in _ALL_SLOTS:
            if slot not in slot_pub_norm.columns:
                continue
            fig_slot.add_trace(go.Bar(
                name=slot.capitalize(),
                y=slot_pub_norm.index.tolist(),
                x=slot_pub_norm[slot].tolist(),
                orientation="h",
                marker_color=_SLOT_COLOR.get(slot, "#888"),
                marker_opacity=0.80,
                hovertemplate=slot.capitalize() + ": %{x:.1f}%<extra></extra>",
            ))
        fig_slot.update_layout(
            barmode="stack", height=max(180, 28 * len(top_pubs)),
            margin=dict(l=0, r=0, t=6, b=0),
            xaxis=dict(ticksuffix="%", range=[0, 100]),
            yaxis=dict(tickvals=top_pubs, ticktext=_y_labels_e, autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font_size=10),
        )
        st.plotly_chart(fig_slot, width='stretch',
                        config={"displayModeBar": False})

        # Class share per publisher (horizontal 100% stacked)
        st.markdown("**Class share per publisher**")
        merged["ftype"] = merged["label"].map(ftype_map).fillna("actor")
        ft_pub = (
            merged.groupby(["publisher", "ftype"])["id"].count()
            .unstack(fill_value=0).reindex(index=top_pubs, fill_value=0)
        )
        ft_pub_norm = ft_pub.div(ft_pub.sum(axis=1).clip(lower=1), axis=0) * 100
        _FT_L2 = {"actor": "Actor", "abstract_force": "Abstract force",
                  "ideology": "Ideology"}
        _FT_C2 = {"actor": "#5a72c0", "abstract_force": "#c06030",
                  "ideology": "#8a5ac8"}
        fig_ft2 = go.Figure()
        for ft in ["actor", "abstract_force", "ideology"]:
            if ft not in ft_pub_norm.columns:
                continue
            fig_ft2.add_trace(go.Bar(
                name=_FT_L2[ft],
                y=ft_pub_norm.index.tolist(),
                x=ft_pub_norm[ft].tolist(),
                orientation="h",
                marker_color=_FT_C2[ft],
                marker_opacity=0.80,
                hovertemplate=_FT_L2[ft] + ": %{x:.1f}%<extra></extra>",
            ))
        fig_ft2.update_layout(
            barmode="stack", height=max(180, 28 * len(top_pubs)),
            margin=dict(l=0, r=0, t=6, b=0),
            xaxis=dict(ticksuffix="%", range=[0, 100]),
            yaxis=dict(tickvals=top_pubs, ticktext=_y_labels_e, autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font_size=10),
        )
        st.plotly_chart(fig_ft2, width='stretch',
                        config={"displayModeBar": False})

        # Per-publisher deviation bars (one bordered section per publisher)
        if _pub_dev_rows_e and _avail_m:
            from components.card import _dev_bars_html, _DEV_BAR_CSS
            st.markdown("**Sentiment & emotions vs. publisher average**")
            for _pub_lbl, _pub_ent, _pub_base in _pub_dev_rows_e:
                _bars = _dev_bars_html(_pub_ent, _pub_base, _avail_m)
                if _bars:
                    st.markdown(
                        _DEV_BAR_CSS
                        + "<div style='border:1px solid rgba(128,128,128,0.18);"
                        + "border-radius:6px;padding:7px 10px;margin:5px 0;'>"
                        + f"<div style='font-size:0.70rem;font-weight:600;"
                        + f"color:rgba(80,80,80,0.65);margin:0 0 4px 0;'>{_pub_lbl}</div>"
                        + _bars + "</div>",
                        unsafe_allow_html=True,
                    )

