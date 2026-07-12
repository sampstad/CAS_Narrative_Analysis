"""Explore clusters — 2×2 map grid + narrative card (inline diffusion + publishers).

Left column: 4-panel map grid (article cloud, macro, meso, micro).
Right column: narrative card for the selected cluster; expanders for diffusion
and publisher breakdown.

Sidebar: Browse dropdown to jump directly to any cluster by title.
Click any centroid on the maps to open its card on the right.
"""
import numpy as np
import pandas as pd
import streamlit as st

import data as D
import search_core as SC
from components import card


# ---------------------------------------------------------------------------
# id helpers
# ---------------------------------------------------------------------------
def _node_id(level, cluster):
    return f"{level}:{int(cluster)}"


def _decode_node_id(node_id):
    level, cid = node_id.split(":")
    return level, float(cid)




def _macro_ancestor(meta, level, cluster):
    if level == "macro":
        return cluster
    if level == "meso":
        r = meta[(meta.level == "meso") & (meta.cluster == cluster)]
        return r["parent"].iloc[0] if len(r) else np.nan
    mr = meta[(meta.level == "micro") & (meta.cluster == cluster)]
    if not len(mr):
        return np.nan
    ms = mr["parent"].iloc[0]
    sr = meta[(meta.level == "meso") & (meta.cluster == ms)]
    return sr["parent"].iloc[0] if len(sr) else np.nan



# ---------------------------------------------------------------------------
# main render
# ---------------------------------------------------------------------------
def render():
    st.title("Explore clusters")
    with st.expander("ℹ️ How to read this view", expanded=False):
        st.markdown(
            "Narratives are organised in **three nested levels** of granularity:"
            "  \n- **auto-cluster / macro** \u2014 broad topic poles across the whole corpus"
            "  \n- **auto-cluster / meso** \u2014 distinct story angles within a macro theme"
            "  \n- **auto-cluster / micro** \u2014 fine-grained framings within a meso cluster"
            "  \n\nDot size reflects article count. "
            "Click any centroid to open its narrative card on the right."
        )

    meta = D.load_cluster_meta()
    art  = D.load_article_table()

    if not D.map_available():
        st.warning(
            "2D map not found. Run the `cell-2d-map` step in "
            "`07_precompute.ipynb` to generate `map_xy` and `map_centroids`."
        )
        return

    _browse_panel(meta)
    sel = st.session_state.get("overview_selected")

    highlight = _selection_highlights(meta, sel)
    active    = _compose(highlight, None)

    cmap = _macro_colour_map(meta)

    # Map visibility: reset to visible whenever nothing is selected
    if sel is None:
        st.session_state["maps_visible"] = True
    show_maps = st.session_state.get("maps_visible", True)

    if show_maps:
        xy   = D.load_map_xy()
        _xpad  = (xy["x_2d"].max() - xy["x_2d"].min()) * 0.03
        _ypad  = (xy["y_2d"].max() - xy["y_2d"].min()) * 0.03
        xrange = [xy["x_2d"].min() - _xpad, xy["x_2d"].max() + _xpad]
        yrange = [xy["y_2d"].min() - _ypad, xy["y_2d"].max() + _ypad]

        cloud_highlight = None
        if sel is not None:
            _lv, _cl = sel
            cloud_highlight = set(art[art[f"{_lv}_cluster"] == _cl]["id"])

        all_cents = D.load_map_centroids()
        size_ref  = (float(all_cents["n_articles"].min()),
                     float(all_cents["n_articles"].max()))

        def _count(lvl):
            return len(meta[meta.level == lvl])

        left, right = st.columns([11, 9])

        with left:
            if sel is not None:
                st.button(
                    "\u2190 Hide maps", key="hide_maps_btn",
                    on_click=lambda: st.session_state.update(maps_visible=False),
                )
            r1 = st.columns(2)
            with r1[0]:
                n_cloud = (len(cloud_highlight) if cloud_highlight is not None
                           else art["id"].nunique())
                cap = (f"({n_cloud:,} highlighted)" if cloud_highlight is not None
                       else f"({n_cloud:,} articles)")
                st.markdown(f"**All articles** {cap}")
                _cloud_map(xy, cmap, cloud_highlight, xrange, yrange)
                st.caption("Reference \u2014 coloured by macro theme, not interactive.")
            with r1[1]:
                st.markdown(f"**Macro themes** ({_count('macro')}) \u00b7 click a dot")
                _one_map(meta, xy, cmap, "macro", active.get("macro"), size_ref, xrange, yrange)
            r2 = st.columns(2)
            with r2[0]:
                st.markdown(f"**Meso topics** ({_count('meso')}) \u00b7 click a dot")
                _one_map(meta, xy, cmap, "meso", active.get("meso"), size_ref, xrange, yrange)
            with r2[1]:
                st.markdown(f"**Micro framings** ({_count('micro')}) \u00b7 click a dot")
                _one_map(meta, xy, cmap, "micro", active.get("micro"), size_ref, xrange, yrange)
            _colour_legend(cmap, meta)

        with right:
            _card_panel(meta, art, sel)

    else:
        # Full-width card — maps hidden
        st.button(
            "\U0001f5fa\ufe0f Show maps", key="show_maps_btn",
            on_click=lambda: st.session_state.update(maps_visible=True),
        )
        _card_panel(meta, art, sel)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Search panel — returns {level: set(clusters)} of matches, or None if no query
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Browse panel (sidebar) — jump directly to any cluster by dropdown
# ---------------------------------------------------------------------------
def _browse_panel(meta):
    """Sidebar section that lets the user jump to any cluster by dropdown,
    bypassing the map. Sets overview_selected just like a map click would."""
    sb = st.sidebar
    sb.divider()
    sb.subheader("Browse")

    browse_level = sb.selectbox(
        "Level", D.LEVELS, format_func=str.capitalize,
        key="browse_level", label_visibility="collapsed",
    )
    level_meta = D.cluster_meta_for(meta, browse_level).sort_values("n_articles", ascending=False)

    labels = ["— pick a narrative —"] + [
        f"{r['title']}  \u00b7  {int(r['n_articles'])}  \u00b7  #{int(r['cluster'])}"
        for _, r in level_meta.iterrows()
    ]
    clusters = [None] + [r["cluster"] for _, r in level_meta.iterrows()]

    chosen = sb.selectbox(
        "Narrative", labels, index=0,
        key=f"browse_narrative_{browse_level}",
        label_visibility="collapsed",
    )
    chosen_cluster = clusters[labels.index(chosen)] if chosen in labels else None

    if chosen_cluster is not None:
        new_sel = (browse_level, chosen_cluster)
        if st.session_state.get("overview_selected") != new_sel:
            st.session_state.overview_selected = new_sel
            st.rerun()


# ---------------------------------------------------------------------------
# Selection -> descendant highlight sets
# ---------------------------------------------------------------------------
def _selection_highlights(meta, sel):
    """Given a selected (level, cluster), return {level: set|None} of which
    centroids to emphasise: the selected node and its descendants."""
    out = {"macro": None, "meso": None, "micro": None}
    if sel is None:
        return out
    level, cid = sel
    if level == "macro":
        out["macro"] = {cid}
        meso_kids = meta[(meta.level == "meso") & (meta.parent == cid)]["cluster"]
        out["meso"] = set(meso_kids)
        out["micro"] = set(meta[(meta.level == "micro") &
                                (meta.parent.isin(meso_kids))]["cluster"])
    elif level == "meso":
        out["meso"] = {cid}
        out["micro"] = set(meta[(meta.level == "micro") & (meta.parent == cid)]["cluster"])
        macro_anc = _macro_ancestor(meta, "meso", cid)
        out["macro"] = {macro_anc} if pd.notna(macro_anc) else None
    else:  # micro
        out["micro"] = {cid}
        meso_anc = meta[(meta.level == "micro") & (meta.cluster == cid)]["parent"]
        meso_anc = meso_anc.iloc[0] if len(meso_anc) else np.nan
        out["meso"] = {meso_anc} if pd.notna(meso_anc) else None
        macro_anc = _macro_ancestor(meta, "micro", cid)
        out["macro"] = {macro_anc} if pd.notna(macro_anc) else None
    return out


def _compose(highlight, matched):
    """Combine selection-highlights and search-matches.
    - neither -> None (all normal)
    - only one -> that set
    - both -> intersection (matches within the selected branch)
    """
    out = {}
    for lvl in D.LEVELS:
        h = highlight.get(lvl)
        m = matched.get(lvl) if matched else None
        if h is None and m is None:
            out[lvl] = None
        elif h is None:
            out[lvl] = m
        elif m is None:
            out[lvl] = h
        else:
            out[lvl] = h & m
    return out


# ---------------------------------------------------------------------------
# Rendering one map
# ---------------------------------------------------------------------------
def _macro_colour_map(meta):
    macro_ids = sorted(meta[meta.level == "macro"]["cluster"].unique())
    palette = _qualitative_palette(len(macro_ids))
    return {m: palette[i] for i, m in enumerate(macro_ids)}



def _cloud_map(xy, cmap, cloud_ids, xrange, yrange):
    """Orientation quadrant: the whole article cloud, coloured by macro cluster.
    When a filter is active, in-scope articles are shown at full visibility and
    everything else fades to a faint backdrop (highlight, not filter)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    if "macro_cluster" in xy.columns:
        cloud = xy.dropna(subset=["macro_cluster"]).copy()
        colours = [cmap.get(m, "#888") for m in cloud["macro_cluster"]]
        if cloud_ids is None:
            # no filter: all articles at normal visibility
            fig.add_trace(go.Scattergl(
                x=cloud["x_2d"], y=cloud["y_2d"], mode="markers",
                marker=dict(size=3, opacity=0.30, color=colours),
                hoverinfo="skip", showlegend=False))
        else:
            in_scope = cloud["id"].astype(str).isin(cloud_ids)
            bg = cloud[~in_scope]; fg = cloud[in_scope]
            # faint backdrop of everything else
            fig.add_trace(go.Scattergl(
                x=bg["x_2d"], y=bg["y_2d"], mode="markers",
                marker=dict(size=2, opacity=0.05,
                            color=[cmap.get(m, "#888") for m in bg["macro_cluster"]]),
                hoverinfo="skip", showlegend=False))
            # highlighted scoped articles on top
            fig.add_trace(go.Scattergl(
                x=fg["x_2d"], y=fg["y_2d"], mode="markers",
                marker=dict(size=3, opacity=0.55,
                            color=[cmap.get(m, "#888") for m in fg["macro_cluster"]]),
                hoverinfo="skip", showlegend=False))
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=4, b=0),
                      xaxis=dict(visible=False, range=xrange, autorange=False),
                      yaxis=dict(visible=False, range=yrange, autorange=False),
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False},
                    key="cloud_only_map")


def _one_map(meta, xy, cmap, level, active_set, size_ref, xrange, yrange):
    import plotly.graph_objects as go

    cents = D.load_map_centroids()
    cents = cents[cents["level"] == level].copy()

    lm = D.cluster_meta_for(meta, level).set_index("cluster")
    cents["title"] = cents["cluster"].map(lm["title"])
    cents["sentiment"] = cents["cluster"].map(lm["sentiment"])
    cents["node_id"] = [_node_id(level, c) for c in cents["cluster"]]
    cents["macro_anc"] = [_macro_ancestor(meta, level, c) for c in cents["cluster"]]

    cents["disp_n"] = cents["n_articles"]

    if active_set is None:
        opacity = np.full(len(cents), 0.95)
        line_w = np.full(len(cents), 1.0)
    else:
        opacity = np.where(cents["cluster"].isin(active_set), 0.98, 0.12)
        line_w = np.where(cents["cluster"].isin(active_set), 2.0, 0.3)

    fig = go.Figure()

    # article cloud backdrop
    if "macro_cluster" in xy.columns:
        cloud = xy.dropna(subset=["macro_cluster"])
        if len(cloud):
            fig.add_trace(go.Scattergl(
                x=cloud["x_2d"], y=cloud["y_2d"], mode="markers",
                marker=dict(size=2, opacity=0.05,
                            color=[cmap.get(m, "#888") for m in cloud["macro_cluster"]]),
                hoverinfo="skip", showlegend=False,
            ))

    if len(cents):
        fig.add_trace(go.Scatter(
            x=cents["x"], y=cents["y"], mode="markers",
            marker=dict(
                size=_sizes(cents["disp_n"], size_ref),
                color=[cmap.get(m, "#888") for m in cents["macro_anc"]],
                opacity=opacity,
                line=dict(width=line_w, color="rgba(0,0,0,0.5)"),
            ),
            customdata=np.stack([cents["title"], cents["disp_n"],
                                 cents["sentiment"], cents["node_id"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,} articles"
                          "<br>sentiment %{customdata[2]:+.2f}<extra></extra>",
            showlegend=False,
        ))
    else:
        st.caption(f"No {level} narratives match the active filters.")

    fig.update_layout(height=340, margin=dict(l=0, r=0, t=4, b=0),
                      xaxis=dict(visible=False, range=xrange, autorange=False),
                      yaxis=dict(visible=False, range=yrange, autorange=False),
                      plot_bgcolor="rgba(0,0,0,0)")

    event = st.plotly_chart(
        fig, width='stretch', key=f"map_{level}",
        on_select="rerun", selection_mode="points",
        config={"displayModeBar": False},
    )
    _handle_selection(event)


def _handle_selection(event):
    try:
        pts = event["selection"]["points"] if event and "selection" in event else []
    except (KeyError, TypeError):
        pts = []
    if not pts:
        return
    node_id = None
    for p in pts:
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)) and len(cd) >= 4 and ":" in str(cd[3]):
            node_id = str(cd[3]); break
    if not node_id:
        return
    level, cluster = _decode_node_id(node_id)
    if st.session_state.get("overview_selected") != (level, cluster):
        st.session_state.overview_selected = (level, cluster)
        st.rerun()


def _sizes(n_articles, size_ref):
    """Map article counts to pixel sizes on a GLOBAL scale shared across all
    three maps, so equal article counts render at equal size regardless of level.
    size_ref is (min_n, max_n) computed once over all centroids at all levels.
    Uses sqrt so area ~ article count (perceptually fairer than linear radius)."""
    n = np.asarray(n_articles, dtype=float)
    lo, hi = size_ref
    if hi <= lo:
        return np.full_like(n, 14.0)
    frac = (np.sqrt(n) - np.sqrt(lo)) / (np.sqrt(hi) - np.sqrt(lo))
    frac = np.clip(frac, 0.0, 1.0)
    return 6 + frac * 34.0   # 6..40 px, globally consistent


def _qualitative_palette(k):
    import plotly.express as px
    base = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    return [base[i % len(base)] for i in range(k)]


def _colour_legend(cmap, meta):
    """Compact dot-label legend for macro cluster colours, shown below the 2×2 grid."""
    macro = D.cluster_meta_for(meta, "macro").sort_values("cluster")
    items = []
    for _, row in macro.iterrows():
        col   = cmap.get(row["cluster"], "#888")
        label = str(row["title"])[:38]
        dot   = (f"<span style='display:inline-block;width:10px;height:10px;"
                 f"border-radius:50%;background:{col};vertical-align:middle;"
                 f"margin-right:4px;'></span>")
        items.append(
            f"<span style='white-space:nowrap;margin-right:18px;font-size:0.76rem;"
            f"opacity:0.85;'>{dot}{label}</span>"
        )
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;row-gap:4px;"
        "margin-top:6px;padding:4px 0;border-top:1px solid rgba(128,128,128,0.2);'>"
        + "".join(items) + "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
def _card_panel(meta, art, sel):
    """Right panel: placeholder when nothing selected, full card otherwise."""
    if sel is None:
        st.markdown("#### Select a narrative")
        st.markdown(
            "Select a narrative to explore its details here:\n"
            "- **Click** any cluster dot on the maps\n"
            "- **Browse** via the dropdown in the sidebar\n\n"
            "Use **Search** (the other page) to find narratives by theme."
        )
        return

    import scope as SCP
    level, cluster = sel
    row = D.get_cluster_row(meta, level, cluster)
    if row is None:
        st.warning("Cluster not found.")
        return

    cluster_art = art[art[f"{level}_cluster"] == cluster]

    top = st.columns([7, 1])
    top[0].caption("Descendants highlighted on the maps.")
    top[1].button("\u2715", key="close_card",
                  on_click=lambda: st.session_state.update(overview_selected=None),
                  help="Clear selection")

    level_meta = meta[meta["level"] == level]
    corpus_avg = {
        "sentiment":  float(level_meta["sentiment"].mean()),
        "complexity": float(level_meta["complexity"].mean()),
        "anger":      float(level_meta["anger"].mean()),
        "fear":       float(level_meta["fear"].mean()),
        "joy":        float(level_meta["joy"].mean()),
        "sadness":    float(level_meta["sadness"].mean()),
    }
    card.render_card(
        meta_row=row, article_table=art,
        cluster_week=D.load_cluster_week(),
        cluster_publisher=D.load_cluster_publisher(),
        level=level, relative=True, corpus_avg=corpus_avg,
    )
