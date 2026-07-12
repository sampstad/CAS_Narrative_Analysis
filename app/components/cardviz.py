"""Plotly figures for narrative cards: actantial flow, timeline, publisher timeline."""
from __future__ import annotations
import pandas as pd
import streamlit as st

import theme


_FLOW_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


_FLOW_BOX = {
    "Helper":   (0.000, 0.530, 0.290, 0.970),
    "Subject":  (0.355, 0.530, 0.645, 0.970),
    "Opponent": (0.710, 0.530, 1.000, 0.970),
    "Sender":   (0.000, 0.030, 0.290, 0.470),
    "Object":   (0.355, 0.030, 0.645, 0.470),
    "Receiver": (0.710, 0.030, 1.000, 0.470),
}


_FLOW_SLOT_LINE = {
    "Helper": "#4a9a4a", "Subject": "#5a72c0", "Opponent": "#c04040",
    "Sender": "#c88a3a", "Object":  "#3a9a9a", "Receiver": "#8a5ac8",
}


_FLOW_SLOT_FILL = {
    "Helper":   "rgba(74,154,74,0.10)",   "Subject":  "rgba(90,114,192,0.13)",
    "Opponent": "rgba(192,64,64,0.10)",   "Sender":   "rgba(200,138,58,0.08)",
    "Object":   "rgba(58,154,154,0.13)",  "Receiver": "rgba(138,90,200,0.08)",
}


_FLOW_CONNECTIONS = [
    ("Helper",   "right",  "Helper",   "Subject",  "left",   "Helper"),
    ("Opponent", "left",   "Opponent", "Subject",  "right",  "Opponent"),
    ("Subject",  "bottom", "Object",   "Object",   "top",    "Object"),
    ("Object",   "right",  "Object",   "Receiver", "left",   "Receiver"),
]


_FLOW_EDGE_PAD = 0.022


def _flow_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _flow_banded_segs(df_pivot: pd.DataFrame, other_col: str,
                      slot_name: str, edge_side: str) -> dict:
    """Divide one box edge into (Subject_type, other_type) bands."""
    if other_col not in df_pivot.columns:
        return {}
    x0, y0, x1, y1 = _FLOW_BOX[slot_name]
    if edge_side in ("left", "right"):
        fixed = x0 if edge_side == "left" else x1
        lo, hi = y0 + _FLOW_EDGE_PAD, y1 - _FLOW_EDGE_PAD - 0.06
        axis = "y"
    else:
        fixed = y1 if edge_side == "top" else y0
        lo, hi = x0 + _FLOW_EDGE_PAD, x1 - _FLOW_EDGE_PAD
        axis = "x"
    span  = hi - lo
    valid = df_pivot[["Subject", other_col]].dropna()
    if valid.empty:
        return {}
    counts    = valid.groupby(["Subject", other_col]).size()
    subj_tots = counts.groupby(level=0).sum()
    segs: dict = {}
    cursor = lo
    for stype in subj_tots.sort_values(ascending=False).index:
        subj_span = subj_tots[stype] / subj_tots.sum() * span
        if stype not in counts:
            cursor += subj_span
            continue
        other_cnts = counts[stype].sort_values(ascending=False)
        inner = cursor
        for otype, cnt in other_cnts.items():
            seg_len = cnt / subj_tots[stype] * subj_span
            segs[(stype, otype)] = (inner, inner + seg_len, fixed, axis)
            inner += seg_len
        cursor += subj_span
    return segs


def _flow_midpt(seg: tuple) -> tuple:
    lo, hi, fixed, axis = seg
    mid = (lo + hi) / 2
    return (fixed, mid) if axis == "y" else (mid, fixed)


def _flow_half(seg: tuple) -> float:
    return (seg[1] - seg[0]) / 2


def _flow_ctrl(x, y, edge, dist):
    d = dist * 0.40
    return {"right": (x+d, y), "left": (x-d, y),
            "top":   (x, y+d), "bottom": (x, y-d)}[edge]


def _flow_draw_ribbon(fig, x0, y0, x1, y1, src_edge, tgt_edge, hw, fill_color, hover):
    import numpy as np
    import plotly.graph_objects as go
    t    = np.linspace(0, 1, 60)
    dist = float(np.hypot(x1-x0, y1-y0))
    c1x, c1y = _flow_ctrl(x0, y0, src_edge, dist)
    c2x, c2y = _flow_ctrl(x1, y1, tgt_edge, dist)
    bx  = (1-t)**3*x0 + 3*(1-t)**2*t*c1x + 3*(1-t)*t**2*c2x + t**3*x1
    by  = (1-t)**3*y0 + 3*(1-t)**2*t*c1y + 3*(1-t)*t**2*c2y + t**3*y1
    dbx = 3*(1-t)**2*(c1x-x0) + 6*(1-t)*t*(c2x-c1x) + 3*t**2*(x1-c2x)
    dby = 3*(1-t)**2*(c1y-y0) + 6*(1-t)*t*(c2y-c1y) + 3*t**2*(y1-c2y)
    n   = np.hypot(dbx, dby) + 1e-9
    px, py = -dby/n, dbx/n
    xu = bx + hw*px;  yu = by + hw*py
    xl = bx[::-1] - hw*px[::-1];  yl = by[::-1] - hw*py[::-1]
    fig.add_trace(go.Scatter(
        x=np.concatenate([xu, xl, xu[:1]]),
        y=np.concatenate([yu, yl, yu[:1]]),
        fill="toself", fillcolor=fill_color,
        line=dict(width=0.5, color=fill_color),
        mode="lines", showlegend=False,
        hovertemplate=hover + "<extra></extra>",
    ))


def _build_actantial_fig(df_pivot: pd.DataFrame):
    """Build the actantial flow Plotly figure from a pivoted slot DataFrame."""
    import plotly.graph_objects as go

    if df_pivot.empty or "Subject" not in df_pivot.columns:
        fig = go.Figure()
        fig.add_annotation(text="No slot data", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color="#888"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                          height=260, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        return fig

    # Subject-type color palette
    subj_order = df_pivot["Subject"].dropna().value_counts().index.tolist()
    subj_color = {t: _FLOW_PALETTE[i % len(_FLOW_PALETTE)] for i, t in enumerate(subj_order)}

    fig = go.Figure()

    # ── Ribbons ───────────────────────────────────────────────────────────
    for src_slot, src_edge, src_col, tgt_slot, tgt_edge, tgt_col in _FLOW_CONNECTIONS:
        src_segs = _flow_banded_segs(df_pivot, src_col, src_slot, src_edge)
        tgt_segs = _flow_banded_segs(df_pivot, tgt_col, tgt_slot, tgt_edge)
        if not src_segs or not tgt_segs:
            continue

        if src_col == tgt_col:
            valid = df_pivot[["Subject", src_col]].dropna()
            for (stype, otype), n in valid.groupby(["Subject", src_col]).size().items():
                key = (stype, otype)
                if key not in src_segs or key not in tgt_segs:
                    continue
                x0_, y0_ = _flow_midpt(src_segs[key])
                x1_, y1_ = _flow_midpt(tgt_segs[key])
                hw_ = max(min(_flow_half(src_segs[key]), _flow_half(tgt_segs[key])) * 0.90, 0.002)
                other_lbl = tgt_slot if src_slot == "Subject" else src_slot
                hover = (f"<b>Subject:</b> {stype}<br>"
                         f"<b>{other_lbl}:</b> {otype}<br>{n:,} articles")
                _flow_draw_ribbon(fig, x0_, y0_, x1_, y1_, src_edge, tgt_edge, hw_,
                                  _flow_rgba(subj_color.get(stype, "#888888"), 0.38), hover)
        else:
            valid = df_pivot[["Subject", src_col, tgt_col]].dropna()
            for (stype, src_t, tgt_t), n in valid.groupby(["Subject", src_col, tgt_col]).size().items():
                sk, tk = (stype, src_t), (stype, tgt_t)
                if sk not in src_segs or tk not in tgt_segs:
                    continue
                x0_, y0_ = _flow_midpt(src_segs[sk])
                x1_, y1_ = _flow_midpt(tgt_segs[tk])
                hw_ = max(min(_flow_half(src_segs[sk]), _flow_half(tgt_segs[tk])) * 0.90, 0.002)
                hover = (f"<b>Subject:</b> {stype}<br>"
                         f"<b>{src_slot}:</b> {src_t}<br>"
                         f"<b>{tgt_slot}:</b> {tgt_t}<br>{n:,} articles")
                _flow_draw_ribbon(fig, x0_, y0_, x1_, y1_, src_edge, tgt_edge, hw_,
                                  _flow_rgba(subj_color.get(stype, "#888888"), 0.38), hover)

    # ── Boxes ─────────────────────────────────────────────────────────────
    for slot, (x0, y0, x1, y1) in _FLOW_BOX.items():
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=_FLOW_SLOT_FILL[slot],
                      line=dict(color=_FLOW_SLOT_LINE[slot], width=1.8),
                      layer="above")

    # ── Slot labels + type bars ───────────────────────────────────────────
    for slot, (x0, y0, x1, y1) in _FLOW_BOX.items():
        cx = (x0 + x1) / 2
        tc = _FLOW_SLOT_LINE[slot]
        fig.add_annotation(x=cx, y=y1 - 0.020, text=f"<b>{slot}</b>",
                           font=dict(size=11, color=tc),
                           showarrow=False, xanchor="center", yanchor="top")

        if slot not in df_pivot.columns:
            continue
        counts = df_pivot[slot].value_counts()
        total  = counts.sum()
        top    = counts.head(4)
        bw_    = x1 - x0
        row_h  = min(0.072, (y1 - y0 - 0.10) / max(len(top), 1))

        for i, (typ, cnt) in enumerate(top.items()):
            frac   = cnt / total
            bar_y  = y1 - 0.075 - i * row_h
            bx0_   = x0 + 0.015
            bx1_   = bx0_ + frac * (bw_ - 0.03)
            bar_fc = (_flow_rgba(subj_color.get(typ, tc), 0.55) if slot == "Subject"
                      else _flow_rgba(tc, 0.28))
            fig.add_shape(type="rect",
                x0=bx0_, y0=bar_y - row_h*0.28, x1=x1 - 0.015, y1=bar_y + row_h*0.28,
                fillcolor="rgba(128,128,128,0.07)", line=dict(width=0))
            fig.add_shape(type="rect",
                x0=bx0_, y0=bar_y - row_h*0.28, x1=bx1_, y1=bar_y + row_h*0.28,
                fillcolor=bar_fc, line=dict(width=0))
            fig.add_annotation(
                x=bx0_ + 0.007, y=bar_y,
                text=f"{typ}  <span style='opacity:0.5;font-size:9px'>{frac:.0%}</span>",
                font=dict(size=9.5, color="#222"),
                showarrow=False, xanchor="left", yanchor="middle")

    # ── Arrows ────────────────────────────────────────────────────────────
    for xa, ya, xb, yb in [
        (0.290, 0.750, 0.355, 0.750),
        (0.645, 0.750, 0.710, 0.750),
        (0.500, 0.530, 0.500, 0.470),
        (0.645, 0.250, 0.710, 0.250),
    ]:
        fig.add_annotation(x=xb, y=yb, ax=xa, ay=ya,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                           arrowcolor="rgba(80,80,80,0.45)")
    fig.add_annotation(x=0.355, y=0.690, ax=0.290, ay=0.350,
                       xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                       arrowcolor="rgba(80,80,80,0.45)")

    # ── Subject-type legend ───────────────────────────────────────────────
    for stype, color in list(subj_color.items())[:8]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=10, color=color, symbol="square"),
                                 name=stype, showlegend=True))

    fig.update_layout(
        xaxis=dict(range=[-0.01, 1.01], visible=False, fixedrange=True),
        yaxis=dict(range=[-0.01, 1.01], visible=False, fixedrange=True),
        height=520,
        margin=dict(l=4, r=4, t=38, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(title=dict(text="Subject type", font=dict(size=10)),
                    x=1.01, y=1.0, xanchor="left", font=dict(size=9.5)),
        title=dict(
            text="<b>Actantial flow</b>  "
                 "<sup style='font-size:10px'>colored by Subject type · hover for detail</sup>",
            x=0.01, xanchor="left", font=dict(size=12),
        ),
    )
    return fig


def render_timeline(cluster_week: pd.DataFrame, level: str, cluster: float, relative: bool,
                    key: str = None, static: bool = False, highlight_week: "str | None" = None):
    """Bar chart of weekly prevalence for this cluster.
    relative=False -> article count; True -> within-week share.
    static=True renders a display-only chart (no click-to-filter).
    highlight_week adds a vertical reference line at the given ISO-week value."""
    import plotly.graph_objects as go
    d = cluster_week[(cluster_week["level"] == level) & (cluster_week["cluster"] == cluster)]
    d = d.sort_values("iso_week")
    if d.empty:
        st.caption("No timeline data.")
        return None
    y = d["within_week_share"] if relative else d["n_articles"]
    ylab = "Share of week" if relative else "Articles"
    fig = go.Figure(go.Bar(x=d["iso_week"], y=y))
    fig.update_layout(height=180, margin=dict(l=0, r=0, t=6, b=0),
                      yaxis_title=ylab, xaxis_title=None, showlegend=False)
    if relative:
        fig.update_yaxes(tickformat=".0%")
    if highlight_week:
        fig.add_vline(x=highlight_week, line_dash="dash",
                      line=dict(color="rgba(74,154,74,0.7)", width=1.5))
    if static:
        return st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    return st.plotly_chart(fig, width='stretch', config={"displayModeBar": False},
                           key=key, on_select="rerun", selection_mode="points")


def _render_publisher_timeline(
    art_sub: pd.DataFrame,
    cluster_week: pd.DataFrame,
    level: str,
    cluster: float,
    key: str,
    pub_all: "pd.DataFrame | None" = None,
) -> None:
    """Weekly prevalence for publisher-filtered slice.
    pub_all: all articles for this publisher in the corpus (share denominator).
    Publisher curve: share of publisher corpus per week.
    Reference dashed line: narrative within_week_share (share of total corpus).
    """
    import plotly.graph_objects as go

    weekly_pub = (
        art_sub.groupby("iso_week")["id"].count()
        .reset_index(name="n")
        .sort_values("iso_week")
    ) if not art_sub.empty else pd.DataFrame(columns=["iso_week", "n"])

    ref = cluster_week[
        (cluster_week["level"] == level) & (cluster_week["cluster"] == cluster)
    ].sort_values("iso_week")

    if weekly_pub.empty:
        st.caption("No articles in this selection.")
        return

    if pub_all is not None and "iso_week" in pub_all.columns and not pub_all.empty:
        weekly_pub_total = (
            pub_all.groupby("iso_week")["id"].count().reset_index(name="n_total")
        )
        weekly_pub = weekly_pub.merge(weekly_pub_total, on="iso_week", how="left")
        weekly_pub["share"] = weekly_pub["n"] / weekly_pub["n_total"].clip(lower=1)
        y_pub  = weekly_pub["share"]
        y_ref  = ref["within_week_share"] if (not ref.empty and "within_week_share" in ref.columns) else None
        ylab   = "Share of publisher corpus"
        use_pct = True
    else:
        y_pub  = weekly_pub["n"]
        y_ref  = ref["n_articles"] if not ref.empty else None
        ylab   = "Articles"
        use_pct = False

    fig = go.Figure()
    if y_ref is not None and not ref.empty:
        fig.add_trace(go.Scatter(
            x=ref["iso_week"], y=y_ref,
            mode="lines", name="All publishers",
            line=dict(width=1.5, color="rgba(160,160,160,0.45)", dash="dot"),
        ))
    fig.add_trace(go.Bar(
        x=weekly_pub["iso_week"], y=y_pub,
        name="Selected publisher",
    ))
    fig.update_layout(
        height=200, margin=dict(l=0, r=0, t=6, b=0),
        yaxis_title=ylab, xaxis_title=None,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if use_pct:
        fig.update_yaxes(tickformat=".1%")
    return st.plotly_chart(fig, width='stretch', config={"displayModeBar": False},
                           key=key, on_select="rerun", selection_mode="points")
