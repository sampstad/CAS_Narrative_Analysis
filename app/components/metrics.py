"""Metric bars, profile rows, and diverging deviation-bar HTML."""
from __future__ import annotations
import pandas as pd
import streamlit as st

import theme
from components.htmlutil import _esc


_METRIC_CSS = """
<style>
.prof-section { margin:2px 0 8px 0; }
.prof-title { font-size:0.67rem; text-transform:uppercase; letter-spacing:0.07em;
              opacity:0.40; font-weight:700; margin:0 0 5px 0; }
.prof-hl { display:flex; gap:0; align-items:baseline; flex-wrap:wrap; margin:0 0 4px 0; }
.prof-hl-item { display:inline-flex; align-items:baseline; gap:3px; padding-right:14px; }
.prof-hl-label { font-size:0.65rem; color:rgba(120,120,120,0.80); }
.prof-hl-val { font-size:0.82rem; font-weight:700; }
.prof-sep { border:none; border-top:1px solid rgba(128,128,128,0.13); margin:3px 0; }
.prof-row { margin:2px 0 5px 0; }
.prof-row-hd { display:flex; justify-content:space-between; align-items:baseline;
               margin-bottom:2px; }
.prof-label { font-size:0.67rem; color:rgba(120,120,120,0.80);
              overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.prof-val { font-size:0.74rem; font-weight:700; white-space:nowrap; }
.prof-bar-track { height:5px; border-radius:2px;
                  background:rgba(128,128,128,0.12); overflow:hidden; }
.prof-bar-fill { height:100%; border-radius:2px; }
.metric-section { margin:2px 0 6px 0; }
.metric-section-title { font-size:0.67rem; text-transform:uppercase;
                        letter-spacing:0.07em; opacity:0.40; font-weight:700;
                        margin:0 0 5px 0; }
</style>
"""


def _pct_dev(val, avg) -> "float | None":
    """Percent deviation of val from avg. Returns None on NaN / None / zero avg."""
    try:
        v, a = float(val), float(avg)
        return None if (v != v or a != a or a == 0) else (v - a) / abs(a) * 100
    except (TypeError, ValueError):
        return None


def _metric_bar_html(
    label: str,
    value: str,
    bar_pct: "float | None" = None,
    bar_color: str = "#5a72c0",
    delta: "float | None" = None,
    reverse: bool = False,
) -> str:
    """Single-line metric row: label | colored bar | value | delta."""
    if bar_pct is not None:
        bw = f"{min(max(bar_pct, 0), 100):.0f}%"
        bar = (f'<div class="mbar-wrap">'
               f'<div class="mbar" style="width:{bw};background:{bar_color};"></div>'
               f'</div>')
    else:
        bar = '<div class="mbar-wrap"></div>'
    if delta is not None:
        eff = -delta if reverse else delta
        if abs(delta) < 0.5:
            d = '<span class="metric-delta" style="color:#aaa">~</span>'
        else:
            col = "#2d9f2d" if eff >= 0 else "#cc3333"
            sign = "+" if delta >= 0 else ""
            d = f'<span class="metric-delta" style="color:{col}">{sign}{delta:.0f}%</span>'
    else:
        d = '<span class="metric-delta"></span>'
    return (
        f'<div class="metric-row">'
        f'<span class="metric-label">{label}</span>'
        f'{bar}'
        f'<span class="metric-value">{value}</span>'
        f'{d}</div>')


_METRIC_COLORS = {
    "sentiment":  None,
    "joy":        "#3a9a4a",
    "anger":      "#cc3333",
    "fear":       "#c07030",
    "sadness":    "#6080b8",
}


def _bar_for_metric(key: str, value: "float | None") -> "tuple[float | None, str]":
    """Returns (bar_pct 0-100, color) for a metric key and its raw value."""
    if value is None:
        return None, "#888"
    if key == "sentiment":          # -1..+1
        return (value + 1) / 2 * 100, ("#4a7aaa" if value >= 0 else "#cc3333")
    if key in ("joy", "anger", "fear", "sadness"):   # 0..1
        return min(value * 100, 100), _METRIC_COLORS[key]
    return None, "#888"


def _prof_row_html(label: str, val_str: str, key: str,
                   raw: "float | None") -> str:
    """One profile metric row: label + value, then full-width proportional bar."""
    rng = _PROF_METRIC_RANGE.get(key)
    if rng and raw is not None:
        lo, hi = rng
        pct = max(0.0, min(100.0, (raw - lo) / ((hi - lo) or 1.0) * 100))
        if key == "sentiment":
            bar_color = "#5a72c0" if raw >= 0 else "#c04040"
        else:
            _rgb = _PROF_METRIC_RGB.get(key, "128,128,128")
            bar_color = f"rgb({_rgb})" if _rgb else "#5a72c0"
        bar = (
            f'<div class="prof-bar-track">'
            f'<div class="prof-bar-fill"'
            f' style="width:{pct:.1f}%;background:{bar_color};opacity:0.75;"></div>'
            f'</div>'
        )
    else:
        bar = '<div class="prof-bar-track"></div>'
    return (
        f'<div class="prof-row">'
        f'<div class="prof-row-hd">'
        f'<span class="prof-label">{_esc(label)}</span>'
        f'<span class="prof-val">{val_str}</span>'
        f'</div>'
        f'{bar}</div>'
    )


def _prof_headline_html(n_str: str, share_str: str) -> str:
    """One-liner: article count + corpus share (no deviation)."""
    return (
        f'<div class="prof-hl">'
        f'<span class="prof-hl-item">'
        f'<span class="prof-hl-label">Articles\u00a0</span>'
        f'<span class="prof-hl-val">{n_str}</span>'
        f'</span>'
        f'<span class="prof-hl-item">'
        f'<span class="prof-hl-label">Corpus\u00a0</span>'
        f'<span class="prof-hl-val">{share_str}</span>'
        f'</span>'
        f'</div>'
    )


_DEV_METRIC_COLS  = ["sentiment", "joy", "anger", "fear", "sadness"]


_DEV_BAR_CSS = (
    "<style>"
    ".devrow{display:grid;grid-template-columns:30% 1fr 11%;"
    "align-items:center;gap:4px;margin:2px 0;}"
    ".devlabel{font-size:0.67rem;color:rgba(120,120,120,0.80);overflow:hidden;"
    "text-overflow:ellipsis;white-space:nowrap;}"
    ".devtrack{position:relative;height:7px;border-radius:3px;"
    "background:rgba(128,128,128,0.10);}"
    ".devval{font-size:0.72rem;font-weight:700;text-align:right;white-space:nowrap;}"
    "</style>"
)


_DEV_BAR_METRIC_COLS = ["sentiment", "joy", "anger", "fear", "sadness"]


_DEV_BAR_RGB = {
    "sentiment": "90,114,192",   # blue — matches deviation table / metric headers
    "joy":        "74,154,74",   # green
    "anger":      "192,64,64",
    "fear":       "192,96,48",
    "sadness":    "106,90,160",
}


_DEV_BAR_FULL = {
    "sentiment": "Sentiment", "joy": "Joy",
    "anger": "Anger", "fear": "Fear", "sadness": "Sadness",
}


def _dev_bars_html(vals: dict, baseline: dict,
                   cols: "list[str] | None" = None) -> str:
    """Return inline HTML for a set of diverging deviation bar rows."""
    SQ = "'"  # single quote helper for f-string HTML attributes
    rows = []
    for m in (cols or _DEV_BAR_METRIC_COLS):
        v, b = vals.get(m), baseline.get(m)
        if v is None or b is None or float(b) == 0:
            continue
        pct  = (float(v) - float(b)) / abs(float(b)) * 100
        hw   = round(min(50.0, abs(pct)), 1)
        rgb  = _DEV_BAR_RGB.get(m, "128,128,128")
        sign = "+" if pct > 0 else ""
        if pct > 0:
            bar = (f"<div style={SQ}position:absolute;left:50%;top:0;height:100%;"
                   f"width:{hw}%;background:rgb({rgb});"
                   f"border-radius:0 3px 3px 0;opacity:0.80;{SQ}></div>")
            vc = f"rgb({rgb})"
        else:
            bar = (f"<div style={SQ}position:absolute;right:50%;top:0;height:100%;"
                   f"width:{hw}%;background:rgb({rgb});"
                   f"border-radius:3px 0 0 3px;opacity:0.50;{SQ}></div>")
            vc = f"rgb({rgb})"
        rows.append(
            f"<div class={SQ}devrow{SQ}>"
            f"<span class={SQ}devlabel{SQ}>{_DEV_BAR_FULL.get(m, m)}</span>"
            f"<div class={SQ}devtrack{SQ}>"
            f"<div style={SQ}position:absolute;left:50%;top:0;width:1px;height:100%;"
            f"background:rgba(128,128,128,0.30);{SQ}></div>{bar}</div>"
            f"<span class={SQ}devval{SQ} style={SQ}color:{vc};{SQ}>{sign}{pct:.0f}%</span>"
            f"</div>"
        )
    return "".join(rows)
