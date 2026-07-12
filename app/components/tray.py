"""Comparison tray — shared state and sidebar UI for the narrative comparison set.

Each item in comparison_set is a dict:
  {type: "cluster"|"search", level: str, title: str,
   cluster: float (cluster only), query: str (search only)}
"""
from __future__ import annotations
import streamlit as st

import theme

LEVEL_COLORS = theme.LEVEL_COLORS
_LEVEL_BG = theme.LEVEL_BG
LEVEL_DISPLAY = theme.LEVEL_DISPLAY

MAX_COMPARE = 8  # maximum number of narratives in the comparison set


def _key(item: dict) -> str:
    pub  = item.get("pub") or "all"
    week = str(item.get("week") or "all")
    if item.get("type") == "cluster":
        return f"c_{item.get('level','')}_{int(item.get('cluster', 0))}_{pub}_{week}"
    return f"s_{abs(hash(item.get('query', '')))}_{pub}_{week}"


def _filter_label(item: dict) -> str:
    """Short human-readable filter string for sidebar/badge display."""
    parts = []
    if item.get("pub"):
        parts.append(item["pub"])
    if item.get("week"):
        try:
            import pandas as pd
            d = pd.Timestamp(item["week"])
            parts.append(f"w/{d.strftime('%d %b')}")
        except Exception:
            parts.append(str(item["week"])[:10])
    return " · ".join(parts) if parts else ""


def in_set(item: dict) -> bool:
    return any(_key(x) == _key(item)
               for x in st.session_state.get("comparison_set", []))


def _key_base(item: dict) -> str:
    """Narrative identity key — ignores pub/week filters."""
    if item.get("type") == "cluster":
        return f"c_{item.get('level','')}_{int(item.get('cluster', 0))}"
    return f"s_{abs(hash(item.get('query', '')))}_{item.get('method','')}"


def in_set_base(item: dict) -> bool:
    """True if same narrative (ignoring pub/week) is in comparison set."""
    base = _key_base(item)
    return any(_key_base(x) == base for x in st.session_state.get("comparison_set", []))


def update_filters(item: dict):
    """Replace the base-matching slot's pub/week with current item's values."""
    base = _key_base(item)
    st.session_state["comparison_set"] = [
        item if _key_base(x) == base else x
        for x in st.session_state.get("comparison_set", [])
    ]


def toggle(item: dict):
    cs = st.session_state.setdefault("comparison_set", [])
    k  = _key(item)
    if any(_key(x) == k for x in cs):
        st.session_state["comparison_set"] = [x for x in cs if _key(x) != k]
    elif len(cs) < MAX_COMPARE:
        cs.append(item)


def remove(key: str):
    cs = st.session_state.get("comparison_set", [])
    st.session_state["comparison_set"] = [x for x in cs if _key(x) != key]


# ---------------------------------------------------------------------------
# Sidebar tray — called from app.py so it appears on every page
# ---------------------------------------------------------------------------
def render_sidebar_tray(pg_compare=None):
    cs = st.session_state.get("comparison_set", [])
    if not cs:
        return
    st.sidebar.divider()
    st.sidebar.markdown(f"**Selected narratives ({len(cs)})**")

    # ── Determine the single slot to highlight ────────────────────────────
    # Mirrors inspect.py: build cur_info with live filter state, then find the
    # exact-match key (or the first base-match key for the "↻ Update" target).
    _highlight_key = None
    _explore = st.session_state.get("explore_card") or {}
    if _explore:
        _cur = dict(_explore)
        if _explore.get("type") == "cluster":
            _lv, _cl = _explore.get("level", ""), _explore.get("cluster")
            if _lv and _cl is not None:
                _ks2 = f"_{_lv}_{int(_cl)}"
                _pr  = st.session_state.get(f"pub_sel{_ks2}")
                _cp  = None if (not _pr or _pr == "All") else _pr
                _wk  = None
                try:
                    _ws2 = st.session_state.get(f"card_timeline{_ks2}_{_cp or 'all'}")
                    if _ws2 and _ws2.selection and _ws2.selection.points:
                        _wk = _ws2.selection.points[0].get("x")
                except Exception:
                    pass
                _cur["pub"] = _cp
                _cur["week"] = _wk
        elif _explore.get("type") == "search":
            _pr = st.session_state.get("pseudo_pub_sel")
            _cp = None if (not _pr or _pr == "All") else _pr
            _wk = None
            try:
                _ws2 = st.session_state.get(f"pseudo_card_timeline_{_cp or 'all'}")
                if _ws2 and _ws2.selection and _ws2.selection.points:
                    _wk = _ws2.selection.points[0].get("x")
            except Exception:
                pass
            _cur["pub"] = _cp
            _cur["week"] = _wk
        if in_set(_cur):
            _highlight_key = _key(_cur)          # exact filter match
        elif in_set_base(_cur):
            _base = _key_base(_cur)
            for _x in cs:
                if _key_base(_x) == _base:
                    _highlight_key = _key(_x)    # first base-match (↻ Update target)
                    break

    _tray_q = (
        st.sidebar.text_input("", placeholder="Filter narratives\u2026",
                              key="tray_filter_q", label_visibility="collapsed").strip().lower()
        if len(cs) >= 4 else ""
    )

    for idx, item in enumerate(list(cs)):
        k     = _key(item)
        level = item.get("level", "search")
        color = LEVEL_COLORS.get(level, "#888")
        title = item.get("title", "—")[:50]
        disp  = LEVEL_DISPLAY.get(level, level)
        _active = (k == _highlight_key)
        if _tray_q and _tray_q not in item.get("title", "").lower():
            continue
        _wrap_open  = f'<div style="background:rgba(90,114,192,0.13);border-left:3px solid {color};border-radius:4px;padding:2px 5px;">'
        _wrap_close = '</div>'
        _content = (
            f'<span style="color:{color};font-size:0.65rem;font-weight:700;">'
            f'{disp}</span>'
            f'<br><span style="font-size:0.82rem;">{title}</span>'
            + (f'<br><span style="font-size:0.68rem;opacity:0.6;">{_filter_label(item)}</span>'
               if _filter_label(item) else "")
        )
        c1, cup, cdn, c2, c3 = st.sidebar.columns([3, 1, 1, 1, 1])
        c1.markdown(
            (_wrap_open + _content + _wrap_close) if _active else _content,
            unsafe_allow_html=True,
        )
        # Reorder buttons
        if idx > 0:
            if cup.button("\u2191", key=f"tray_up_{k}", help="Move left in comparison"):
                new_cs = list(cs)
                new_cs[idx - 1], new_cs[idx] = new_cs[idx], new_cs[idx - 1]
                st.session_state["comparison_set"] = new_cs
                st.rerun()
        else:
            cup.write("")
        if idx < len(cs) - 1:
            if cdn.button("\u2193", key=f"tray_dn_{k}", help="Move right in comparison"):
                new_cs = list(cs)
                new_cs[idx], new_cs[idx + 1] = new_cs[idx + 1], new_cs[idx]
                st.session_state["comparison_set"] = new_cs
                st.rerun()
        else:
            cdn.write("")
        pg_inspect = st.session_state.get("_pg_inspect")
        if pg_inspect and c2.button("↗", key=f"tray_ins_{k}", help="Open in Inspect"):
            _navigate_to_inspect(item)
        if c3.button("✕", key=f"tray_rm_{k}", help="Remove"):
            remove(k)
            st.rerun()
    st.sidebar.markdown("")
    if len(cs) >= 2 and pg_compare is not None:
        if st.sidebar.button("\u2696\ufe0f Compare", type="primary", width='stretch',
                             key="tray_go_compare"):
            st.switch_page(pg_compare)
        if len(cs) >= MAX_COMPARE:
            st.sidebar.caption(f"Maximum {MAX_COMPARE} \u2014 remove one to add another.")
        else:
            remaining = MAX_COMPARE - len(cs)
            st.sidebar.caption(
                f"You can add {remaining} more narrative"
                f"{'s' if remaining != 1 else ''} to compare."
            )
    elif pg_compare is not None:
        st.sidebar.caption("Add 1 more narrative to enable comparison.")


def _navigate_to_inspect(item: dict):
    """Pre-populate filter state and switch to the Inspect page."""
    explore_card = {k: item[k] for k in
                    ("type", "level", "title", "cluster", "query", "method", "threshold")
                    if k in item}
    st.session_state["explore_card"] = explore_card
    st.session_state["_inspect_origin"] = "compare"
    # Pre-populate publisher radio so inspect starts with the saved filter
    if item.get("pub"):
        if item.get("type") == "cluster" and item.get("level") and item.get("cluster") is not None:
            _ks = f"_{item['level']}_{int(item['cluster'])}"
            st.session_state[f"pub_sel{_ks}"] = item["pub"]
        elif item.get("type") == "search":
            st.session_state["pseudo_pub_sel"] = item["pub"]
    pg_inspect = st.session_state.get("_pg_inspect")
    if pg_inspect:
        st.switch_page(pg_inspect)

