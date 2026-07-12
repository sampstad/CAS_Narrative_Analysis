"""Entity comparison tray — sidebar widget for the entity comparison set."""
from __future__ import annotations
import streamlit as st

MAX_ENTITY_COMPARE = 8

_FT_COLORS = {
    "actor":          "#5a72c0",
    "abstract_force": "#c06030",
    "ideology":       "#8a5ac8",
}


def in_set(label: str) -> bool:
    return label in st.session_state.get("entity_compare_set", [])


def toggle(label: str):
    ecs = st.session_state.setdefault("entity_compare_set", [])
    if label in ecs:
        ecs.remove(label)
    else:
        if len(ecs) < MAX_ENTITY_COMPARE:
            ecs.append(label)


def render_entity_tray(pg_ea_compare=None):
    ecs = st.session_state.get("entity_compare_set", [])
    if not ecs:
        return

    import data as D
    ftype_map = D.load_entity_ftype_map()

    st.sidebar.divider()
    st.sidebar.markdown(f"**Entities ({len(ecs)})**")

    for idx, label in enumerate(list(ecs)):
        _content = f'<span style="font-size:0.82rem;">{label[:50]}</span>'
        c1, cup, cdn, c2, c3 = st.sidebar.columns([3, 1, 1, 1, 1])
        c1.markdown(_content, unsafe_allow_html=True)

        if idx > 0:
            if cup.button("\u2191", key=f"etray_up_{idx}", help="Move up"):
                new = list(ecs)
                new[idx - 1], new[idx] = new[idx], new[idx - 1]
                st.session_state["entity_compare_set"] = new
                st.rerun()
        else:
            cup.write("")

        if idx < len(ecs) - 1:
            if cdn.button("\u2193", key=f"etray_dn_{idx}", help="Move down"):
                new = list(ecs)
                new[idx], new[idx + 1] = new[idx + 1], new[idx]
                st.session_state["entity_compare_set"] = new
                st.rerun()
        else:
            cdn.write("")

        pg_ins = st.session_state.get("_pg_ea_explore")
        if pg_ins and c2.button("\u2197", key=f"etray_ins_{idx}", help="Inspect"):
            st.session_state["ea_inspect_label"] = label
            st.switch_page(pg_ins)

        if c3.button("\u2715", key=f"etray_rm_{idx}", help="Remove"):
            ecs.remove(label)
            st.rerun()

    st.sidebar.markdown("")
    if len(ecs) >= 2 and pg_ea_compare is not None:
        if st.sidebar.button("\u21d4 Compare entities", type="primary",
                             width='stretch', key="etray_go_compare"):
            st.switch_page(pg_ea_compare)
        if len(ecs) >= MAX_ENTITY_COMPARE:
            st.sidebar.caption(f"Maximum {MAX_ENTITY_COMPARE} \u2014 remove one to add another.")
        else:
            remaining = MAX_ENTITY_COMPARE - len(ecs)
            st.sidebar.caption(
                f"You can add {remaining} more "
                f"{'entities' if remaining != 1 else 'entity'} to compare."
            )
    elif pg_ea_compare is not None:
        st.sidebar.caption("Add 1 more entity to enable comparison.")
