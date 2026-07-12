"""Narrative explorer — entry point.

Two-section multipage app:
  Welcome      — introduction and workflow
  ── Narrative analysis ──
  Explore      — cluster map + semantic search + entity search
  Inspect      — full narrative card with publisher/time filters
  Compare      — side-by-side narrative comparison
  ── Entity analysis ──
  Explore      — entity search, distributions, publisher breakdown
  Compare      — side-by-side entity profiles
  ── Reference ──
  Glossary     — definitions of slot roles and entity types
"""
import streamlit as st

import data as D

st.set_page_config(page_title="Narrative explorer", layout="wide")


def _require_data():
    if not D.data_available():
        st.error(
            "Precomputed tables not found. Run `07_precompute.ipynb` first, "
            "then set APP_DATA_DIR to the folder containing the parquet files "
            "(default `data/viz`)."
        )
        st.stop()


# --- view entry points (thin wrappers so navigation stays declarative) ---
def view_welcome():
    from views import welcome
    welcome.render()


def view_explore():
    _require_data()
    from views import explore
    explore.render()


def view_compare():
    _require_data()
    from views import compare
    compare.render()


def view_inspect():
    _require_data()
    from views import inspect as inspect_view
    inspect_view.render()


def view_glossary():
    from views import glossary
    glossary.render()


def view_entity_analysis():
    _require_data()
    from views import entity_analysis
    entity_analysis.render()


def view_entity_compare():
    _require_data()
    from views import entity_compare
    entity_compare.render()


pg_welcome        = st.Page(view_welcome,        title="Welcome",            icon=":material/home:",        default=True)
pg_explore        = st.Page(view_explore,        title="Explore narratives", icon=":material/explore:")
pg_inspect        = st.Page(view_inspect,        title="Inspect narrative",  icon=":material/article:")
pg_compare        = st.Page(view_compare,        title="Compare narratives", icon=":material/compare:")
pg_ea_explore     = st.Page(view_entity_analysis,title="Explore entities",   icon=":material/search:")
pg_ea_compare     = st.Page(view_entity_compare, title="Compare entities",   icon=":material/difference:")
pg_glossary       = st.Page(view_glossary,       title="Glossary",           icon=":material/menu_book:")

# Store page references so child views can navigate programmatically
st.session_state["_pg_explore"]    = pg_explore
st.session_state["_pg_inspect"]    = pg_inspect
st.session_state["_pg_compare"]    = pg_compare
st.session_state["_pg_ea_explore"] = pg_ea_explore
st.session_state["_pg_ea_compare"] = pg_ea_compare

# Narrative comparison tray — sidebar widget visible on every page
from components import tray as T
T.render_sidebar_tray(pg_compare)

# Entity comparison tray
from components import entity_tray as ET
ET.render_entity_tray(pg_ea_compare)

nav = st.navigation({
    "": [pg_welcome],
    "Narrative analysis": [pg_explore, pg_inspect, pg_compare],
    "Entity analysis":    [pg_ea_explore, pg_ea_compare],
    "Reference":          [pg_glossary],
})
nav.run()