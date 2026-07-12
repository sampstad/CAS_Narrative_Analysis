"""HTML escaping helper shared across card components."""
from __future__ import annotations
import pandas as pd
import streamlit as st

import theme


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
