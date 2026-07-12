"""Canonical colour + label palettes shared across the app.

Single source of truth for:
  * SLOT_COLOR   — the six Greimas actantial roles
  * LEVEL_*      — narrative card types (macro/meso/micro/narrative/topic/corpus)
  * FT_*         — entity filler types (actor / abstract_force / ideology)

Import from here instead of re-declaring these dicts locally; historically they
were copy-pasted across modules and drifted out of sync.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Greimas actantial slot colours
# ---------------------------------------------------------------------------
SLOT_COLOR: dict[str, str] = {
    "subject":  "#5a72c0",
    "object":   "#3a9a9a",
    "helper":   "#4a9a4a",
    "opponent": "#c04040",
    "sender":   "#c88a3a",
    "receiver": "#8a5ac8",
}

# ---------------------------------------------------------------------------
# Narrative card types (levels)
# ---------------------------------------------------------------------------
LEVEL_COLORS: dict[str, str] = {
    "macro":     "#e8a838",
    "meso":      "#5a72c0",
    "micro":     "#4a9a4a",
    "narrative": "#8a5ac8",
    "topic":     "#c06030",
    "search":    "#8a5ac8",   # legacy alias
    "corpus":    "#4a9a9a",
}
LEVEL_BG: dict[str, str] = {
    "macro":     "rgba(232,168,56,0.13)",
    "meso":      "rgba(90,114,192,0.13)",
    "micro":     "rgba(74,154,74,0.13)",
    "narrative": "rgba(138,90,200,0.13)",
    "topic":     "rgba(192,96,48,0.13)",
    "search":    "rgba(138,90,200,0.13)",
    "corpus":    "rgba(74,154,154,0.13)",
}
LEVEL_DISPLAY: dict[str, str] = {
    "macro":     "auto-cluster / macro",
    "meso":      "auto-cluster / meso",
    "micro":     "auto-cluster / micro",
    "narrative": "semantic search",
    "topic":     "entity search",
    "search":    "semantic search",
    "corpus":    "entire dataset",
}

# ---------------------------------------------------------------------------
# Entity filler types
# ---------------------------------------------------------------------------
FT_COLOR: dict[str, str] = {
    "actor":          "#5a72c0",
    "abstract_force": "#c06030",
    "ideology":       "#8a5ac8",
}
FT_LABEL: dict[str, str] = {
    "actor":          "Actor",
    "abstract_force": "Abstract force",
    "ideology":       "Ideology",
}
FT_RGB: dict[str, str] = {
    "actor":          "90,114,192",
    "abstract_force": "192,96,48",
    "ideology":       "138,90,200",
}
