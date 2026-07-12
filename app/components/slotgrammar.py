"""Greimas slot-grammar rendering (static + typed) and slot-type grouping."""
from __future__ import annotations
import pandas as pd
import streamlit as st

import theme
from components.htmlutil import _esc


def parse_top_entities(s: str) -> dict[str, list[str]]:
    """'subject::a ¦ b ¦ c ¦¦ object::d ¦ e' -> {'subject':[a,b,c], 'object':[d,e]}.
    Uses delimiters (' ¦¦ ', ' ¦ ', '::') chosen to not collide with entity text."""
    out: dict[str, list[str]] = {}
    if not isinstance(s, str) or not s.strip():
        return out
    for chunk in s.split(" ¦¦ "):
        chunk = chunk.strip()
        if not chunk or "::" not in chunk:
            continue
        slot, rest = chunk.split("::", 1)
        ents = [e.strip() for e in rest.split(" ¦ ") if e.strip()]
        if ents:
            out[slot.strip()] = ents
    return out


_CORE_SLOTS = ["subject", "object"]


_AUX_SLOTS  = ["sender", "receiver", "helper", "opponent"]


_ROW1_SLOTS = ["subject", "object", "opponent"]


_ROW2_SLOTS = ["sender", "receiver", "helper"]


_SLOT_DISPLAY = {
    "subject":  "Subject",
    "object":   "Object",
    "sender":   "Sender",
    "receiver": "Receiver",
    "helper":   "Helper",
    "opponent": "Opponent",
}


_NON_SPECIFIC_TYPES: frozenset = frozenset({
    # filler types
    "actor", "abstract_force", "ideology",
    # actor categories
    "civil_society", "domestic_state_actor", "foreign_actor",
    "general_public", "market_economic", "media_knowledge",
    "supranational_actor",
    # abstract_force categories
    "deployed_instrument", "structural_condition",
    # ideology categories
    "authority_axis", "cultural_axis", "economic_axis", "legitimation_mode",
})


_SLOT_CSS = """
<style>
.slotgrid {display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:6px 0 2px 0;}
.slotcard {border:1px solid rgba(128,128,128,0.25); border-radius:8px; padding:8px 10px;
           background:rgba(128,128,128,0.06); min-height:150px;}
.slotcard.core {border-color:rgba(90,120,200,0.55); background:rgba(90,120,200,0.10);}
.slotname {font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;
           opacity:0.65; margin-bottom:4px;}
.slotcard.core .slotname {opacity:0.85; font-weight:600;}
.slotents {font-size:0.86rem; line-height:1.4;}
.slotlist {margin:2px 0 0 0; padding-left:14px;}
.slotlist li {margin-bottom:2px;}
.slotaux {display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px;}
.slotrow {display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:10px;}
.slotempty {opacity:0.35; font-style:italic;}
.slot-subject  {border:2px solid #5a72c0; background:rgba(90,114,192,0.28);}
.slot-object   {border:2px solid #3a9a9a; background:rgba(58,154,154,0.18);}
.slot-opponent {border-color:#c04040; background:rgba(192,64,64,0.06);}
.slot-sender   {border-color:#c88a3a; background:rgba(200,138,58,0.06);}
.slot-receiver {border-color:#8a5ac8; background:rgba(138,90,200,0.06);}
.slot-helper   {border-color:#4a9a4a; background:rgba(74,154,74,0.06);}
.slot-subject .slotname, .slot-object .slotname {font-size:0.8rem; opacity:1.0; font-weight:800; color:#5a72c0;}
.slot-object .slotname {color:#3a9a9a;}
.slot-subject .slotents, .slot-object .slotents {color:rgba(10,14,40,0.88);}
.slot-subject .slotname, .slot-object .slotname, .slot-opponent .slotname,
.slot-sender .slotname, .slot-receiver .slotname, .slot-helper .slotname
  {opacity:0.90; font-weight:600;}
.slot-subject  .typegroup {border-left:2px solid rgba(90,114,192,0.45); padding-left:7px;}
.slot-object   .typegroup {border-left:2px solid rgba(58,154,154,0.45); padding-left:7px;}
.slot-opponent .typegroup {border-left:2px solid rgba(192,64,64,0.45); padding-left:7px;}
.slot-sender   .typegroup {border-left:2px solid rgba(200,138,58,0.45); padding-left:7px;}
.slot-receiver .typegroup {border-left:2px solid rgba(138,90,200,0.45); padding-left:7px;}
.slot-helper   .typegroup {border-left:2px solid rgba(74,154,74,0.45); padding-left:7px;}
/* Actantial model grid layout */
.actantial-grid {
  display:grid;
  grid-template-columns:1fr 26px 1fr 26px 1fr;
  grid-template-rows:auto 30px auto;
  align-items:stretch;
  margin:6px 0;
}
.act-arrow-h {
  display:flex; align-items:center; justify-content:center; align-self:center;
  font-size:1.6rem; color:rgba(120,120,120,1.0); line-height:1; font-weight:700;
}
.act-arrow-v {
  display:flex; align-items:center; justify-content:center;
  font-size:1.6rem; color:rgba(120,120,120,1.0); font-weight:700;
}
.act-empty {}
.typegroups {margin-top:5px; display:flex; flex-direction:column; gap:5px;}
.typegroup {}
.tg-bar-wrap {position:relative; height:22px; border-radius:4px; overflow:hidden;
             background:rgba(128,128,128,0.10); margin-bottom:2px;}
.tg-bar {position:absolute; top:0; left:0; height:100%; border-radius:4px; opacity:0.70;}
.tg-bar-label {position:absolute; top:0; left:7px; height:100%; display:flex;
               align-items:center; font-size:0.82rem; font-weight:600;
               white-space:nowrap; overflow:hidden;}
.tg-pct {opacity:0.60; margin-right:5px; font-size:0.75rem; font-weight:normal;}
.tg-meta {font-size:0.69rem; opacity:0.55; margin:0 0 1px 2px;}
.tg-labels-list {margin:0; padding-left:10px; font-size:0.65rem; font-style:italic; opacity:0.55; zoom:0.65;}
.tg-labels-list li {margin-bottom:0; line-height:1.2;}
/* Compact vertical slot stack (comparison view) */
.compact-slots { display:flex; flex-direction:column; gap:3px; margin:3px 0; }
.compact-slots .slotcard { min-height:auto; padding:4px 7px; }
.compact-slots .typegroups { gap:2px; }
.compact-slots .tg-bar-wrap { height:17px; margin-bottom:1px; }
.compact-slots .tg-bar-label { font-size:0.74rem; right:34px; } /* stop before delta */
.compact-slots .tg-pct { font-size:0.68rem; }
.compact-slots .slotname { margin-bottom:2px; font-size:0.68rem; }
</style>
"""


def _slot_html_block(slot: str, ents: list[str], core: bool) -> str:
    cls          = "slotcard core" if core else "slotcard"
    display_name = _SLOT_DISPLAY.get(slot, slot.capitalize())
    if ents:
        items = "".join(f"<li>{_esc(e)}</li>" for e in ents)
        inner = f'<ul class="slotlist">{items}</ul>'
    else:
        inner = '<span class="slotempty">—</span>'
    return (f'<div class="{cls}"><div class="slotname">{display_name}</div>'
            f'<div class="slotents">{inner}</div></div>')


def render_slot_grammar(top_entities: str):
    """Render the six-slot Greimas grammar as a styled HTML grid.
    Subject/Object emphasized on top; the four auxiliaries below, compact."""
    slots = parse_top_entities(top_entities)
    core_html = "".join(
        _slot_html_block(s, slots.get(s, []), core=True) for s in _CORE_SLOTS
    )
    aux_html = "".join(
        _slot_html_block(s, slots.get(s, []), core=False) for s in _AUX_SLOTS
    )
    html = (
        _SLOT_CSS
        + '<div class="slotgrid">' + core_html + "</div>"
        + '<div class="slotaux">' + aux_html + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def parse_slot_type_groups(json_str: str) -> dict:
    """Parse the slot_type_groups JSON column from cluster_meta."""
    import json
    if not isinstance(json_str, str) or not json_str.strip():
        return {}
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return {}


SPECIFIC_TYPE_LABELS: dict[str, str] = {
    # ── Filler types ─────────────────────────────────────────────────────
    "actor":                                            "Actors",
    "abstract_force":                                   "Structural Factors",
    "ideology":                                         "Ideological Frames",
    # ── Actor categories ─────────────────────────────────────────────────
    "civil_society":                                    "Civil Society",
    "domestic_state_actor":                             "Swiss State",
    "foreign_actor":                                    "Foreign State",
    "general_public":                                   "General Public",
    "market_economic":                                  "Business & Market",
    "media_knowledge":                                  "Media & Knowledge",
    "supranational_actor":                              "Supranational Bodies",
    # ── Abstract force categories ─────────────────────────────────────────
    "deployed_instrument":                              "Instruments & Tools",
    "structural_condition":                             "Structural Conditions",
    # ── Ideology categories ──────────────────────────────────────────────
    "authority_axis":                                   "Social Values",
    "cultural_axis":                                    "Cultural Values",
    "economic_axis":                                    "Economic Values",
    "legitimation_mode":                                "Legitimation Mode",
    # ── civil_society ────────────────────────────────────────────────────
    "civic_association":                                "Civic Associations",
    "civil_society / civic_association":                "Civic Associations",
    "legal_professional":                               "Legal Professionals",
    "civil_society / legal_professional":               "Legal Professionals",
    "ngo_advocacy":                                     "NGOs & Advocacy Groups",
    "civil_society / ngo_advocacy":                     "NGOs & Advocacy Groups",
    "political_party":                                  "Swiss Political Parties",
    "civil_society / political_party":                  "Swiss Political Parties",
    "religious_institution":                            "Religious Institutions",
    "social_movement_initiative":                       "Social Movements",
    "civil_society / social_movement_initiative":       "Social Movements",
    # ── domestic_state_actor (Swiss) ─────────────────────────────────────
    "executive":                                        "Swiss Government",
    "domestic_state_actor / executive":                 "Swiss Government",
    "domestic_executive":                               "Swiss Government",
    "judiciary":                                        "Swiss Judiciary",
    "domestic_state_actor / judiciary":                 "Swiss Judiciary",
    "legislature":                                      "Swiss Parliament",
    "domestic_state_actor / legislature":               "Swiss Parliament",
    "public_administration":                            "Swiss Administration",
    "domestic_state_actor / public_administration":     "Swiss Administration",
    "domestic_public_administration":                   "Swiss Administration",
    "regulatory_agency":                                "Swiss Regulatory Agency",
    "domestic_state_actor / regulatory_agency":         "Swiss Regulatory Agency",
    "security_enforcement":                             "Swiss Security Forces",
    "domestic_state_actor / security_enforcement":      "Swiss Security Forces",
    "domestic_security":                                "Swiss Security Forces",
    # ── foreign_actor ────────────────────────────────────────────────────
    "foreign_civil_society":                            "Foreign Civil Society",
    "foreign_executive":                                "Foreign Governments",
    "foreign_actor / foreign_executive":                "Foreign Governments",
    "foreign_judiciary_regulatory":                     "Foreign Judiciaries",
    "foreign_legislature":                              "Foreign Parliaments",
    "foreign_military":                                 "Foreign Militaries",
    "foreign_paramilitary":                             "Foreign Paramilitaries",
    "foreign_political_party":                          "Foreign Political Parties",
    "foreign_public_administration":                    "Foreign Administrations",
    "foreign_actor / foreign_public_administration":    "Foreign Administrations",
    "foreign_regulatory":                               "Foreign Regulators",
    "foreign_regulatory_agency":                        "Foreign Regulatory Agencies",
    "foreign_security_enforcement":                     "Foreign Security Forces",
    # ── general_public ───────────────────────────────────────────────────
    "affected_community":                               "Affected Communities",
    "general_public / affected_community":              "Affected Communities",
    "citizens_voters":                                  "Citizens & Voters",
    "general_public / citizens_voters":                 "Citizens & Voters",
    "consumers_households":                             "Consumers & Households",
    "individual_unaffiliated":                          "Unaffiliated Individuals",
    "general_public / individual_unaffiliated":         "Unaffiliated Individuals",
    "workers_employees":                                "Workers & Employees",
    # ── market_economic ──────────────────────────────────────────────────
    "corporation_firm":                                 "Corporations",
    "market_economic / corporation_firm":               "Corporations",
    "market_economic_corporation_firm":                 "Corporations",
    "financial_institution":                            "Financial Institutions",
    "individual_entrepreneur":                          "Individual Entrepreneurs",
    "industry_trade_association":                       "Industry Associations",
    "labour_organisation":                              "Labour Organisations",
    # ── media_knowledge ──────────────────────────────────────────────────
    "academic_scientific":                              "Academic & Scientific Institutions",
    "media_knowledge / academic_scientific":            "Academic & Scientific Institutions",
    "individual_expert":                                "Individual Experts",
    "journalist_media_outlet":                          "Journalists & Media Outlets",
    "media_knowledge / journalist_media_outlet":        "Journalists & Media Outlets",
    "media_knowledge_journalist_media_outlet":          "Journalists & Media Outlets",
    "polling_statistics":                               "Polling & Statistics",
    "think_tank":                                       "Think Tanks",
    # ── supranational_actor ──────────────────────────────────────────────
    "eu_institution":                                   "EU Institutions",
    "supranational_actor / eu_institution":             "EU Institutions",
    "intergovernmental_organisation":                   "Intergovernmental Organisations",
    "supranational_actor / intergovernmental_organisation": "Intergovernmental Organisations",
    "intl_financial_institution":                       "International Financial Institutions",
    "intl_humanitarian":                                "International Humanitarian Organisations",
    "intl_security_alliance":                           "International Security Alliances",
    # ── deployed_instrument ──────────────────────────────────────────────
    "communicative_instrument":                         "Communication Tools",
    "economic_instrument":                              "Economic Instruments",
    "legal_instrument":                                 "Legal Instruments",
    "policy_instrument":                                "Policy Instruments",
    "technological_instrument":                         "Technological Instruments",
    # ── structural_condition ─────────────────────────────────────────────
    "cultural_historical":                              "Cultural & Historical Forces",
    "demographic":                                      "Demographic Forces",
    "economic":                                         "Economic Forces",
    "structural_condition / economic":                  "Economic Forces",
    "environmental":                                    "Environmental Forces",
    "structural_condition / environmental":             "Environmental Forces",
    "geopolitical":                                     "Geopolitical Forces",
    "institutional_framework":                          "Institutional Frameworks",
    "technological":                                    "Technological Forces",
    # ── authority_axis ───────────────────────────────────────────────────
    "authoritarian":                                    "Authoritarian Ideology",
    "libertarian":                                      "Liberal Ideology",
    # ── cultural_axis ────────────────────────────────────────────────────
    "communitarian":                                    "Nationalist Ideology",
    "cosmopolitan":                                     "Cosmopolitan Ideology",
    # ── economic_axis ────────────────────────────────────────────────────
    "market_liberal":                                   "Market-Liberal Ideology",
    "redistributive":                                   "Redistributive Ideology",
    # ── legitimation_mode ────────────────────────────────────────────────
    "ecological":                                       "Ecological Legitimation",
    "religious":                                        "Religious Legitimation",
    "technocratic":                                     "Technocratic Legitimation",
    # ── Standalone actors ────────────────────────────────────────────────
    "politician":                                       "Politicians",
    "municipality":                                     "Municipalities",
    "healthcare_professional":                          "Healthcare Professionals",
    "healthcare_provider":                              "Healthcare Providers",
    "judicial_investigator":                            "Judicial Investigators",
    "interest_association":                             "Interest Groups",
    "professional_association":                         "Professional Associations",
    "judiciary_regulatory":                             "Judiciary & Regulators",
    "information_media":                                "Information & Media",
    # ── Misc ─────────────────────────────────────────────────────────────
    "none":                                             "—",
}


def _fmt_type(s: str | None) -> str:
    """Map a raw specific_type / category token to a legible display label."""
    if not s:
        return ""
    return SPECIFIC_TYPE_LABELS.get(s, s.replace("_", " "))


_FTYPE_COLOR = {
    "actor":          "#5a78c8",   # blue
    "abstract_force": "#c8783a",   # amber
    "ideology":       "#8a5ac8",   # purple
}


_FTYPE_COLOR_DEFAULT = "#888888"


_CATEGORY_EMOJI: dict = {
    # actor categories
    "domestic_state_actor": "\U0001f3db\ufe0f",   # 🏛️  Swiss state / government
    "foreign_actor":        "\U0001f30d",          # 🌍  foreign actors
    "civil_society":        "\U0001f91d",          # 🤝  civil society
    "market_economic":      "\U0001f4bc",          # 💼  business & market
    "media_knowledge":      "\U0001f4f0",          # 📰  media & knowledge
    "general_public":       "\U0001f465",          # 👥  general public
    "supranational_actor":  "\U0001f310",          # 🌐  supranational bodies
    # abstract_force categories
    "deployed_instrument":  "\U0001f527",          # 🔧  instruments & tools
    "structural_condition": "\u2699\ufe0f",        # ⚙️  structural conditions
    # ideology categories
    "authority_axis":       "\u2696\ufe0f",        # ⚖️  social/authority values
    "cultural_axis":        "\U0001f3ad",          # 🎭  cultural values
    "economic_axis":        "\U0001f4ca",          # 📊  economic values
    "legitimation_mode":    "\U0001f4dc",          # 📜  legitimation
    # top-level filler types (fallback when no category)
    "actor":                "\U0001f464",          # 👤
    "abstract_force":       "\u26a1",             # ⚡
    "ideology":             "\U0001f4a1",          # 💡
}


def _type_group_html(group: dict, core: bool, pct: float, ref_pct: "float | None" = None, show_labels: bool = True) -> str:
    raw_ftype = group.get("filler_type") or ""
    ftype  = _fmt_type(raw_ftype)
    cat    = _fmt_type(group.get("category"))
    stype  = _fmt_type(group.get("specific_type"))
    labels = group.get("labels", [])
    color  = _FTYPE_COLOR.get(raw_ftype, _FTYPE_COLOR_DEFAULT)

    # Emoji prefix — look up category first, fall back to filler_type
    _emoji = _CATEGORY_EMOJI.get(group.get("category") or raw_ftype, "")

    # Primary label: most specific non-empty tier (no emoji — emoji goes on meta line)
    primary = stype or cat or ftype or "—"

    # Meta line: emoji + remaining tiers (less specific, dimmed)
    meta_parts = []
    if stype and cat:
        meta_parts.append(cat)
    if ftype and (stype or cat):
        meta_parts.append(ftype)
    meta_text = " \u00b7 ".join(meta_parts)
    meta = (f"{_emoji}\u00a0{meta_text}" if (_emoji and meta_text) else meta_text)

    bar_w   = f"{pct:.0f}%"
    pct_str = f"{pct:.0f}%"

    # Delta vs reference (percentage-point diff on the right of the bar)
    delta_html = ""
    if ref_pct is not None:
        delta = pct - ref_pct
        if abs(delta) < 0.5:
            dcol, txt = "#888888", "0%"
        else:
            dcol = "#2d9f2d" if delta >= 0 else "#cc3333"
            sign = "+" if delta >= 0 else ""
            txt  = f"{sign}{delta:.0f}%"
        delta_html = (
            f'<div style="position:absolute;right:6px;top:0;height:100%;'
            f'display:flex;align-items:center;font-size:0.71rem;font-weight:700;'
            f'color:{dcol};white-space:nowrap">{txt}</div>'
        )

    html  = '<div class="typegroup">'
    html += (f'<div class="tg-bar-wrap" title="{_esc(primary)} {pct_str}">'
             f'<div class="tg-bar" style="width:{bar_w};background-color:{color};"></div>'
             f'<div class="tg-bar-label"><span class="tg-pct">{pct_str}</span>'
             f'{_esc(primary)}</div>'
             f'{delta_html}</div>')
    if show_labels and meta:
        html += f'<div class="tg-meta">{_esc(meta)}</div>'
    if show_labels and labels:
        items = "".join(f"<li>{_esc(l)}</li>" for l in labels[:3])
        html += f'<ul class="tg-labels-list">{items}</ul>'
    html += '</div>'
    return html


def _slot_type_card_html(slot: str, groups: list, core: bool, ref_groups: "list | None" = None, compact: bool = False) -> str:
    cls          = f"slotcard slot-{slot}"
    display_name = _SLOT_DISPLAY.get(slot, slot.capitalize())
    # Only show entries with a true specific_type — skip where stype repeats the category or filler_type
    sgroups = [g for g in (groups or [])
               if g.get("specific_type")
               and g["specific_type"] not in _NON_SPECIFIC_TYPES]
    if sgroups:
        top_n = sgroups[:3] if compact else sgroups[:4]
        total = sum(g.get("n_articles", 0) for g in sgroups) or 1  # denominator = specific-type articles only
        # Build reference pct map (specific-type entries only, same denominator logic)
        ref_pct_map: dict = {}
        if ref_groups:
            ref_sgroups = [g for g in ref_groups if g.get("specific_type")]
            ref_total = sum(g.get("n_articles", 0) for g in ref_sgroups) or 1
            for g in ref_sgroups:
                st_key = g.get("specific_type")
                if st_key:
                    ref_pct_map[st_key] = g.get("n_articles", 0) / ref_total * 100

        # Top entries — ref_pct defaults to 0.0 (not None) when ref is provided
        entries = [
            (g, g.get("n_articles", 0) / total * 100,
             ref_pct_map.get(g.get("specific_type"), 0.0) if ref_groups else None)
            for g in top_n
        ]

        inner = ('<div class="typegroups">'
                 + "".join(_type_group_html(g, core, pct, ref_pct, show_labels=not compact) for g, pct, ref_pct in entries)
                 + '</div>')
    else:
        inner = '<span class="slotempty">\u2014</span>'
    return (f'<div class="{cls}"><div class="slotname">{display_name}</div>'
            f'{inner}</div>')


def render_slot_grammar_compact(slot_type_groups_json: str, ref_json: "str | None" = None):
    """Compact vertical-stack slot rendering for comparison view.
    Order: Subject, Object, Helper, Opponent, Sender, Receiver.
    Top-3 specific types per slot, no entity examples, no category labels."""
    groups     = parse_slot_type_groups(slot_type_groups_json)
    ref_groups = parse_slot_type_groups(ref_json) if ref_json else {}
    _ORDER = ["subject", "object", "helper", "opponent", "sender", "receiver"]
    inner = "".join(
        _slot_type_card_html(slot, groups.get(slot, []), slot in _CORE_SLOTS,
                             ref_groups.get(slot), compact=True)
        for slot in _ORDER
    )
    html = _SLOT_CSS + '<div class="compact-slots">' + inner + '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_slot_grammar_typed(slot_type_groups_json: str, ref_json: "str | None" = None, compact: bool = False):
    """Type-first rendering: specific_type as primary, entities as examples.
    Layout: two rows of three columns each.
      Row 1: Subject · Object · Opponent
      Row 2: Sender  · Receiver · Helper
    When ref_json is provided, each bar shows a ↑/↓ delta vs the reference.
    compact=True uses render_slot_grammar_compact (vertical stack, top-3 types)."""
    if compact:
        return render_slot_grammar_compact(slot_type_groups_json, ref_json)
    groups    = parse_slot_type_groups(slot_type_groups_json)
    ref_groups = parse_slot_type_groups(ref_json) if ref_json else {}

    def _card(slot):
        core = slot in _CORE_SLOTS
        return _slot_type_card_html(slot, groups.get(slot, []), core, ref_groups.get(slot), compact=compact)

    _E = '<div class="act-empty"></div>'
    _AH_R = '<div class="act-arrow-h">\u2192</div>'   # →
    _AH_L = '<div class="act-arrow-h">\u2190</div>'   # ←
    _AV_D = '<div class="act-arrow-v">\u2193</div>'   # ↓

    html = (
        _SLOT_CSS
        + '<div class="actantial-grid">'
        # Row 1: Helper → Subject ← Opponent
        + _card("helper") + _AH_R + _card("subject") + _AH_L + _card("opponent")
        # Arrow row: only centre cell has ↓
        + _E + _E + _AV_D + _E + _E
        # Row 2: Sender → Object → Receiver
        + _card("sender") + _AH_R + _card("object") + _AH_R + _card("receiver")
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


_SGI_SLOT_COLOR = theme.SLOT_COLOR


def _slot_groups_from_ext(
    ext: pd.DataFrame,
    article_ids: "list | pd.Series",
) -> dict:
    """Compute per-slot specific_type groups from raw extractions (filler_rank 1)."""
    _SLOTS = ["subject", "object", "sender", "receiver", "helper", "opponent"]
    ids_set = set(str(i) for i in article_ids)
    sub = ext[
        ext["narrative_present"]
        & ext["slot"].isin(_SLOTS)
        & (ext["filler_rank"] == 1)
        & ext["id"].astype(str).isin(ids_set)
    ]
    if sub.empty:
        return {}
    result: dict = {}
    for slot in _SLOTS:
        s = sub[sub["slot"] == slot].dropna(subset=["specific_type"])
        if s.empty:
            continue
        groups: list = []
        for stype, g in s.groupby("specific_type"):
            labels: list = []
            if "label" in g.columns:
                labels = (
                    g.dropna(subset=["label"])
                    .groupby("label")["id"].nunique()
                    .sort_values(ascending=False)
                    .head(5)
                    .index.tolist()
                )
            ft  = g["filler_type"].dropna().mode()
            cat = g["category"].dropna().mode()
            groups.append({
                "specific_type": str(stype),
                "filler_type":   ft.iloc[0]  if len(ft)  else None,
                "category":      cat.iloc[0] if len(cat) else None,
                "labels":        labels,
                "n_articles":    int(g["id"].nunique()),
            })
        groups.sort(key=lambda x: x["n_articles"], reverse=True)
        result[slot] = groups
    return result


_CARD_SLOTS = ["subject", "object", "sender", "receiver", "helper", "opponent"]


def _compute_slot_type_groups(ext: pd.DataFrame, ids, max_groups: int | None = 5) -> str:
    """Aggregate slot_type_groups JSON from raw extraction rows for a set of
    article IDs. Mirrors _top_slot_type_groups in precompute_lib.
    max_groups caps results per slot (None = no cap, used for reference data)."""
    import json
    if ext.empty:
        return "{}"
    ids_set = set(str(i) for i in ids)
    sub = ext[ext["id"].astype(str).isin(ids_set)]
    if sub.empty:
        return "{}"
    result = {}
    for slot in _CARD_SLOTS:
        s = sub[sub["slot"] == slot].dropna(subset=["label"]).copy()
        if s.empty:
            continue
        for col in ("filler_type", "category", "specific_type"):
            if col not in s.columns:
                s[col] = None
        s["_ftype"] = s["filler_type"].fillna("")
        s["_cat"]   = s["category"].fillna("")
        s["_stype"] = s["specific_type"].fillna("")
        groups = []
        for (ftype, cat, stype), g in s.groupby(["_ftype", "_cat", "_stype"]):
            if not stype or stype == cat or stype == ftype or stype in {
                "actor", "abstract_force", "ideology",
                "civil_society", "domestic_state_actor", "foreign_actor",
                "general_public", "market_economic", "media_knowledge",
                "supranational_actor", "deployed_instrument", "structural_condition",
                "authority_axis", "cultural_axis", "economic_axis", "legitimation_mode",
            }:
                continue
            top_labels = (
                g.groupby("label")["id"].nunique()
                .sort_values(ascending=False)
                .head(5)
                .index.tolist()
            )
            if not top_labels:
                continue
            groups.append({
                "filler_type":   ftype or None,
                "category":      cat   or None,
                "specific_type": stype or None,
                "labels":        top_labels,
                "n_articles":    int(g["id"].nunique()),
            })
        groups.sort(key=lambda x: x["n_articles"], reverse=True)
        if groups:
            result[slot] = groups if max_groups is None else groups[:max_groups]
    return json.dumps(result, ensure_ascii=False)
