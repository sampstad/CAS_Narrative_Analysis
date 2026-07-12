"""Welcome — introduction and workflow explanation."""
import streamlit as st

from components.tray import LEVEL_COLORS, LEVEL_DISPLAY


def render():
    st.title("Swiss German News · Narrative Analysis")
    st.markdown(
        "News articles do not just report facts — they tell stories. "
        "This app makes those stories visible by extracting the **narrative structure** "
        "of each article — who does what to whom, and why — and grouping articles "
        "whose storylines converge into **narrative clusters**. "
        "The same structured representation is used to build a searchable "
        "**entity vocabulary** that lets you trace specific actors, forces, and "
        "ideological frames across the whole corpus."
    )

    st.info(
        "**Getting started** — use the sidebar to navigate. Begin with "
        "**Explore narratives** to browse the cluster map, or **Explore entities** "
        "to look up a specific actor, force, or ideology."
    )

    st.divider()

    # ── Two branches of the app ───────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        st.markdown("### 📖 Narrative analysis")
        st.markdown(
            "Explore automatically generated narrative clusters on an interactive map, "
            "or search for specific narratives and topics. Open any cluster as a full card, "
            "filter by publisher and time period, then compare up to eight narratives "
            "side-by-side."
        )
    with right:
        st.markdown("### 🏷️ Entity analysis")
        st.markdown(
            "Search for specific actors, organisations, ideological frames, or abstract "
            "forces by name. Explore how they are distributed across narrative roles and "
            "publishers, inspect a single entity in depth, or compare multiple entities "
            "head-to-head."
        )

    st.divider()

    # ── How it works — Narratives ─────────────────────────────────────────
    st.markdown("### How it works — narratives")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("🗺️  Explore",
         "Browse auto-generated clusters on the interactive map (macro / meso / micro levels), "
         "or use the three search panels: describe a **narrative proposition**, name a "
         "**topic or entity**, or load the **entire dataset**."),
        ("🔍  Inspect",
         "Open any cluster or search result as a full narrative card. "
         "Filter by **publisher** (radio) or **week** (click a bar in the timeline). "
         "The profile metrics, emotion bars, and slot grammar update live."),
        ("＋  Select & compare",
         "Add cards to your comparison set with **＋ Compare**. "
         "As soon as two are selected, the **Compare** button appears in the sidebar. "
         "In the Compare view, cards are shown side-by-side with their saved filters."),
        ("↕  Reorder",
         "Use **↑ / ↓** in the sidebar to change the left-to-right column order in the "
         "Compare view. Use **↗** to jump back to any card in Inspect. "
         "Remove cards with **✕**."),
    ]
    for col, (title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"#### {title}")
            st.markdown(desc)

    st.divider()

    # ── How it works — Entities ───────────────────────────────────────────
    st.markdown("### How it works — entities")
    e1, e2, e3 = st.columns(3)
    ent_steps = [
        ("🔎  Search",
         "Type a name to filter the entity vocabulary. Narrow the results by "
         "**narrative role** (Subject, Object, Helper…) and **filler type** "
         "(actor, structural factor, ideology); the bars show how many matching "
         "labels fall in each category."),
        ("🏷️  Inspect",
         "Open an entity to see its full profile: role and class distribution, "
         "sentiment and emotion deviation versus the corpus, publisher breakdown, "
         "and the clusters it appears in most."),
        ("↔️  Compare",
         "Add entities to a comparison set and view their profiles side-by-side — "
         "role usage, publisher bias, and temporal prominence."),
    ]
    for col, (title, desc) in zip([e1, e2, e3], ent_steps):
        with col:
            st.markdown(f"#### {title}")
            st.markdown(desc)

    st.divider()

    # ── What a narrative card shows ───────────────────────────────────────
    st.markdown("### What a narrative card shows")
    nc = st.columns(2)
    with nc[0]:
        st.markdown("**Corpus profile**")
        st.markdown(
            "- Article count and share of the whole corpus\n"
            "- **Sentiment** and four emotion dimensions — **Joy, Anger, Fear, "
            "Sadness** — shown as **deviation bars**: each bar points right when the "
            "narrative scores *above* its baseline and left when *below*\n"
            "- The baseline is the whole corpus, or — when a publisher or week filter "
            "is active — that publisher's or week's own average, so you always see how "
            "this slice differs from its context"
        )
    with nc[1]:
        st.markdown("**Narrative slots**")
        st.markdown(
            "- Six Greimas roles displayed as labelled frequency bars\n"
            "- Labels show the most common **specific actor types** "
            "(e.g. *Swiss Government*, *Corporations*, *Foreign Militaries*) "
            "with a category emoji\n"
            "- The reference bar (grey) shows the same slot across all articles "
            "when a publisher or week filter is active\n"
            "- In compact comparison mode, three bars per slot are shown"
        )

    st.divider()

    # ── What an entity card shows ─────────────────────────────────────────
    st.markdown("### What an entity card shows")
    ec = st.columns(2)
    with ec[0]:
        st.markdown("**Profile**")
        st.markdown(
            "- Number of articles and distinct publishers the entity appears in\n"
            "- A weekly **coverage sparkline** showing when it was prominent\n"
            "- **Role** bars — which narrative slots it tends to fill — and "
            "**Class** bars — actor, structural factor, or ideology\n"
            "- **Sentiment and emotion deviation** versus the corpus average"
        )
    with ec[1]:
        st.markdown("**Publisher breakdown**")
        st.markdown(
            "- **Role share per publisher** — a stacked bar showing how each outlet "
            "casts the entity across the six roles\n"
            "- **Class share per publisher** — how each outlet frames it as actor, "
            "structural factor, or ideology\n"
            "- The clusters the entity appears in most, grouped by level"
        )

    st.divider()

    # ── Narrative types ───────────────────────────────────────────────────
    st.markdown("### Narrative types")
    st.markdown(
        "Every narrative is represented as a card. The badge at the top identifies its type."
    )
    level_info = [
        ("macro",     "Broad topic poles — the major narrative themes across the whole corpus."),
        ("meso",      "Story angles within a macro theme — distinct framings of the same topic."),
        ("micro",     "Fine-grained framings — the most specific narrative units in a meso cluster."),
        ("narrative", "A card built from a free-text semantic search — finds articles whose extracted story structure matches the query."),
        ("topic",     "A card built from selecting specific entity labels from the corpus vocabulary and choosing which narrative roles they must fill."),
        ("corpus",    "The entire dataset — all articles, no filter. Useful for publisher-level baselines."),
    ]
    lcols = st.columns(len(level_info))
    for col, (lvl, desc) in zip(lcols, level_info):
        color   = LEVEL_COLORS.get(lvl, "#888")
        display = LEVEL_DISPLAY.get(lvl, lvl)
        with col:
            col.markdown(
                f'<div style="border:1px solid {color};border-radius:6px;'
                f'padding:8px 12px;background:rgba(0,0,0,0.02);height:100%;">'
                f'<b style="color:{color};font-size:0.72rem;">{display}</b><br><br>'
                f'<span style="font-size:0.85rem;">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    st.caption(
        "The six narrative roles (Subject, Object, Helper, Opponent, Sender, Receiver) "
        "and the full entity taxonomy are defined in the **Glossary** (sidebar)."
    )

