"""Glossary — definitions of actantial roles and entity types (rendered names)."""
import streamlit as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _term(name: str, definition: str, color: str = "#5a72c0") -> str:
    """One term card: rendered name (bold) + definition, colour-keyed left border."""
    return (
        f'<div style="border-left:3px solid {color};padding:4px 9px;'
        f'margin-bottom:5px;background:rgba(0,0,0,0.015);border-radius:0 4px 4px 0;">'
        f'<b style="font-size:0.80rem;color:{color};">{name}</b><br>'
        f'<span style="font-size:0.80rem;opacity:0.82;line-height:1.35;">{definition}</span></div>'
    )


def _grid_html(items: "list[tuple]", color: "str | None" = None) -> str:
    """Build a responsive multi-column grid of term cards (all open, no expanders).
    Each item is (name, definition) with a shared colour, or (name, definition,
    colour) for per-card colours."""
    cards = "".join(
        _term(it[0], it[1], it[2] if len(it) > 2 else color)
        for it in items
    )
    return (
        '<div style="display:grid;'
        'grid-template-columns:repeat(auto-fill,minmax(240px,1fr));'
        'gap:6px 14px;margin-bottom:6px;">' + cards + "</div>"
    )


def _category(name: str, emoji: str, color: str, items: "list[tuple]") -> None:
    """A category header followed by its specific-type grid (fully visible)."""
    st.markdown(
        f'<p style="font-size:0.80rem;font-weight:700;color:{color};'
        f'margin:12px 0 5px 0;">{emoji}&nbsp;{name}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(_grid_html(items, color), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def render():
    st.title("Glossary")
    st.markdown(
        "Definitions of every label the app displays. The annotation scheme is grounded "
        "in Greimas's actantial model (1966) and a multi-level societal actor taxonomy. "
        "The names below are exactly those shown on the narrative and entity cards."
    )

    st.divider()

    # ── Actantial roles ────────────────────────────────────────────────────
    st.markdown("## Actantial Roles")
    st.markdown(
        "Every narrative is analysed through six structural roles (the *actantial model*, "
        "Greimas 1966). The **Subject–Object** axis carries the main quest; "
        "**Sender–Receiver** frame why the quest matters; "
        "**Helper–Opponent** define who enables or blocks it."
    )
    slot_cols = st.columns(2)
    slots_left = [
        ("#5a72c0", "Subject", "The protagonist actively pursuing the Object. "
            "Determined by whose perspective structures the headline and opening frame."),
        ("#3a9a9a", "Object", "What is desired, defended, or at stake — "
            "a concrete goal, mechanism, or normative end-state."),
        ("#4a9a4a", "Helper", "What or who actively and causally enables the Subject's quest. "
            "Mere bystanders or reporters do not qualify."),
    ]
    slots_right = [
        ("#c88a3a", "Sender", "The source of the mandate or value that authorises the quest — "
            "often implicit; infer from narrative logic."),
        ("#8a5ac8", "Receiver", "Who ultimately benefits from (or suffers because of) the outcome."),
        ("#c04040", "Opponent", "What or who resists, blocks, or threatens the Subject. "
            "The Opponent is the opposing party in the dominant conflict, not a side detail."),
    ]
    with slot_cols[0]:
        for color, name, desc in slots_left:
            st.markdown(_term(name, desc, color), unsafe_allow_html=True)
    with slot_cols[1]:
        for color, name, desc in slots_right:
            st.markdown(_term(name, desc, color), unsafe_allow_html=True)

    st.divider()

    # ── Filler types ──────────────────────────────────────────────────────
    st.markdown("## Filler Types")
    st.markdown(
        "Each slot filler is classified into one of three high-level types "
        "(shown as the **Class** of an entity):"
    )
    st.markdown(_grid_html([
        ("Actors",
            "An entity attributed agency: can act, decide, pursue, or be affected. "
            "Has an institutional or social identity independent of the narrative. "
            "Includes individuals, organisations, institutions, and collectives.",
            "#5a72c0"),
        ("Structural Factors",
            "A condition, constraint, or causal pressure that shapes what actors can do "
            "without itself being a willed action by any identifiable agent. "
            "Covers structural conditions (background) and deployed instruments (wielded tools).",
            "#c06030"),
        ("Ideological Frames",
            "A positioned normative worldview that authorises, motivates, or contests action. "
            "Political ideologies, value systems, moral frameworks, and policy goals "
            "with ideological valence.",
            "#8a5ac8"),
    ]), unsafe_allow_html=True)

    st.divider()

    # ── Actor types ───────────────────────────────────────────────────────
    st.markdown("## Actor Types")
    st.markdown(
        "Actors are classified by their social or institutional identity. "
        "The **Swiss State** category applies to Swiss state institutions only — all "
        "non-Swiss governments and state bodies fall under **Foreign State** or "
        "**Supranational Bodies**."
    )

    _category("Swiss State", "🏛️", "#e8a838", [
        ("Swiss Government", "Federal Council, cantonal governments, and heads of state/government acting in an official executive capacity."),
        ("Swiss Parliament", "National Council, Council of States, cantonal parliaments, and legislative committees."),
        ("Swiss Judiciary", "Federal Supreme Court, cantonal courts, and the Staatsanwaltschaft (public prosecutor), which belongs to the judicial branch in Swiss law."),
        ("Swiss Regulatory Agency", "Independent Swiss regulatory bodies: FINMA, BAFU, Preisüberwacher, Swissmedic, and equivalent agencies."),
        ("Swiss Security Forces", "Swiss armed forces, cantonal police, border guard (Grenzwachtkorps), and domestic intelligence services."),
        ("Swiss Administration", "Federal and cantonal ministries, departments, and administrative units acting in a bureaucratic (non-executive, non-legislative) capacity."),
    ])

    _category("Business & Market", "💼", "#5a72c0", [
        ("Corporations", "Private companies, corporations, start-ups, and business enterprises — including foreign firms operating in Switzerland."),
        ("Financial Institutions", "Banks, insurance companies, asset managers, stock exchanges, and other financial intermediaries."),
        ("Industry Associations", "Sector associations, employer federations, and industry lobbying bodies (e.g. economiesuisse, Swissmem)."),
        ("Labour Organisations", "Trade unions, employee associations, and worker collectives (e.g. Unia, SGB)."),
        ("Individual Entrepreneurs", "Individual business owners, self-employed persons, or named executives acting in a private commercial capacity."),
    ])

    _category("Civil Society", "🤝", "#4a9a4a", [
        ("NGOs & Advocacy Groups", "Non-governmental organisations pursuing a specific cause — Amnesty International, WWF, Greenpeace, and equivalent advocacy bodies."),
        ("Social Movements", "Politically mobilised, goal-oriented collective action: referendum committees, protest movements, Initiativkomitees, petition campaigns. Distinguished from Civic Associations by active political mobilisation."),
        ("Swiss Political Parties", "Swiss political parties — SVP, SP, FDP, Mitte, Grüne, GLP, and others — acting as collective entities. Candidates running for office are also coded here."),
        ("Religious Institutions", "Churches, religious communities, and faith-based organisations."),
        ("Civic Associations", "Ongoing membership organisations defined by shared identity or interest: Vereine, cooperatives, neighbourhood associations, alumni groups, sports clubs. Not primarily engaged in political mobilisation."),
        ("Legal Professionals", "Private attorneys, barristers, and legal advocates acting in a non-state capacity (not public prosecutors or judges)."),
    ])

    _category("Media & Knowledge", "📰", "#c06030", [
        ("Journalists & Media Outlets", "Named journalists, news organisations, broadcasters, and online media platforms. Authority derives from institutional affiliation."),
        ("Think Tanks", "Policy research institutes, foundations, and independent research centres producing politically relevant analysis."),
        ("Academic & Scientific Institutions", "Universities, research institutions, and scientific bodies (e.g. Swiss National Science Foundation, IPCC national delegations)."),
        ("Individual Experts", "Domain specialists — academics, doctors, lawyers, engineers — speaking in a personal expert capacity, not as members of an institution."),
        ("Polling & Statistics", "Polling companies, statistical offices (BFS), and monitoring bodies publishing data or survey results."),
    ])

    _category("General Public", "👥", "#888888", [
        ("Citizens & Voters", "The Swiss electorate or citizenry acting as a political principal — e.g. 'das Volk' in a referendum context."),
        ("Consumers & Households", "Private individuals in their economic role as buyers, tenants, or domestic units."),
        ("Workers & Employees", "Wage-earners considered as a labour category, not as organised labour (use Labour Organisations for that)."),
        ("Affected Communities", "A specific population directly affected by the narrative event — residents of a flood zone, patients in a medical story, asylum seekers in a policy debate."),
        ("Unaffiliated Individuals", "A named or unnamed individual who cannot be classified under any institutional category — an ordinary person, an unnamed suspect, a bystander."),
    ])

    _category("Supranational Bodies", "🌐", "#5a72c0", [
        ("EU Institutions", "European Commission, European Parliament, European Council, ECB, and other EU bodies."),
        ("Intergovernmental Organisations", "UN and its agencies (WHO, UNHCR, WFP), OECD, WTO, OSCE, and similar inter-state bodies."),
        ("International Humanitarian Organisations", "ICRC, Médecins Sans Frontières, and other international humanitarian organisations."),
        ("International Security Alliances", "NATO, OSCE security functions, and other international security or defence alliances."),
        ("International Financial Institutions", "IMF, World Bank, BIS, and credit rating agencies (Moody's, S&P, Fitch) — the last given their quasi-regulatory function."),
    ])

    _category("Foreign State", "🌍", "#c04040", [
        ("Foreign Governments", "Heads of state, prime ministers, foreign governments, and heads of foreign agencies acting in an official executive capacity — including the White House and all US federal departments."),
        ("Foreign Parliaments", "Foreign parliaments, legislative chambers, and parliamentary committees."),
        ("Foreign Judiciaries", "Foreign courts (US Supreme Court, German Bundesverfassungsgericht, ECHR) and foreign regulatory bodies (US Fed, foreign financial regulators)."),
        ("Foreign Administrations", "Foreign government ministries, federal agencies, and bureaucracies acting in their own institutional name — distinct from the executive head of government."),
        ("Foreign Political Parties", "Non-Swiss political parties acting as collective entities."),
        ("Foreign Civil Society", "Non-Swiss NGOs, trade unions, and civil society organisations."),
        ("Foreign Militaries", "Official armed forces of a foreign state."),
        ("Foreign Paramilitaries", "Non-state armed groups, militias, and insurgent organisations."),
    ])

    st.divider()

    # ── Structural factor types ───────────────────────────────────────────
    st.markdown("## Structural Factor Types")
    st.markdown(
        "Structural factors divide into **structural conditions** (background forces that obtain "
        "independently of any actor's deliberate action) and **instruments & tools** "
        "(specific tools actively wielded by actors within the narrative)."
    )

    _category("Structural Conditions", "⚙️", "#c06030", [
        ("Institutional Frameworks", "The structural weight of a legal, regulatory, or constitutional order as a background condition: Schengen framework, constitutional constraints, treaty obligations, institutional rules."),
        ("Economic Forces", "Market dynamics, capital flows, financial conditions, price pressures, competitive structures, housing shortages, inequality as structural background. Distinct from deliberate Economic Instruments."),
        ("Demographic Forces", "Population trends, migration flows, ageing, urbanisation, and epidemiological conditions."),
        ("Technological Forces", "Technological change, digital infrastructure, technical systems, and algorithmic forces as background conditions."),
        ("Geopolitical Forces", "International power configurations, alliances, conflict zones, and diplomatic conditions as background context."),
        ("Environmental Forces", "Climate conditions, ecological constraints, and natural resource pressures as background."),
        ("Cultural & Historical Forces", "Historical legacies, cultural norms, and deep-rooted social conventions that constrain action: Swiss political culture, historical memory, entrenched gender norms."),
    ])

    _category("Instruments & Tools", "🔧", "#5a72c0", [
        ("Legal Instruments", "A specific legal tool actively deployed: a subpoena, injunction, court ruling invoked as authority, treaty provision cited, freedom of information request."),
        ("Policy Instruments", "A specific policy or administrative measure actively deployed: VAT increase, budget allocation, regulatory decision, sanctions package, referendum, voting procedure."),
        ("Economic Instruments", "Specific economic tools deliberately deployed: tariffs, subsidies, interest rate decisions, currency interventions. Distinct from Economic Forces as a structural condition."),
        ("Communication Tools", "Specific communicative tools actively deployed: press releases, propaganda campaigns, public statements, disinformation, advertising campaigns."),
        ("Technological Instruments", "Specific technological tools actively deployed: surveillance systems, border technology, algorithmic content moderation, weapons systems."),
    ])

    st.divider()

    # ── Ideological frame types ───────────────────────────────────────────
    st.markdown("## Ideological Frame Types")
    st.markdown(
        "Ideological frames are classified along three value dimensions plus cross-cutting "
        "modes of legitimation. When a narrative claim draws on multiple dimensions, "
        "each is coded as a separate entity in the same slot."
    )

    _category("Economic Values", "📊", "#e8a838", [
        ("Market-Liberal Ideology", "Market freedom, deregulation, competition, fiscal restraint, property rights — the normative claim that outcomes are best determined by markets with minimal state intervention. Economic nationalism and tariff protectionism map onto Nationalist Ideology instead."),
        ("Redistributive Ideology", "Redistribution, labour rights, social protection, welfare state, solidarity — the normative claim that the state should actively correct market inequalities."),
    ])

    _category("Cultural Values", "🎭", "#5a72c0", [
        ("Cosmopolitan Ideology", "Multilateralism, human-rights universalism, openness to cultural diversity, pro-integration, global solidarity — the normative claim that political community extends beyond national borders. Covers Swiss internationalist framing of neutrality."),
        ("Nationalist Ideology", "National identity, cultural continuity, sovereignty, borders, anti-immigration — the normative claim that political community is rooted in shared culture and territory. Covers Swiss sovereigntist framing of neutrality and direct democracy."),
    ])

    _category("Social Values", "⚖️", "#4a9a4a", [
        ("Liberal Ideology", "Individual freedom, civil liberties, limited state power, pluralism, anti-discrimination, and rule of law as a constraint on power."),
        ("Authoritarian Ideology", "Strong state, order, security, deference to authority, and tradition as a source of legitimacy."),
    ])

    _category("Legitimation Mode", "📜", "#8a5ac8", [
        ("Technocratic Legitimation", "Expert authority, evidence-based governance, efficiency, depoliticised management — legitimation through knowledge and competence rather than values or interests."),
        ("Religious Legitimation", "Faith-grounded values, family, moral conservatism, subsidiarity — legitimation through religious or transcendent authority."),
        ("Ecological Legitimation", "Environmental protection, sustainability, post-growth, intergenerational responsibility — legitimation through the natural world and its limits."),
    ])
