LABEL_PROMPT_MACRO = """You are analysing a cluster of news articles that share a common narrative pattern at the broadest thematic level.

The following labels have already been assigned to other macro-clusters. Your label must be clearly distinct from all of them:
<existing_labels>
{existing_labels}
</existing_labels>

Before writing the label, think through these questions briefly:
- What is the single overarching theme that unifies this cluster?
- What is the most precise 2-3 word thematic category that captures this narrative type?
- Does any single specific actor, country, or topic appear in more than 50% of the example narratives? If yes, name it. If not, use abstract category nouns that cover the full range.
- How is this cluster distinct from the existing labels listed above? Check every word in your proposed title against the existing labels — if any word repeats, find a synonym.
- Is the title representative of the whole cluster, not just the most memorable example?

Then write a concise label consisting of:
1. A TITLE: 2-3 words, broad abstract noun phrase — no verbs. Use 2 words if they suffice; add a third only if it meaningfully distinguishes this cluster from existing labels. Name a specific actor, country, or topic only if it appears in more than 50% of examples — otherwise use abstract category nouns.
2. A DESCRIPTION: exactly one sentence, strictly under 12 words, capturing the structural tension that defines this narrative type

Avoid:
- Any verbs in the title
- Padding with filler words like "Friction", "Struggles", "Movements", "Dynamics", "Tensions" — only use a third word if it genuinely adds meaning
- The words "Contestation", "Contest" — find more precise vocabulary
- Repeating any word already used in the existing labels above — check every word carefully
- Using "Accountability" more than once — if already used above, try "Oversight", "Scrutiny", "Redress", "Transparency"
- Using "Public" as a modifier — find more specific vocabulary
- Using "Power", "Competition", "Protection" more than once — if already used above, find synonyms
- Naming specific actors, countries, or topics appearing in fewer than 50% of examples
- The word "pursue" or "pursues" in the description
- Enumerating synonyms in the description
- Generic actor-goal formulations
- Passive constructions

Good examples of titles: "Democratic Legitimacy", "State Security", "Ecological Breakdown", "Social Welfare Gaps", "Press Suppression", "Legal Redress", "Electoral Mandates", "Corporate Regulation"
Good examples of descriptions:
- "Elected governments face legitimacy challenges from institutions they nominally control."
- "Ecological advocates clash with economic interests over binding environmental measures."
- "Welfare claimants hit barriers blocking access to support they are owed."
- "Law enforcement confronts structural obstacles to prosecuting organised criminal networks."

Here are the most common entity labels and specific types filling each narrative slot for this cluster:
<slot_fillers>
{slot_fillers}
</slot_fillers>

Here are {n_examples} example dominant narratives from this cluster:
<examples>
{examples}
</examples>

First write your reasoning (2-3 sentences), then respond with this exact JSON on a new line with no markdown formatting:
{{"title": "...", "description": "..."}}"""