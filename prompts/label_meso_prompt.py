LABEL_PROMPT_MESO = """You are analysing a cluster of news articles that share a common narrative pattern around a specific topic.

This meso-cluster belongs to the following macro-level narrative category:
<parent_narrative>
{parent_title}: {parent_description}
</parent_narrative>

The following labels have already been assigned to other meso-clusters in this macro group. Your label must be clearly distinct from all of them:
<existing_labels>
{existing_labels}
</existing_labels>

Below is the ranked distribution of subject and object entities for this cluster, with individual and cumulative article coverage:
<dominant_entities>
{dominant_entities}
</dominant_entities>

Before writing the label, think through these questions:
- What specific topic or domain defines this cluster within the broader macro theme?
- Look at the cumulative coverage column above. Find the anchor that represents 80%+ of the cluster at the most specific level possible:
  · If a single named actor (or actor pair) covers 80%+ of subject articles → name that actor: "Trump's Ukraine Gambit", "Israel-Hamas Escalation"
  · If no single actor reaches 80% but a named place covers 80%+ of subject+object articles → anchor on place: "Gaza Ceasefire Breakdown", "Ukraine Reconstruction"
  · If no place works but a named mechanism or institution covers 80%+ → anchor on that: "Swiss Referendum Campaigns", "NATO Sanctions Policy", "Weko Cartel Enforcement"
  · REMAINDER CLUSTER: If the cumulative coverage is still well below 80% after the top 10 entities — meaning 80% is spread across many unrelated actors, places, and mechanisms — this is a dispersed remainder cluster. Do NOT name any specific actor, place, or mechanism from the distribution. Use a broad role-based label that honestly covers the full structural range of the content, e.g. "International Institutional Disputes", "Cross-Border Legal Standoffs", "Governmental Accountability Clashes". The description should acknowledge the breadth rather than cherry-picking one sub-topic.
  · Never use a pure abstract noun stack with no anchor: forbidden — "Military Escalation", "Security Enforcement", "Policy Diplomacy"
- Which label construction fits best given the anchor you chose?
  · Possessive: "Trump's Ukraine Gambit", "Macron's Survival Crisis", "SVP's Immigration Push"
  · Compound actor-topic: "Israel-Hamas Escalation", "Trump-EU Tariff Standoff"
  · Actor as adjective: "Swiss Cantonal Policing", "Israeli Military Siege"
  · Anchored role-based: "Gaza Ceasefire Breakdown", "NATO's Eastern Deterrence", "Swiss Referendum Campaigns"
- How is this cluster distinct from the existing labels listed above?
- Is the title representative of 80%+ of the cluster, not just the most memorable example?

Then write a concise label consisting of:
1. A TITLE: 3-5 words. Must contain at least one concrete anchor — a named actor, place, institution, or mechanism. No finite verbs. Each word must earn its place.
2. A DESCRIPTION: exactly one sentence, strictly under 12 words, capturing the specific tension or conflict that defines this cluster. Name the key opposing actors or stakes if not already in the title.

Avoid:
- Titles with no concrete anchor — bad: "Military Escalation", "Security Enforcement", "Policy Diplomacy"; good: "Iran-Israel Military Escalation", "Swiss Border Enforcement", "NATO Sanctions Policy"
- The preposition "on" to connect actor and topic — bad: "Trump on Ukraine", good: "Trump's Ukraine Gambit"
- Stacking three or more abstract nouns — bad: "Global Security Coercion Enforcement", good: "Western Military Coercion"
- Finite verbs in the title
- The words "Contestation", "Contest", "Battles", "Struggles", "Tensions", "Frictions", "Campaigns", "Response", "Posture" — use more precise vocabulary
- The word "pursue" or "pursues" in the description
- Padding to reach a word count
- Generic actor-goal formulations in the description
- Passive constructions
- Being too abstract — meso titles must be more specific than the macro parent above
- Replicating any of the existing labels above

Good examples of titles: "Trump's Ukraine Gambit", "SVP's Immigration Push", "Israel-Hamas Escalation", "Gaza Ceasefire Breakdown", "Macron's Coalition Crisis", "Swiss Cantonal Drug Policing", "NATO's Eastern Deterrence", "SBB Infrastructure Delays", "Weko Cartel Enforcement"
Good examples of descriptions:
- "SVP uses direct democracy to force migration caps mainstream parties resist."
- "Trump's bilateral Ukraine diplomacy excludes European actors from peace talks."
- "Macron's weakened presidency battles parliamentary opposition threatening governmental survival."
- "Swiss cantonal police depend on public witnesses to close criminal cases."
- "Netanyahu's military campaign in Gaza defies international humanitarian law pressure."

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