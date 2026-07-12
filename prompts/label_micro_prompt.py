LABEL_PROMPT_MICRO = """You are analysing a cluster of news articles that share a common structural narrative perspective within a specific topic area.

This micro-cluster sits within the following narrative hierarchy:
<parent_narratives>
Macro: {macro_title} — {macro_description}
Meso: {meso_title} — {meso_description}
</parent_narratives>

The following labels have already been assigned to other micro-clusters within the same meso group. Your label must be clearly distinct from all of them:
<existing_labels>
{existing_labels}
</existing_labels>

Below is the ranked distribution of subject and object entities for this cluster, with individual and cumulative article coverage:
<dominant_entities>
{dominant_entities}
</dominant_entities>

Before writing the label, think through these questions briefly:
- Look at the cumulative coverage column. Does a single named entity cover 50%+ of subject articles? If yes, name it. If coverage is spread thinly across many entities (no entity near 50%, 50% not reached until many entries deep), this is a DISPERSED cluster — do NOT name any specific actor; use the most common specific_type as a role-based noun phrase instead.
- What entity dominates the Subject slot (protagonist)? Does any single entity appear in more than 50% of subject fillers?
- What entity dominates the Object slot (what is being pursued or contested)? Does any single entity appear in more than 50% of object fillers?
- What verb best captures the relationship between Subject and Object in these articles?
- Are there dominant entities (>50%) in any other slots — Opponent, Helper, Sender, Receiver?
- CRITICAL — for each candidate extension slot, ask: does this entity directly relate to the Subject or the Subject's specific quest? An entity that appears in the same articles but serves a different role must NOT be used as an extension. Only include an extension if the entity genuinely fulfils that actantial role in relation to the Subject:
  · Opponent: actively resists or obstructs the Subject's quest
  · Helper: directly assists the Subject in achieving the Object
  · Sender: provides the mandate or ideological justification that motivates the Subject
  · Receiver: directly receives the outcome of the Subject's quest (positively or negatively)
- Does the proposed extension make logical and narrative sense given the Subject and verb?
- How is this cluster distinct from the existing labels listed above?

Then write a concise label consisting of:
1. A TITLE structured as follows:
   · Baseline: [Subject] [present participle verb] [Object]
     — Use the dominant entity label ONLY if one appears in >50% of articles for that slot (check the cumulative coverage above)
     — Otherwise use the most common specific_type as a descriptive noun phrase — do NOT name a specific actor that covers only a minority of the cluster
     — DISPERSED CLUSTER: if coverage is spread thinly (50% not reached until many entities deep), keep BOTH subject and object as role-based specific_type noun phrases; an honest general title covering the whole cluster is better than a specific one covering a fraction
     — Choose the most precise present participle verb to capture the dynamic (e.g. "undermining", "blocking", "demanding", "confronting", "exploiting", "resisting", "attacking", "displacing", "protecting", "invoking")
   · Extension (no comma, only one, only if dominant >50% AND genuinely related to Subject's quest):
     — Pick only the single most dominant qualifying slot
     — Connect using the single most natural and precise connector word — a preposition or short conjunction that best captures the relationship (e.g. "against", "despite", "with", "through", "for", "without", "amid", "over", "via"). Choose freely based on what fits best.
     — If no slot qualifies, no extension — the description handles the rest
   · Keep the full label under 12 words total
   · No finite verbs — the baseline present participle is the only verb form allowed

2. A DESCRIPTION: exactly one sentence, strictly under 12 words, describing the specific framing dynamic. Include any remaining actantially coherent dominant slot fillers not captured in the title. Do not re-describe the meso topic.

Avoid:
- Finite verbs — bad: "Trump bypasses Europe", good: "Trump bypassing European allies"
- A second present participle in the extension — the connector word replaces the verb
- A comma before the extension — bad: "Trump undermining sovereignty, despite resistance", good: "Trump undermining sovereignty despite resistance"
- The words "pursue", "pursues", "targeting", "Contestation", "Contest", "Battles", "Struggles", "Tensions", "Frictions"
- Extensions where the entity does not genuinely fulfil the actantial role in relation to the Subject — bad: "Russia destroying infrastructure with humanitarian NGOs" (NGOs do not assist Russia), good: "Russia destroying infrastructure amid civilian displacement"
- Re-describing the meso topic
- Stacking abstract nouns
- Naming entities appearing in fewer than 50% of fillers for that slot
- Logically or narratively contradictory extensions
- Replicating any of the existing labels above

Good examples of titles:
- "Trump undermining Ukraine sovereignty despite European resistance"
- "Cantonal police pursuing unknown suspects with public witness support"
- "SVP blocking immigration reform through direct democracy"
- "Russia attacking Ukrainian infrastructure amid civilian displacement"
- "Humanitarian actors demanding Gaza ceasefire for civilian protection"
- "Federal Council resisting parliamentary accountability"
- "Prosecutors confronting defendant resistance in Swiss courts"

Good examples of descriptions:
- "US unilateral power bypasses European security guarantees in Ukraine."
- "Swiss cantonal police depend on public witnesses to close cases."
- "SVP frames immigration as existential threat to Swiss identity."
- "Russia frames civilian infrastructure destruction as legitimate military necessity."

Here are the top 10 entity labels and specific types filling each narrative slot for this cluster:
<slot_fillers>
{slot_fillers}
</slot_fillers>

Here are {n_examples} example dominant narratives from this cluster:
<examples>
{examples}
</examples>

First write your reasoning (2-3 sentences), then respond with this exact JSON on a new line with no markdown formatting:
{{"title": "...", "description": "..."}}"""