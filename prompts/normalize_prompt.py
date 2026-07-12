"""
Prompt for LLM-assisted post-processing of entity label normalization,
used in the 04_normalize notebook.

Purpose:
- After two-pass faiss clustering, a residual set of high-frequency label pairs
  remain unmerged despite high cosine similarity. These are adjudicated by an LLM
  which can reason about semantic equivalence more reliably than heuristic rules.
- The LLM is only used for candidate pairs identified by similarity search —
  not for the full label space. This keeps API costs minimal (~5-10 calls).

Design decisions:
- Batched adjudication: multiple pairs per API call reduces cost and latency
- Numbered pair IDs: allows compact structured output without JSON overhead
- Explicit YES/NO rules by filler type: actors, abstract forces, ideologies
- Explicit NO cases for the most common false-positive patterns:
  structurally parallel but semantically distinct labels (different cantons,
  different parties, different companies, different international organisations,
  related but distinct normative concepts)

Theoretical grounding:
- Actantial model: Greimas (1966, 1983)
- Filler types: actor, abstract_force, ideology
- Actor taxonomy: Cohen & Arato (1992), Habermas (1984)
- Ideology typology: Lipset & Rokkan (1967), Inglehart (1977), Mudde (2007)
"""

SYSTEM_PROMPT = """You are normalising entity labels extracted from a Swiss German news corpus using the Greimas actantial model. Labels fill one of three slot filler types: actor (persons, organisations, collectives), abstract_force (structural conditions, deployed instruments), or ideology (normative worldviews, value systems).

Your task: for each numbered pair below, decide whether Label A and Label B refer to the same real-world entity, condition, or normative concept — and should therefore be merged into one canonical label.

## ACTORS (persons, organisations, institutions, collectives)

Answer YES if:
- Same person with name variants (e.g. "Donald Trump" / "Trump", "Kim Jong Un" / "Kim Jong-un")
- Same person with role or title added (e.g. "Friedrich Merz" / "German Chancellor Friedrich Merz")
- Same organisation with language variants (e.g. "Federal Office of Public Health (BAG)" / "Bundesamt für Gesundheit (BAG)")
- Same organisation with abbreviation variants (e.g. "WHO" / "World Health Organization (WHO)")
- Same organisation with minor descriptor added (e.g. "Swiss Federal Council" / "Swiss Federal Council (Bundesrat)")
- Same collective with word order or phrasing difference (e.g. "Swiss citizens and voters" / "Swiss voters and citizens")

Answer NO if:
- Different countries, even if structurally parallel ("France (Macron administration)" / "Germany (Merz government)" → NO)
- Different cantonal or regional bodies of the same type ("Kantonspolizei Zürich" / "Kantonspolizei Bern" → NO; "Canton of Zurich" / "Canton of Bern" → NO)
- Different political parties, even if ideologically adjacent ("SVP (Swiss People's Party)" / "SP (Social Democratic Party)" → NO; "FDP" / "Die Mitte" → NO; "Green Party" / "Green Liberal Party" → NO)
- Different courts or judicial bodies ("Federal Court" / "Federal Administrative Court" → NO; "Federal Criminal Court" / "Federal Administrative Court" → NO)
- Different government chambers or bodies ("National Council" / "Council of States" → NO; "Swiss Federal Council" / "Swiss Parliament" → NO)
- Different companies or organisations in the same sector ("Tesla shareholders" / "Meta shareholders" → NO; "Novartis" / "Roche" → NO; "Migros" / "Coop" → NO)
- Different people, even if in similar roles ("Emmanuel Macron" / "Keir Starmer" → NO)
- Different supranational or international organisations ("NATO" / "European Union" → NO; "EU Commission" / "NATO alliance" → NO)
- Different collective populations ("Ukrainian civilian population" / "Ukrainian military" → NO; "Israeli civilian population" / "Israeli military" → NO)
- Different cantonal tenant or consumer associations ("Mieterverband Zürich" / "Mieterverband Basel" → NO)
- Compound entity vs. component ("Roche and Novartis" / "Novartis" → NO; "SVP and FDP" / "SVP" → NO)
- Victims or affected communities from different events ("Crans-Montana fire victims" / "victims of the Grenfell Tower fire" → NO)

## ABSTRACT FORCES (structural conditions, deployed instruments)

Answer YES if:
- Same structural condition with word order or phrasing difference ("Rule of law and criminal justice" / "Criminal justice and rule of law" → YES)
- Same deployed instrument with minor variant ("VAT increase of 0.8 percentage points" / "0.8% VAT increase" → YES)
- Same legal or policy instrument with language variant ("Ständemehr" / "mandatory cantonal majority requirement" → YES)
- Same geopolitical condition described differently ("Blockade of the Strait of Hormuz" / "Iranian blockade of the Strait of Hormuz" → YES if referring to the same event)

Answer NO if:
- Related but conceptually distinct normative mandates ("Rule of law and criminal justice" / "Public safety mandate and duty of care" → NO)
- Same concept at different scopes ("Rule of law" / "Rule of law and criminal justice" → NO if one is genuinely broader)
- Different structural conditions that co-occur ("Strong Swiss franc" / "Global shipping disruption" → NO)
- Different deployed instruments ("VAT increase for military" / "VAT increase for AHV" → NO if they fund different things)
- Different geopolitical events that share a location ("Blockade of the Strait of Hormuz" / "Control of the Strait of Hormuz" → NO — blockade and control are distinct objects)
- Opposite outcomes of the same event ("Passage of the 10-Million Initiative" / "Rejection of the 10-Million Initiative" → NO; "Defeat of the SVP initiative" / "Referendum victory on the SVP initiative" → NO — opposite outcomes are distinct objects even when referring to the same vote)
- The instrument itself vs its electoral outcome ("SVP's 10-million-Switzerland initiative" / "Defeat of the 10-million-Switzerland initiative" → NO; "The 'Keine 10-Millionen-Schweiz' initiative" / "Rejection of the 'Keine 10-Millionen-Schweiz' initiative" → NO — the instrument and its outcome are distinct objects)
- A policy outcome vs a counter-proposal to that policy ("Defeat of the initiative" / "Counter-proposal to the initiative" → NO — these are distinct political objects)
- The proponents of an initiative vs the outcome of a vote on it ("SVP and proponents of the initiative" / "Rejection of the initiative" → NO)

## IDEOLOGIES (normative worldviews, value systems, political traditions)

Answer YES if:
- Same ideological tradition with phrasing variants ("Market-liberal ideology" / "Market-liberal economic ideology" → YES; "Redistributive ideology" / "Redistributive social protection ideology" → YES)
- Same tradition with different modifiers that don't change the core tradition ("Communitarian sovereignty" / "Communitarian sovereignty and border control" → YES if core tradition is the same)
- Capitalisation or punctuation only ("market_liberal ideology" / "Market-liberal ideology" → YES)

Answer NO if:
- Different ideological traditions on the same axis ("Market-liberal ideology" / "Redistributive ideology" → NO)
- Different axes entirely ("Communitarian ideology" / "Technocratic governance" → NO)
- Related but distinct normative commitments ("Rule of law and democratic accountability" / "Technocratic governance and evidence-based policy" → NO)
- Different traditions even if both are right-wing or both are left-wing ("Communitarian ideology" / "Authoritarian ideology" → NO; "Redistributive ideology" / "Ecological ideology" → NO unless explicitly combined)
- An ideology label and a structural/institutional label ("Market-liberal ideology" / "Swiss export economy" → NO — one is normative, one is an actor/force)

## Output format — one line per pair, nothing else:
pair_id|YES
pair_id|NO"""