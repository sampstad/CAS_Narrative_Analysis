"""
Central configuration for the 03_extract notebook.
"""

# --- Paths ---
INPUT_TSV    = "data/preprocessed_dataset.tsv"
STATE_PATH   = "data/extraction_state.json"   # tracks per-article status across runs
RAW_OUTPUT    = "data/extractions_raw.jsonl"   # one raw model output per line
OUTPUT_CSV    = "data/narrative_extractions.csv"
REASONING_CSV = "data/narrative_reasoning.csv"   # article_id + reasoning text for clustering

# --- Filtering ---
MIN_CHARS = 500  # articles below this char_count are excluded

# --- Anthropic API ---
MODEL            = "claude-haiku-4-5-20251001"
MAX_TOKENS       = 4000
TEMPERATURE      = 0
BATCH_SIZE       = 10_000   # Anthropic Batches API max per batch
MAX_RETRIES      = 2        # max retry attempts per article on failure
POLL_INTERVAL    = 20      # seconds between batch status polls

# --- Output schema ---
SLOTS = ["subject", "object", "sender", "receiver", "helper", "opponent"]

_SLOT_ITEM_SCHEMA = {
    "type": "object",
    "required": ["label", "filler_type", "specific_type", "domain", "uncertain"],
    "properties": {
        "label":            {"type": "string"},
        "filler_type":      {"type": "string", "enum": ["actor", "abstract_force", "ideology"]},
        "specific_type":    {"type": "string"},
        "domain":           {"type": "string", "enum": [
                                "political_power", "institutional_governance",
                                "economic_fiscal", "social_welfare",
                                "domestic_security", "international_security",
                                "environmental", "rights_justice",
                                "identity_culture", "science_technology",
                                "information_media", "international_diplomatic"
                            ]},
        "uncertain":        {"type": "boolean"},
        "uncertainty_note": {"type": ["string", "null"]},
    }
}

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["narrative_present", "dominant_narrative", "actantial_slots"],
    "properties": {
        "narrative_present":  {"type": "boolean"},
        "dominant_narrative": {"type": ["string", "null"]},
        "actantial_slots": {
            "type": "object",
            "required": SLOTS,
            "properties": {slot: {"type": "array", "items": _SLOT_ITEM_SCHEMA} for slot in SLOTS}
        }
    }
}


# --- Lookup tables (derived fields — not annotated directly) ---
SUBTYPE_TO_SECTOR = {
    # domestic_state_actor
    "executive":                "domestic_state_actor",
    "legislature":              "domestic_state_actor",
    "judiciary":                "domestic_state_actor",
    "regulatory_agency":        "domestic_state_actor",
    "security_enforcement":     "domestic_state_actor",
    "public_administration":    "domestic_state_actor",
    # market_economic
    "corporation_firm":             "market_economic",
    "financial_institution":        "market_economic",
    "industry_trade_association":   "market_economic",
    "labour_organisation":          "market_economic",
    "individual_entrepreneur":      "market_economic",
    # civil_society
    "ngo_advocacy":                 "civil_society",
    "social_movement_initiative":   "civil_society",
    "political_party":              "civil_society",
    "religious_institution":        "civil_society",
    "civic_association":            "civil_society",
    "legal_professional":           "civil_society",
    # media_knowledge
    "journalist_media_outlet":  "media_knowledge",
    "think_tank":               "media_knowledge",
    "academic_scientific":      "media_knowledge",
    "individual_expert":        "media_knowledge",
    "polling_statistics":       "media_knowledge",
    # general_public
    "citizens_voters":          "general_public",
    "consumers_households":     "general_public",
    "workers_employees":        "general_public",
    "affected_community":       "general_public",
    "individual_unaffiliated":  "general_public",
    # supranational_actor
    "eu_institution":                 "supranational_actor",
    "intergovernmental_organisation": "supranational_actor",
    "intl_humanitarian":              "supranational_actor",
    "intl_security_alliance":         "supranational_actor",
    "intl_financial_institution":     "supranational_actor",
    # foreign_actor
    "foreign_executive":             "foreign_actor",
    "foreign_legislature":           "foreign_actor",
    "foreign_judiciary_regulatory":  "foreign_actor",
    "foreign_public_administration": "foreign_actor",
    "foreign_political_party":       "foreign_actor",
    "foreign_civil_society":         "foreign_actor",
    "foreign_military":              "foreign_actor",
    "foreign_paramilitary":          "foreign_actor",
}

MECHANISM_TO_CATEGORY = {
    # structural_condition
    "institutional_framework":  "structural_condition",
    "economic":                 "structural_condition",
    "demographic":              "structural_condition",
    "technological":            "structural_condition",
    "geopolitical":             "structural_condition",
    "environmental":            "structural_condition",
    "cultural_historical":      "structural_condition",
    # deployed_instrument
    "legal_instrument":         "deployed_instrument",
    "policy_instrument":        "deployed_instrument",
    "economic_instrument":      "deployed_instrument",
    "communicative_instrument": "deployed_instrument",
    "technological_instrument": "deployed_instrument",
}

TRADITION_TO_AXIS = {
    # economic axis
    "market_liberal":   "economic_axis",
    "redistributive":   "economic_axis",
    # cultural-identity axis
    "cosmopolitan":     "cultural_axis",
    "communitarian":    "cultural_axis",
    # authority axis
    "libertarian":      "authority_axis",
    "authoritarian":    "authority_axis",
    # legitimation mode
    "technocratic":     "legitimation_mode",
    "religious":        "legitimation_mode",
    "ecological":       "legitimation_mode",
    # default
    "none":             None,
}