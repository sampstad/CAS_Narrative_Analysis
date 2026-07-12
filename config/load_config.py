"""
Central configuration for the 01_load notebook.
"""

# path to the raw data TSV file downloaded from Swissdox
INPUT_TSV = "data\\raw_dataset.tsv"

# rubrics to be kept for each source
RELEVANT_RUBRICS_PER_SOURCE = {
    "ZWAO": [], # ZWAO has no rubric values in either the rubric column or the URL, therefore the source must be filtered separately (done by using a keyword filter on the URL of each article)
    "AZM": ["Schweiz", "Wirtschaft", "Ausland", "Meinung"],
    "BAZ": ["Politik & Wirtschaft", "International"],
    "BZ": ["Politik & Wirtschaft", "Ausland"],
    "BLIO": ["Ausland", "Schweiz", "Wirtschaft","Politik"],
    "WOZ": ["Schweiz", "International", "Politik", "Wirtschaft"],
    "WEW": ["Diese Woche"],
    "NZZ": ["International", "Wirtschaft", "Meinung und Debatte", "Schweiz"],
    "LUZ": ["Wirtschaft", "Schweiz", "Ausland", "Meinung"],
    "SRF": [], # SRF's rubrics are not in the rubric column but in the URL, therefore the source must be filtered separately
    "SGT": ["Schweiz", "Ausland", "Wirtschaft", "Meinung"],
    "TA": ["Politik & Wirtschaft", "International", "Meinungen"]
}
# rubrics from URL to be kept for SRF
SRF_RELEVANT_RUBRICS_FROM_URL = ["schweiz", "international", "wirtschaft"]

# keywords from URL to be kept for ZWAO (since it has no rubric values at all)
ZWAO_RELEVANT_CONTENT_FROM_URL = [
"schweiz", "schweizer", "bund", "bundesrat", "bundesgericht", "parlament", "nationalrat", "staenderat", "kanton", "kantone", "gemeinde",
"abstimmung", "initiative", "referendum", "wahl", "wahlen", "politik", "politisch", "partei", "svp", "sp", "fdp", "mitte", "gruene", "glp",
"international", "ausland", "eu", "europa", "uno", "nato", "ukraine", "russland", "usa", "china", "nahost", "iran", "israel",
"wirtschaft", "oekonomie", "konjunktur", "inflation", "rezession", "bip", "arbeitsmarkt", "arbeitslosigkeit", "lohn", "preise", "miete",
"boerse", "aktien", "zins", "zinsen", "snb", "bank", "export", "import", "unternehmen", "industrie", "energie", "strom", "gas",
"meinung", "kommentar", "analyse", "kolumne", "gastbeitrag", "debatte", "einordnung", "hintergrund", "leitartikel"
]

# articles below this char_count are excluded
MIN_CHARS = 500  

# path to the output TSV file after filtering
OUTPUT_TSV = "data\\filtered_dataset.tsv"