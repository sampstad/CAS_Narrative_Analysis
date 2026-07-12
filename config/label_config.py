"""
Central configuration for the 04b_label notebook.
"""

# --- Paths ---
ASSIGNMENTS_CSV       = "data/cluster_assignments.csv"
MACRO_ASSIGNMENTS_CSV = "data/cluster_assignments_macro.csv"
MESO_ASSIGNMENTS_CSV  = "data/cluster_assignments_meso.csv"
MICRO_ASSIGNMENTS_CSV = "data/cluster_assignments_micro.csv"
EXTRACTIONS_CSV       = "data/narrative_extractions_normalized.csv"
PREPROCESSED_CSV      = "data/preprocessed_dataset.tsv"
MACRO_LABELS_CSV      = "data/cluster_labels_macro.csv"
MESO_LABELS_CSV       = "data/cluster_labels_meso.csv"
MICRO_LABELS_CSV      = "data/cluster_labels_micro.csv"
CLUSTER_LABELS_CSV    = "data/cluster_labels.csv"

# --- Model ---
MODEL                 = "claude-sonnet-4-6"
MAX_TOKENS            = 800
TEMPERATURE           = 0.1

# --- Examples ---
LABEL_N_EXAMPLES_FRACTION = 0.2
LABEL_N_EXAMPLES_MIN      = 10
LABEL_N_EXAMPLES_MAX      = 200

# --- Slot fillers ---
MACRO_MESO_TOP_N_FILLERS  = 20
MICRO_TOP_N_FILLERS       = 20

# --- Slots ---
MACRO_MESO_SLOTS = ["subject", "object", "sender", "opponent"]
MICRO_SLOTS      = ["subject", "object", "sender", "receiver", "helper", "opponent"]

# --- Description length ---
MAX_DESCRIPTION_WORDS = 12

# --- Uniqueness context ---
N_RECENT_LABELS_CONTEXT = 20   # number of recent sibling labels passed for uniqueness