"""
Central configuration for the 04_normalize notebook.
"""

# --- Paths ---
EXTRACTIONS_CSV             = "data/narrative_extractions.csv"
NORMALIZED_EXTRACTIONS_CSV  = "data/narrative_extractions_normalized.csv"
ENTITY_MAP_CSV              = "data/entity_normalization_map.csv"
ENTITY_EMBEDDINGS_NPY       = "data/entity_label_embeddings.npy"
ENTITY_LABELS_NPY           = "data/entity_label_strings.npy"
PASS1_EMB_NORM_NPY          = "data/pass1_emb_norm.npy"
PASS1_IDS_NPY               = "data/pass1_ids.npy"

# --- Embedding ---
EMBEDDING_MODEL             = "intfloat/multilingual-e5-large"
EMBEDDING_BATCH_SIZE        = 64

# --- Pass 1 clustering (pass 2 dropped) ---
# Threshold 0.04 (similarity > 0.96) excludes opposite-framing pairs like
# "Passage of X" / "Rejection of X" (distance ~0.04-0.05) while retaining
# genuine string variants. LLM threshold lowered to 0.94 to catch the
# 0.04-0.05 range that pass 1 no longer handles.
CLUSTER_THRESHOLD_PASS1     = 0.04   # pass 1 — near-identical string variants only
CLUSTER_KNN_K               = 10     # number of nearest neighbours for pass 1 graph
CLUSTER_MAX_COMPONENT_SIZE  = 500    # components above this are broken into singletons

# --- Diagnostic ---
CLUSTER_DISTANCE_THRESHOLD  = 0.12   # single-pass threshold (used in diagnostic only)
CLUSTER_LINKAGE             = "average"
DIAGNOSTIC_THRESHOLDS       = [0.05, 0.08, 0.12, 0.16, 0.20]
DIAGNOSTIC_SAMPLE_SIZE      = 200    # number of frequent labels to inspect
DIAGNOSTIC_MIN_FREQUENCY    = 10     # min frequency for diagnostic sample
DIAGNOSTIC_SHOW_TOP_N       = 5      # show top N merge groups per threshold

# --- LLM post-processing ---
# LLM_POST_TOP_N is obsolete — replaced by per-type grouping with MIN_FREQUENCY.
# Candidate pairs are now found within each specific_type group separately,
# which eliminates cross-type false positives structurally and gives better
# recall for infrequent entities within smaller type groups.
LLM_POST_SIMILARITY         = 0.94   # cosine similarity threshold for candidate pairs
LLM_POST_BATCH_SIZE         = 150    # pairs per LLM API call
LLM_POST_MIN_FREQUENCY      = 2      # minimum label frequency per type group;
                                     # None = no cutoff (rely on similarity threshold)
                                     # set to 2 or 3 if any type group exceeds ~20k labels
LLM_POST_MAX_ROUNDS         = 5      # maximum iterative merge rounds