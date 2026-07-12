"""
Central configuration for the 05_cluster notebook.
"""

# --- Paths ---
EXTRACTIONS_CSV       = "data/narrative_extractions_normalized.csv"
ASSIGNMENTS_CSV       = "data/cluster_assignments.csv"
MACRO_ASSIGNMENTS_CSV = "data/cluster_assignments_macro.csv"
MESO_ASSIGNMENTS_CSV  = "data/cluster_assignments_meso.csv"
MICRO_ASSIGNMENTS_CSV = "data/cluster_assignments_micro.csv"
PROFILES_CSV          = "data/cluster_profiles.csv"
FEATURE_MATRIX_NPY    = "data/feature_matrix.npy"
FEATURE_IDS           = "data/feature_ids.npy"
FEATURE_COLUMNS_NPY   = "data/feature_columns.npy"
EMBEDDINGS_NPY        = "data/embeddings.npy"
EMBEDDING_IDS         = "data/embedding_ids.npy"
SLOT_EMBEDDINGS_NPY   = "data/slot_embeddings.npy"
SLOT_EMBEDDING_IDS    = "data/slot_embedding_ids.npy"
COMBINED_MATRIX_NPY   = "data/combined_matrix.npy"
REDUCED_NPY           = "data/reduced_embeddings.npy"

# --- Embedding ---
EMBEDDING_MODEL       = "intfloat/multilingual-e5-large"
EMBEDDING_BATCH_SIZE  = 64

# --- Semantic weights (dominant narrative embeddings) ---
MACRO_SEMANTIC_WEIGHT = 0.1
MESO_SEMANTIC_WEIGHT  = 0.7

# --- Slot label embedding weights ---
MACRO_SLOT_WEIGHT     = 0.0    # not used at macro level
MESO_SLOT_WEIGHT      = 0.0    # not used at meso level
MICRO_SLOT_WEIGHT     = 0.1    # primary differentiator at micro level

# --- UMAP (shared settings) ---
UMAP_N_COMPONENTS     = 50
UMAP_MIN_DIST         = 0.0
UMAP_METRIC           = "cosine"

# --- UMAP (visualisation) ---
VIZ_N_COMPONENTS      = 2
VIZ_N_NEIGHBORS       = 30
VIZ_MIN_DIST          = 0.0

# --- HDBSCAN shared ---
HDBSCAN_METRIC        = "euclidean"

# --- Level 1: Macro-narratives (~10-20, structural) ---
MACRO_N_NEIGHBORS        = 200
MACRO_MIN_CLUSTER_SIZE   = 150
MACRO_MIN_SAMPLES        = 50

# --- Level 2: Meso-narratives (semantic content sub-types within macro) ---
MESO_N_NEIGHBORS_FRACTION       = 0.003
MESO_N_NEIGHBORS_MIN            = 10
MESO_N_NEIGHBORS_MAX            = 30
MESO_MIN_CLUSTER_SIZE_FRACTION  = 0.008
MESO_MIN_CLUSTER_SIZE_MIN       = 15
MESO_MIN_CLUSTER_SIZE_MAX       = 75
MESO_MIN_SAMPLES                = 5

# --- Level 3: Micro-narratives (structural perspective within meso topic) ---
MICRO_SEMANTIC_WEIGHT            = 0.05
MICRO_N_NEIGHBORS_FRACTION       = 0.05
MICRO_N_NEIGHBORS_MIN            = 5
MICRO_N_NEIGHBORS_MAX            = 15
MICRO_MIN_CLUSTER_SIZE_FRACTION  = 0.05
MICRO_MIN_CLUSTER_SIZE_MIN       = 20
MICRO_MIN_CLUSTER_SIZE_MAX       = 100
MICRO_MIN_SAMPLES                = 3

# --- Sensitivity analysis ---
SENSITIVITY_MACRO_N_NEIGHBORS      = [100, 200, 300]
SENSITIVITY_MACRO_MIN_CLUSTER_SIZE = [150, 200, 300]
SENSITIVITY_MACRO_MIN_SAMPLES      = [25, 50, 75]

SENSITIVITY_MESO_N_NEIGHBORS       = [15, 30, 50]
SENSITIVITY_MESO_MIN_CLUSTER_SIZE  = [10, 20, 30, 50]
SENSITIVITY_MESO_MIN_SAMPLES       = [3, 5, 10]

SENSITIVITY_MICRO_N_NEIGHBORS      = [5, 10, 15]
SENSITIVITY_MICRO_MIN_CLUSTER_SIZE = [5, 10, 15]
SENSITIVITY_MICRO_MIN_SAMPLES      = [2, 3, 5]

# --- Profiling ---
N_EXAMPLES_PER_CLUSTER = 5
# --- Tuning grids (used by tune_macro and tune_within) ---
TUNE_MACRO_N_NEIGHBORS        = [100, 150, 200, 300]
TUNE_MACRO_MIN_CLUSTER_SIZES  = [100, 150, 200, 300]
TUNE_MACRO_MIN_SAMPLES        = [25, 50, 75]
TUNE_MACRO_TARGET_N_MIN       = 10    # minimum acceptable macro cluster count
TUNE_MACRO_TARGET_N_MAX       = 20    # maximum acceptable macro cluster count

TUNE_MESO_SEMANTIC_WEIGHTS    = [0.7, 0.9]          # 0.5 confirmed worse — dropped
TUNE_MESO_SLOT_WEIGHTS        = [0.0]               # keep at 0 — known good from grid search
TUNE_MESO_N_NEIGHBORS_FRACS   = [0.003]             # confirmed invariant — fixed at current value
TUNE_MESO_MCS_FRACTIONS       = [0.008, 0.012, 0.015]
TUNE_MESO_MIN_SAMPLES         = [5, 10, 15]
TUNE_MESO_TEST_PARENT_IDS     = [0, 4, 8]           # problematic macro groups from validation

TUNE_MICRO_SEMANTIC_WEIGHTS   = [0.05]              # keep low — known good
TUNE_MICRO_SLOT_WEIGHTS       = [0.1, 0.2, 0.3, 0.4]
TUNE_MICRO_N_NEIGHBORS_FRACS  = [0.03, 0.05, 0.08] # proportional n_neighbors for UMAP
TUNE_MICRO_MCS_FRACTIONS      = [0.03, 0.05, 0.08]
TUNE_MICRO_MIN_SAMPLES        = [3, 7]              # low and high

# --- Validation thresholds ---
COHERENCE_CATCHALL_THRESHOLD  = 0.30   # top10_cumul below this → catch-all
COHERENCE_MIN_SIZE_WARNING    = 25     # clusters below this → tiny
COHERENCE_TARGET_NOISE_MAX    = 0.25   # noise rate ceiling for PASS verdict
COHERENCE_TARGET_COHERENCE    = 0.30   # avg top10_cumul floor for PASS verdict
COHERENCE_TARGET_CATCHALL_MAX = 0.40   # catch-all fraction ceiling for PASS verdict