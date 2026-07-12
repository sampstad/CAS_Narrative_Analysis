"""Configuration for the visualisation precompute pipeline and Streamlit app."""

# --- Inputs (produced by earlier pipeline stages) ---
EXTRACTIONS_CSV       = "data/narrative_extractions_normalized.csv"
PREPROCESSED_TSV      = "data/preprocessed_dataset.tsv"

MACRO_ASSIGNMENTS_CSV = "data/cluster_assignments_macro.csv"
MESO_ASSIGNMENTS_CSV  = "data/cluster_assignments_meso.csv"
MICRO_ASSIGNMENTS_CSV = "data/cluster_assignments_micro.csv"

MACRO_LABELS_CSV      = "data/cluster_labels_macro.csv"
MESO_LABELS_CSV       = "data/cluster_labels_meso.csv"
MICRO_LABELS_CSV      = "data/cluster_labels_micro.csv"

# Dominant-narrative embeddings (for representative example selection)
EMBEDDINGS_NPY        = "data/embeddings.npy"
EMBEDDING_IDS         = "data/embedding_ids.npy"

# --- Outputs (four tidy tables the app reads) ---
VIZ_DIR               = "data/viz"
ARTICLE_TABLE         = "data/viz/article_table.parquet"
CLUSTER_META          = "data/viz/cluster_meta.parquet"
CLUSTER_PUBLISHER     = "data/viz/cluster_publisher.parquet"
CLUSTER_WEEK          = "data/viz/cluster_week.parquet"

# Slim extraction-derived tables that let the app aggregate slot-type groups and
# resolve entity filler types WITHOUT loading the full raw extractions CSV.
SLOT_FILLERS          = "data/viz/slot_fillers.parquet"   # per-filler rows for the 6 card slots
ENTITY_FTYPE          = "data/viz/entity_ftype.parquet"   # label -> dominant filler_type

# --- Metric columns carried through the pipeline ---
# Raw per-article emotion / sentiment columns in the preprocessed dataset
EMOTION_COLS   = ["emotion_anger", "emotion_fear", "emotion_joy", "emotion_sadness"]
SENTIMENT_COL  = "sentiment_score"
FLESCH_COL     = "linguistic_complexity"   # raw Flesch reading ease (higher = simpler)

# Averaged metrics available in heatmap / rankings (post-transform names)
AVG_METRICS    = ["complexity", "sentiment", "anger", "fear", "joy", "sadness"]

# --- Levels ---
LEVELS = ["macro", "meso", "micro"]

# --- Representative examples ---
N_REPRESENTATIVE_EXAMPLES = 10   # per cluster, nearest to centroid in narrative embedding space

# --- Time bucketing ---
TRIM_PARTIAL_WEEKS = True       # drop leading/trailing partial ISO weeks

# --- Small-sample floors ---
# averaged-metric cells / rankings below this article count are unreliable
CELL_MIN_ARTICLES  = 30         # publisher x cluster cell floor (averaged metrics only)
RANKING_MIN_ARTICLES = {         # eligibility floor for extreme rankings, by level
    "macro": 0,
    "meso":  30,
    "micro": 30,
}
# diffusion view: micro narratives below this are hidden from the selector
DIFFUSION_MIN_ARTICLES_MICRO = 50

# --- Slots shown on the narrative card ---
CARD_SLOTS = ["subject", "object", "sender", "receiver", "helper", "opponent"]
TOP_ENTITIES_PER_SLOT   = 5   # entity labels shown per type group
TOP_TYPE_GROUPS_PER_SLOT = 4  # distinct (filler_type, category, specific_type) groups per slot

# --- Search ---
import os as _os
# Embedding model for SEARCH (independent of the e5-large model used for clustering).
# e5-small is light enough for free-tier hosting. Override with env var if needed.
# IMPORTANT: all search artifacts (entity_embeddings, narrative embeddings for
# centroids) must be built with this model; query vectors must match.
EMBEDDING_MODEL   = _os.environ.get("SEARCH_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_BATCH   = 64

# e5 asymmetric prefixes. Used for THEMATIC search only (a fresh, properly-built
# space). Entity search stays prefix-free to match the cluster-pipeline convention.
E5_QUERY_PREFIX   = "query: "
E5_PASSAGE_PREFIX = "passage: "
# Search outputs
ENTITY_INDEX      = "data/viz/entity_index.parquet"       # entity x level x cluster x slot counts
ENTITY_VOCAB      = "data/viz/entity_vocab.parquet"       # distinct entities + aggregate stats
ENTITY_EMBEDDINGS = "data/viz/entity_embeddings.npy"      # one row per entity_vocab row
ENTITY_ARTICLE_BRIDGE = "data/viz/entity_article_bridge.parquet"  # (label, id) for entity scoping
CLUSTER_CENTROIDS = "data/viz/cluster_centroids.npy"      # thematic search: centroid per cluster
CLUSTER_CENTROID_KEYS = "data/viz/cluster_centroid_keys.parquet"  # level+cluster per centroid row
SEARCH_META       = "data/viz/search_meta.json"           # records model used to build embeddings
# Search behaviour
# e5-small is coarser than e5-large, so a lower default threshold catches the
# same matches. Env-configurable for tuning against real results.
SEARCH_ENTITY_SIM_THRESHOLD = float(_os.environ.get("SEARCH_ENTITY_THRESHOLD", "0.90"))
SEARCH_MAX_ENTITIES         = int(_os.environ.get("SEARCH_MAX_ENTITIES", "100"))
SEARCH_SLOTS = ["subject", "object", "sender", "receiver", "helper", "opponent"]

# Thematic narrative embeddings (e5-small, passage-prefixed) for centroid search
NARRATIVE_EMB_SMALL = "data/viz/narrative_emb_small.npy"
NARRATIVE_EMB_IDS   = "data/viz/narrative_emb_ids.npy"

# --- 2D semantic map (UMAP of e5-large narrative embeddings) ---
MAP_XY              = "data/viz/map_xy.parquet"          # id -> x_2d, y_2d (article cloud)
MAP_CENTROIDS       = "data/viz/map_centroids.parquet"   # level, cluster, x, y, n_articles
MAP_N_NEIGHBORS     = 30      # UMAP neighbours for the 2D layout
MAP_MIN_DIST        = 0.1     # UMAP min_dist for the 2D layout
MAP_CLOUD_SAMPLE    = 15000   # max article points drawn in the cloud (subsample if larger)