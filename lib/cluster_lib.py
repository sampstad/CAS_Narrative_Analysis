"""
All functions for the 04_cluster notebook.
"""

from config import cluster_config as cfg

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import umap
import hdbscan


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_extractions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    if "pubtime" in df.columns:
        df["pubtime"] = pd.to_datetime(df["pubtime"], errors="coerce", utc=True)
    return df


# ---------------------------------------------------------------------------
# 2. Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(extractions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    articles = extractions[["id"]].drop_duplicates("id").reset_index(drop=True)
    feature_dfs = []
    for col in ["category", "specific_type", "domain"]:
        valid = extractions.dropna(subset=[col, "slot"])
        pivoted = (
            valid.assign(feature=valid["slot"] + "__" + valid[col])
            .groupby(["id", "feature"])
            .size()
            .unstack(fill_value=0)
            .clip(upper=1)
        )
        feature_dfs.append(pivoted)
    combined = pd.concat(feature_dfs, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()].fillna(0).astype(np.uint8)
    combined = articles.set_index("id").join(combined, how="left").fillna(0).astype(np.uint8)
    return combined.values, combined.index.values, combined.columns.values

def save_feature_matrix(matrix: np.ndarray, ids: np.ndarray, columns: np.ndarray,
                         matrix_path: str, ids_path: str, columns_path: str):
    np.save(matrix_path, matrix)
    np.save(ids_path, ids)
    np.save(columns_path, columns)

def load_feature_matrix(matrix_path: str, ids_path: str,
                         columns_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.load(matrix_path), np.load(ids_path, allow_pickle=True), np.load(columns_path, allow_pickle=True)

def feature_matrix_exists(matrix_path: str, ids_path: str) -> bool:
    return Path(matrix_path).exists() and Path(ids_path).exists()


# ---------------------------------------------------------------------------
# 3. Dominant narrative embeddings
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

def save_embeddings(embeddings: np.ndarray, ids: np.ndarray, embeddings_path: str, ids_path: str):
    np.save(embeddings_path, embeddings)
    np.save(ids_path, ids)

def load_embeddings(embeddings_path: str, ids_path: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(embeddings_path), np.load(ids_path, allow_pickle=True)

def embeddings_exist(embeddings_path: str, ids_path: str) -> bool:
    return Path(embeddings_path).exists() and Path(ids_path).exists()


# ---------------------------------------------------------------------------
# 4. Slot label embeddings
# ---------------------------------------------------------------------------

# build slot embeddings per article using per-label mean pooling
# each slot gets its own 1024-dim vector (mean of all filler label embeddings for that slot)
# article representation = concatenation of 6 slot vectors → 6144 dims
# slots with no fillers get a zero vector
# this avoids string concatenation dilution and tokenizer truncation
SLOTS = ["subject", "object", "sender", "receiver", "helper", "opponent"]
EMBEDDING_DIM = 1024

def build_slot_embeddings(extractions: pd.DataFrame,
                           model: SentenceTransformer,
                           batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    article_ids = extractions["id"].drop_duplicates().values
    n_articles  = len(article_ids)
    n_slots     = len(SLOTS)

    # collect all unique labels across all slots — embed them all at once
    all_labels = extractions["label"].dropna().unique().tolist()
    print(f"  Embedding {len(all_labels):,} unique slot labels...")
    all_embs = model.encode(
        all_labels,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    label_to_emb = {label: all_embs[i] for i, label in enumerate(all_labels)}

    # build per-article slot embedding matrix: (n_articles, n_slots * EMBEDDING_DIM)
    slot_matrix = np.zeros((n_articles, n_slots * EMBEDDING_DIM), dtype=np.float32)
    id_to_rows  = extractions.groupby("id")

    for art_idx, aid in enumerate(article_ids):
        try:
            art_rows = id_to_rows.get_group(aid)
        except KeyError:
            continue
        for slot_idx, slot in enumerate(SLOTS):
            slot_labels = art_rows[art_rows["slot"] == slot]["label"].dropna().tolist()
            if not slot_labels:
                continue
            slot_vecs = np.array([label_to_emb[l] for l in slot_labels
                                   if l in label_to_emb])
            if len(slot_vecs) == 0:
                continue
            # mean pool all filler embeddings for this slot
            pooled = slot_vecs.mean(axis=0)
            # normalise pooled vector
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled /= norm
            start = slot_idx * EMBEDDING_DIM
            slot_matrix[art_idx, start:start + EMBEDDING_DIM] = pooled

    return slot_matrix, article_ids

def save_slot_embeddings(embeddings: np.ndarray, ids: np.ndarray,
                          embeddings_path: str, ids_path: str):
    np.save(embeddings_path, embeddings)
    np.save(ids_path, ids)

def load_slot_embeddings(embeddings_path: str, ids_path: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(embeddings_path), np.load(ids_path, allow_pickle=True)

def slot_embeddings_exist(embeddings_path: str, ids_path: str) -> bool:
    return Path(embeddings_path).exists() and Path(ids_path).exists()


# ---------------------------------------------------------------------------
# 5. Feature combination
# ---------------------------------------------------------------------------

# align a secondary matrix to the row order of the primary matrix by id
def _align_by_id(primary_ids: np.ndarray, secondary_ids: np.ndarray,
                  secondary_matrix: np.ndarray) -> np.ndarray:
    id_to_idx = {str(sid): i for i, sid in enumerate(secondary_ids)}
    aligned   = np.zeros((len(primary_ids), secondary_matrix.shape[1]), dtype=np.float32)
    for i, pid in enumerate(primary_ids):
        idx = id_to_idx.get(str(pid))
        if idx is not None:
            aligned[i] = secondary_matrix[idx]
    return aligned

# combine structural features, narrative embeddings, and slot embeddings
def combine_features(matrix: np.ndarray,
                      embeddings: np.ndarray,
                      matrix_ids: np.ndarray,
                      embedding_ids: np.ndarray,
                      semantic_weight: float,
                      slot_embeddings: np.ndarray = None,
                      slot_embedding_ids: np.ndarray = None,
                      slot_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    struct_norm    = normalize(matrix.astype(np.float32), norm="l2")
    aligned_narr   = _align_by_id(matrix_ids, embedding_ids, embeddings)
    components     = [struct_norm, semantic_weight * aligned_narr]

    if slot_embeddings is not None and slot_weight > 0.0:
        aligned_slot = _align_by_id(matrix_ids, slot_embedding_ids, slot_embeddings)
        components.append(slot_weight * aligned_slot)

    combined = np.hstack(components)
    return combined, matrix_ids

def save_combined(combined: np.ndarray, path: str):
    np.save(path, combined)

def load_combined(path: str) -> np.ndarray:
    return np.load(path)

def combined_exists(path: str) -> bool:
    return Path(path).exists()


# ---------------------------------------------------------------------------
# 6. Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_for_clustering(matrix: np.ndarray, n_components: int, n_neighbors: int,
                           min_dist: float, metric: str) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    return reducer.fit_transform(matrix)

def reduce_for_viz(reduced: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=42,
    )
    return reducer.fit_transform(reduced)

def save_reduced(reduced: np.ndarray, path: str):
    np.save(path, reduced)

def load_reduced(path: str) -> np.ndarray:
    return np.load(path)

def reduced_exists(path: str) -> bool:
    return Path(path).exists()


# ---------------------------------------------------------------------------
# 7. Clustering
# ---------------------------------------------------------------------------

def cluster(reduced: np.ndarray, min_cluster_size: int, min_samples: int,
            metric: str) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        prediction_data=True,
    )
    return clusterer.fit_predict(reduced)

def proportional_min_cluster_size(n_parent: int, fraction: float,
                                   min_val: int, max_val: int) -> int:
    return int(np.clip(n_parent * fraction, min_val, max_val))

# run UMAP and HDBSCAN on a subset of articles belonging to one parent cluster
def cluster_within(matrix: np.ndarray,
                    embeddings: np.ndarray,
                    ids: np.ndarray,
                    embedding_ids: np.ndarray,
                    parent_ids: np.ndarray,
                    semantic_weight: float,
                    n_neighbors: int,
                    n_components: int,
                    min_cluster_size: int,
                    min_samples: int,
                    slot_embeddings: np.ndarray = None,
                    slot_embedding_ids: np.ndarray = None,
                    slot_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    id_to_idx     = {str(mid): i for i, mid in enumerate(ids)}
    subset_idx    = np.array([id_to_idx[str(pid)] for pid in parent_ids if str(pid) in id_to_idx])
    subset_ids    = ids[subset_idx]
    subset_matrix = matrix[subset_idx]

    combined, _ = combine_features(
        subset_matrix, embeddings, subset_ids, embedding_ids,
        semantic_weight,
        slot_embeddings=slot_embeddings,
        slot_embedding_ids=slot_embedding_ids,
        slot_weight=slot_weight,
    )

    n_neighbors_capped  = min(n_neighbors, len(subset_ids) - 1)
    n_components_capped = min(n_components, len(subset_ids) - 2)
    reduced = reduce_for_clustering(combined, n_components_capped, n_neighbors_capped,
                                     0.0, "cosine")
    labels  = cluster(reduced, min_cluster_size, min_samples, "euclidean")
    return labels, subset_ids

def sensitivity_n_neighbors(matrix: np.ndarray, n_neighbors_values: list[int],
                              min_dist: float, metric: str, n_components: int,
                              min_cluster_size: int, min_samples: int) -> pd.DataFrame:
    rows, labels_list = [], []
    for n in n_neighbors_values:
        reduced    = reduce_for_clustering(matrix, n_components, n, min_dist, metric)
        labels     = cluster(reduced, min_cluster_size, min_samples, "euclidean")
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_rate = (labels == -1).mean()
        labels_list.append(labels)
        rows.append({"n_neighbors": n, "n_clusters": n_clusters, "noise_rate": round(noise_rate, 3)})
    df = pd.DataFrame(rows)
    for i, n_a in enumerate(n_neighbors_values):
        for j, n_b in enumerate(n_neighbors_values):
            if j > i:
                ari = adjusted_rand_score(labels_list[i], labels_list[j])
                df.loc[df["n_neighbors"] == n_a, f"ARI_vs_{n_b}"] = round(ari, 3)
    return df

def sensitivity_min_cluster_size(reduced: np.ndarray, values: list[int],
                                   min_samples: int, metric: str) -> pd.DataFrame:
    rows, labels_list = [], []
    for v in values:
        labels     = cluster(reduced, v, min_samples, metric)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_rate = (labels == -1).mean()
        labels_list.append(labels)
        rows.append({"min_cluster_size": v, "n_clusters": n_clusters, "noise_rate": round(noise_rate, 3)})
    df = pd.DataFrame(rows)
    for i, v_a in enumerate(values):
        for j, v_b in enumerate(values):
            if j > i:
                ari = adjusted_rand_score(labels_list[i], labels_list[j])
                df.loc[df["min_cluster_size"] == v_a, f"ARI_vs_{v_b}"] = round(ari, 3)
    return df

def sensitivity_min_samples(reduced: np.ndarray, min_cluster_size: int,
                              values: list[int], metric: str) -> pd.DataFrame:
    rows, labels_list = [], []
    for v in values:
        labels     = cluster(reduced, min_cluster_size, v, metric)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_rate = (labels == -1).mean()
        labels_list.append(labels)
        rows.append({"min_samples": v, "n_clusters": n_clusters, "noise_rate": round(noise_rate, 3)})
    df = pd.DataFrame(rows)
    for i, v_a in enumerate(values):
        for j, v_b in enumerate(values):
            if j > i:
                ari = adjusted_rand_score(labels_list[i], labels_list[j])
                df.loc[df["min_samples"] == v_a, f"ARI_vs_{v_b}"] = round(ari, 3)
    return df


# ---------------------------------------------------------------------------
# 8. Hierarchical assignments
# ---------------------------------------------------------------------------

def collect_cluster_records(ids: np.ndarray, labels: np.ndarray,
                              level: str) -> list[dict]:
    return [
        {"id": str(aid), f"{level}_cluster": int(label) if not np.isnan(label) else np.nan}
        for aid, label in zip(ids, labels)
    ]


# ---------------------------------------------------------------------------
# 9. Cluster profiling
# ---------------------------------------------------------------------------

def join_extractions(assignments: pd.DataFrame, extractions: pd.DataFrame) -> pd.DataFrame:
    return extractions.merge(assignments, on="id", how="left")

def cluster_distribution(joined: pd.DataFrame, cluster_col: str, column: str) -> pd.DataFrame:
    valid  = joined[joined[cluster_col].notna()]
    counts = (
        valid.groupby([cluster_col, column])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(cluster_col)["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    return counts.sort_values([cluster_col, "share"], ascending=[True, False])

def cluster_slot_distribution(joined: pd.DataFrame, cluster_col: str,
                               column: str) -> pd.DataFrame:
    valid  = joined[joined[cluster_col].notna()]
    counts = (
        valid.groupby([cluster_col, "slot", column])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby([cluster_col, "slot"])["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    return counts.sort_values([cluster_col, "slot", "share"], ascending=[True, True, False])

def cluster_examples(assignments: pd.DataFrame, extractions: pd.DataFrame,
                      cluster_col: str, n: int) -> pd.DataFrame:
    narratives = extractions[["id", "dominant_narrative"]].drop_duplicates("id")
    merged     = assignments[assignments[cluster_col].notna()].merge(narratives, on="id", how="left")
    samples    = []
    for cid, group in merged.groupby(cluster_col):
        samples.append(group.sample(min(n, len(group)), random_state=42))
    if not samples:
        return pd.DataFrame(columns=[cluster_col, "id", "dominant_narrative"])
    return pd.concat(samples)[[cluster_col, "id", "dominant_narrative"]].reset_index(drop=True)

def cluster_profiles(joined: pd.DataFrame, assignments: pd.DataFrame,
                      cluster_col: str) -> pd.DataFrame:
    rows  = []
    total = assignments[cluster_col].notna().sum()
    for cluster_id in sorted(joined[cluster_col].dropna().unique()):
        sub = joined[joined[cluster_col] == cluster_id]
        n   = sub["id"].nunique()
        top_category = sub["category"].value_counts().head(3).index.tolist() if "category" in sub.columns else []
        top_specific = sub["specific_type"].value_counts().head(3).index.tolist() if "specific_type" in sub.columns else []
        top_domain   = sub["domain"].value_counts().head(3).index.tolist() if "domain" in sub.columns else []
        rows.append({
            "cluster":        int(cluster_id),
            "n_articles":     n,
            "share_of_total": round(n / total, 3) if total > 0 else 0,
            "top_categories": top_category,
            "top_specifics":  top_specific,
            "top_domains":    top_domain,
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 10. Parameter tuning and validation
# ---------------------------------------------------------------------------

import itertools

def _coherence_for_clustering(labels: np.ndarray, subset_ids: np.ndarray,
                               extractions: pd.DataFrame,
                               catchall_threshold: float = 0.30) -> tuple[float, float]:
    """
    Compute coherence metrics for a clustering result.
    Returns (avg_top10_cumul, catchall_rate) across all non-noise clusters.
    top10_cumul = fraction of articles covered by top-10 subject entities.
    catchall_rate = fraction of clusters where top10_cumul < catchall_threshold.
    """
    scores = []
    for cid in set(labels[labels >= 0]):
        cids = subset_ids[labels == cid].astype(str)
        sub  = extractions[extractions["id"].isin(cids)]
        subj = sub[sub["slot"] == "subject"].dropna(subset=["label"])
        if subj.empty:
            continue
        n_arts = len(cids)
        top10  = (
            subj.groupby("label")["id"].nunique()
            .sort_values(ascending=False)
            .head(10)
            .sum() / n_arts
        )
        scores.append(float(top10))
    if not scores:
        return 0.0, 1.0
    avg_top10    = float(np.mean(scores))
    catchall_rt  = float(np.mean([s < catchall_threshold for s in scores]))
    return avg_top10, catchall_rt


def tune_macro(
    combined: np.ndarray,
    n_neighbors_values: list,
    min_cluster_sizes: list,
    min_samples_values: list,
    umap_n_components: int,
    umap_min_dist: float,
    umap_metric: str,
    hdbscan_metric: str = "euclidean",
    target_n_min: int = 10,
    target_n_max: int = 25,
    target_noise_max: float = 0.15,
) -> pd.DataFrame:
    """
    Grid search for macro UMAP + HDBSCAN parameters.
    Takes the combined matrix (not pre-reduced) so n_neighbors can be varied.
    UMAP runs once per n_neighbors value; HDBSCAN params vary on the cached result.

    NOTE: Coherence (top10_cumul) is NOT used here. Macro clusters are intentionally
    broad thematic umbrellas where no single actor dominates — every macro cluster
    will look like a catch-all by that metric. Instead, scoring is based purely on:
      - whether n_clusters falls in the target range
      - noise rate staying below the target ceiling

    Returns ranked DataFrame with RANK column; top row is recommended config.
    """
    results = []
    for n_nb in n_neighbors_values:
        print(f"  UMAP n_neighbors={n_nb}...")
        reduced = reduce_for_clustering(combined, umap_n_components, n_nb, umap_min_dist, umap_metric)
        for mcs, ms in itertools.product(min_cluster_sizes, min_samples_values):
            labels     = cluster(reduced, mcs, ms, hdbscan_metric)
            n_cls      = len(set(labels[labels >= 0]))
            noise_rate = float((labels == -1).mean())
            results.append({
                "n_neighbors":      n_nb,
                "min_cluster_size": mcs,
                "min_samples":      ms,
                "n_clusters":       n_cls,
                "noise_rate":       round(noise_rate, 3),
            })

    df = pd.DataFrame(results)
    df["in_target"] = df["n_clusters"].between(target_n_min, target_n_max).astype(float)
    df["noise_pen"] = (df["noise_rate"] > target_noise_max).astype(float)
    df["score"]     = (df["in_target"] - df["noise_pen"] - df["noise_rate"]).round(3)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "RANK", range(1, len(df) + 1))
    return df


def tune_within(
    matrix: np.ndarray,
    embeddings: np.ndarray,
    ids: np.ndarray,
    embedding_ids: np.ndarray,
    slot_embeddings: np.ndarray,
    slot_embedding_ids: np.ndarray,
    extractions: pd.DataFrame,
    parent_assignments: pd.DataFrame,
    parent_col: str,
    semantic_weights: list,
    slot_weights: list,
    n_neighbors_fractions: list,
    mcs_fractions: list,
    mcs_min: int,
    mcs_max: int,
    min_samples_values: list,
    n_neighbors_min: int,
    n_neighbors_max: int,
    umap_n_components: int = 50,
    umap_min_dist: float = 0.0,
    umap_metric: str = "cosine",
    hdbscan_metric: str = "euclidean",
    n_parent_sample: int = 3,
    test_parent_ids: list = None,
    target_noise_max: float = 0.25,
    catchall_threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Grid search for within-cluster (meso or micro) parameters.
    By default stratifies parent cluster sample by size (smallest, median, largest).
    Pass test_parent_ids to override with specific parent clusters — useful when the
    default sample is too optimistic and you want to tune on known problematic clusters.
    UMAP runs once per (semantic_weight, slot_weight, n_neighbors_fraction, parent_cluster);
    HDBSCAN params (mcs_fraction, min_samples) are varied on the cached reduced space.
    Scores on: coherence (avg_top10_cumul), catch-all rate, noise rate.
    Returns ranked DataFrame with RANK column; top row is recommended config.
    """
    # Parent cluster sample — use override if provided, else stratify by size
    if test_parent_ids is not None:
        test_pids = list(test_parent_ids)
    else:
        sizes = parent_assignments.groupby(parent_col)["id"].nunique().sort_values()
        n     = len(sizes)
        sample_indices = sorted(set([0, n // 2, n - 1]))[:n_parent_sample]
        test_pids      = [sizes.index[i] for i in sample_indices]

    n_umap = len(semantic_weights) * len(slot_weights) * len(n_neighbors_fractions)
    n_hdb  = len(mcs_fractions) * len(min_samples_values)
    print(f"  Tuning on {len(test_pids)} parent clusters: {[int(p) for p in test_pids]}")
    print(f"  UMAP runs: {n_umap} weight combos × {len(test_pids)} clusters = {n_umap * len(test_pids)}")
    print(f"  HDBSCAN combos per UMAP: {n_hdb}  |  Total combinations: {n_umap * n_hdb}")

    id_to_idx = {str(mid): i for i, mid in enumerate(ids)}
    results   = []

    for sw, slw, nb_frac in itertools.product(semantic_weights, slot_weights, n_neighbors_fractions):
        # Cache UMAP reductions for this weight + n_neighbors combination
        reduced_cache    = {}
        subset_ids_cache = {}

        for pid in test_pids:
            parent_ids  = parent_assignments[parent_assignments[parent_col] == pid]["id"].values
            subset_idx  = np.array([id_to_idx[str(p)] for p in parent_ids if str(p) in id_to_idx])
            subset_mat  = matrix[subset_idx]
            subset_id   = ids[subset_idx]
            n_neighbors = proportional_min_cluster_size(
                len(parent_ids), nb_frac, n_neighbors_min, n_neighbors_max
            )
            n_comp = min(umap_n_components, len(subset_idx) - 2)

            combined, out_ids = combine_features(
                subset_mat, embeddings, subset_id, embedding_ids, sw,
                slot_embeddings=slot_embeddings,
                slot_embedding_ids=slot_embedding_ids,
                slot_weight=slw,
            )
            reduced = reduce_for_clustering(combined, n_comp, n_neighbors, umap_min_dist, umap_metric)
            reduced_cache[pid]    = reduced
            subset_ids_cache[pid] = out_ids

        print(f"  sw={sw} slw={slw} nb_frac={nb_frac} — UMAP done, running HDBSCAN grid...")

        # Vary HDBSCAN params on cached reduced spaces
        for frac, ms in itertools.product(mcs_fractions, min_samples_values):
            agg_n, agg_noise, agg_coh, agg_ca = [], [], [], []

            for pid in test_pids:
                reduced  = reduced_cache[pid]
                out_ids  = subset_ids_cache[pid]
                mcs      = proportional_min_cluster_size(len(out_ids), frac, mcs_min, mcs_max)
                labels   = cluster(reduced, mcs, ms, hdbscan_metric)
                n_cls    = len(set(labels[labels >= 0]))
                noise    = float((labels == -1).mean())
                avg_coh, ca = _coherence_for_clustering(labels, out_ids, extractions, catchall_threshold)

                agg_n.append(n_cls)
                agg_noise.append(noise)
                agg_coh.append(avg_coh)
                agg_ca.append(ca)

            results.append({
                "semantic_weight":    sw,
                "slot_weight":        slw,
                "n_neighbors_frac":   nb_frac,
                "mcs_fraction":       frac,
                "min_samples":        ms,
                "avg_n_clusters":     round(float(np.mean(agg_n)), 1),
                "avg_noise_rate":     round(float(np.mean(agg_noise)), 3),
                "avg_top10_cumul":    round(float(np.mean(agg_coh)), 3),
                "catchall_rate":      round(float(np.mean(agg_ca)), 3),
            })

    df = pd.DataFrame(results)
    df["noise_penalty"] = (df["avg_noise_rate"] > target_noise_max).astype(float)
    df["score"] = (
        df["avg_top10_cumul"]
        - df["catchall_rate"] * 0.5
        - df["noise_penalty"] * 0.5
    ).round(3)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "RANK", range(1, len(df) + 1))
    return df


def validate_clustering(
    assignments: pd.DataFrame,
    cluster_col: str,
    extractions: pd.DataFrame,
    catchall_threshold: float = 0.30,
    min_size_warning: int = 25,
    target_noise_max: float = 0.25,
    target_coherence_min: float = 0.30,
    target_catchall_max: float = 0.15,
    check_coherence: bool = True,
) -> pd.DataFrame:
    """
    Coherence validation for a completed clustering result.
    Flags catch-all clusters (top10_cumul below threshold) and tiny clusters.
    Prints a pass/fail verdict with summary stats.
    Set check_coherence=False for macro level — macro clusters are intentionally
    actor-diverse and will always fail coherence thresholds by design.
    Returns DataFrame of all clusters with coherence scores, sorted by top10_cumul.
    """
    results     = []
    cluster_ids = assignments[cluster_col].dropna().unique()
    n_total     = assignments["id"].nunique()

    for cid in sorted(cluster_ids):
        art_ids = assignments[assignments[cluster_col] == cid]["id"].values
        n_arts  = len(art_ids)
        sub     = extractions[extractions["id"].isin(art_ids.astype(str))]
        subj    = sub[sub["slot"] == "subject"].dropna(subset=["label"])

        if subj.empty:
            top10, top1, n_uniq = 0.0, 0.0, 0
        else:
            top_counts = (
                subj.groupby("label")["id"].nunique()
                .sort_values(ascending=False)
            )
            n_uniq = len(top_counts)
            top1   = float(top_counts.iloc[0] / n_arts)
            top10  = float(top_counts.head(10).sum() / n_arts)

        flags = []
        if top10 < catchall_threshold:
            flags.append("CATCH-ALL")
        if n_arts < min_size_warning:
            flags.append("TINY")

        results.append({
            "cluster":     int(cid),
            "n_articles":  n_arts,
            "n_uniq_subj": n_uniq,
            "top1_pct":    round(top1, 2),
            "top10_cumul": round(top10, 2),
            "flags":       ", ".join(flags) if flags else "OK",
        })

    df          = pd.DataFrame(results).sort_values("top10_cumul")
    n_clusters  = len(df)
    n_catchall  = df["flags"].str.contains("CATCH-ALL").sum()
    n_tiny      = df["flags"].str.contains("TINY").sum()
    avg_coh     = df["top10_cumul"].mean()
    noise_ids   = assignments[assignments[cluster_col].isna()]["id"].nunique()
    noise_rate  = noise_ids / n_total
    catchall_rt = n_catchall / n_clusters if n_clusters > 0 else 0

    print(f"\n{'='*60}")
    print(f"VALIDATION — {cluster_col.upper()}")
    print(f"{'='*60}")
    print(f"  Clusters:        {n_clusters}")
    print(f"  Noise rate:      {noise_rate:.1%}  (target < {target_noise_max:.0%})")
    print(f"  Avg coherence:   {avg_coh:.1%}  (top10_cumul, target > {target_coherence_min:.0%})")
    print(f"  Catch-all rate:  {catchall_rt:.0%}  (target < {target_catchall_max:.0%})")
    print(f"  Tiny clusters:   {n_tiny}  (< {min_size_warning} articles)")

    passed = (noise_rate < target_noise_max) and (
        not check_coherence or (
            avg_coh     > target_coherence_min and
            catchall_rt < target_catchall_max
        )
    )
    print(f"\n  VERDICT: {'✓ PASS' if passed else '✗ FAIL — adjust parameters and re-run'}")
    if not check_coherence:
        print(f"  (coherence not evaluated — macro clusters are intentionally actor-diverse)")

    if n_catchall > 0 or n_tiny > 0:
        print(f"\n  Flagged clusters (worst first):")
        flagged = df[df["flags"] != "OK"][["cluster", "n_articles", "top10_cumul", "flags"]]
        print(flagged.to_string(index=False))

    return df


def profile_noise(joined: pd.DataFrame, cluster_col: str, n_examples: int = 10) -> pd.DataFrame:
    noise = joined[joined[cluster_col].isna()]
    print(f"Noise articles at {cluster_col} level: {noise['id'].nunique():,} "
          f"({noise['id'].nunique() / joined['id'].nunique():.1%})")
    return noise[["id", "dominant_narrative"]].drop_duplicates("id").sample(
        min(n_examples, len(noise)), random_state=42)