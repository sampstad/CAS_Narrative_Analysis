"""
All functions for the 05_normalize notebook.
"""

from config import normalize_config as cfg

import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_extractions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    return df

def get_unique_labels(extractions: pd.DataFrame) -> pd.DataFrame:
    counts = (
        extractions["label"]
        .dropna()
        .value_counts()
        .reset_index()
    )
    counts.columns = ["label", "frequency"]
    return counts

def embeddings_exist(embeddings_path: str, labels_path: str) -> bool:
    return Path(embeddings_path).exists() and Path(labels_path).exists()

def load_embeddings(embeddings_path: str, labels_path: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(embeddings_path), np.load(labels_path, allow_pickle=True)

def save_embeddings(embeddings: np.ndarray, labels: np.ndarray,
                    embeddings_path: str, labels_path: str):
    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)


# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------

def embed_labels(unique_labels: np.ndarray, model_name: str,
                 batch_size: int) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        unique_labels.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


# ---------------------------------------------------------------------------
# 3. Diagnostic
# ---------------------------------------------------------------------------

def diagnostic(embeddings: np.ndarray, labels: np.ndarray,
               label_counts: pd.DataFrame, thresholds: list[float],
               sample_size: int, min_frequency: int, show_top_n: int):
    frequent = label_counts[label_counts["frequency"] >= min_frequency]["label"].values
    if len(frequent) > sample_size:
        rng = np.random.default_rng(42)
        frequent = rng.choice(frequent, size=sample_size, replace=False)

    label_to_idx = {l: i for i, l in enumerate(labels)}
    idxs = np.array([label_to_idx[l] for l in frequent if l in label_to_idx])
    sample_labels = labels[idxs]
    sample_emb    = normalize(embeddings[idxs], norm="l2")

    print(f"Diagnostic on {len(sample_labels)} frequent labels (≥{min_frequency} occurrences)\n")

    for threshold in thresholds:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage=cfg.CLUSTER_LINKAGE,
        )
        cluster_ids = clusterer.fit_predict(sample_emb)
        n_clusters  = len(set(cluster_ids))
        n_merged    = len(sample_labels) - n_clusters

        print(f"── threshold={threshold} → {n_clusters} groups, {n_merged} labels merged ──")

        groups = {}
        for label, cid in zip(sample_labels, cluster_ids):
            groups.setdefault(cid, []).append(label)
        sorted_groups = sorted(groups.values(), key=len, reverse=True)
        shown = 0
        for group in sorted_groups:
            if len(group) > 1:
                freq_info = []
                for l in group:
                    freq = label_counts[label_counts["label"] == l]["frequency"].values
                    freq_info.append(f"{l} ({freq[0] if len(freq) else 0})")
                print(f"  · {' | '.join(freq_info)}")
                shown += 1
                if shown >= show_top_n:
                    break
        print()


# ---------------------------------------------------------------------------
# 4. Pass 1 clustering (faiss-accelerated nearest neighbour graph)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "and", "the", "of", "in", "for", "to", "a", "an", "with", "by",
    "its", "their", "this", "that", "from", "on", "at", "as", "or",
    "de", "der", "die", "das", "und", "des", "den", "dem", "ein",
    "eine", "einer", "eines",
}

def _significant_tokens(label: str) -> frozenset:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]{4,}", label.lower())
    return frozenset(t for t in tokens if t not in _STOPWORDS)

def share_significant_token(label_a: str, label_b: str) -> bool:
    return bool(_significant_tokens(label_a) & _significant_tokens(label_b))

# pass 1: faiss approximate nearest neighbour graph + connected components
def _pass1_connected_components(emb_norm: np.ndarray,
                                 threshold: float,
                                 k: int = 10) -> np.ndarray:
    import faiss
    n, d     = emb_norm.shape
    emb_f32  = emb_norm.astype(np.float32)

    n_cells   = max(1, min(4096, n // 39))
    quantiser = faiss.IndexFlatIP(d)
    index     = faiss.IndexIVFFlat(quantiser, d, n_cells, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = max(1, n_cells // 10)

    t0 = time.time()
    print(f"  Building faiss index ({n:,} vectors, {n_cells} cells, nprobe={index.nprobe})...")
    index.train(emb_f32)
    index.add(emb_f32)
    print(f"  Index built in {time.time() - t0:.1f}s")

    t0 = time.time()
    print(f"  Searching {k} nearest neighbours per label (batched)...")
    batch_size  = 10000
    all_sims    = np.zeros((n, k + 1), dtype=np.float32)
    all_indices = np.zeros((n, k + 1), dtype=np.int64)
    for start in tqdm(range(0, n, batch_size), desc="  Searching", unit="batch"):
        end = min(start + batch_size, n)
        all_sims[start:end], all_indices[start:end] = index.search(emb_f32[start:end], k + 1)
    sims, indices = all_sims, all_indices
    print(f"  Search complete in {time.time() - t0:.1f}s")

    t0 = time.time()
    print(f"  Building edge graph...")
    distances = 1 - sims[:, 1:]
    nbr_ids   = indices[:, 1:]
    mask = (nbr_ids >= 0) & (distances <= threshold)
    rows = np.repeat(np.arange(n), k)[mask.ravel()]
    cols = nbr_ids.ravel()[mask.ravel()]
    print(f"  Edges found: {len(rows):,} (in {time.time() - t0:.1f}s)")

    t0 = time.time()
    print(f"  Computing connected components...")
    data   = np.ones(len(rows), dtype=np.float32)
    graph  = csr_matrix((data, (rows, cols)), shape=(n, n))
    graph  = graph + graph.T
    n_components, component_ids = connected_components(graph, directed=False)
    print(f"  Pass 1 complete: {n_components:,} components (in {time.time() - t0:.1f}s)")
    return component_ids



# ---------------------------------------------------------------------------
# 5. Canonical map
# ---------------------------------------------------------------------------

def build_canonical_map(labels: np.ndarray, cluster_ids: np.ndarray,
                         label_counts: pd.DataFrame) -> pd.DataFrame:
    freq_map = label_counts.set_index("label")["frequency"].to_dict()
    records  = []
    groups   = {}
    for label, cid in zip(labels, cluster_ids):
        groups.setdefault(int(cid), []).append(label)

    for cid, group_labels in groups.items():
        canonical = max(group_labels, key=lambda l: freq_map.get(l, 0))
        for label in group_labels:
            records.append({
                "raw_label":       label,
                "canonical_label": canonical,
                "cluster_id":      cid,
                "n_variants":      len(group_labels),
            })

    return pd.DataFrame(records).sort_values("n_variants", ascending=False)


# ---------------------------------------------------------------------------
# 6. Apply normalization
# ---------------------------------------------------------------------------

def apply_normalization(extractions: pd.DataFrame,
                         canonical_map: pd.DataFrame) -> pd.DataFrame:
    mapping = canonical_map.set_index("raw_label")["canonical_label"].to_dict()
    result  = extractions.copy()
    result["label"] = result["label"].map(lambda x: mapping.get(x, x) if pd.notna(x) else x)
    return result

def save_outputs(normalized: pd.DataFrame, canonical_map: pd.DataFrame):
    normalized.to_csv(cfg.NORMALIZED_EXTRACTIONS_CSV, index=False)
    canonical_map.to_csv(cfg.ENTITY_MAP_CSV, index=False)
    print(f"Saved normalized extractions → {cfg.NORMALIZED_EXTRACTIONS_CSV}")
    print(f"Saved entity map → {cfg.ENTITY_MAP_CSV}")




# ---------------------------------------------------------------------------
# 7. LLM split step (fix pass-1 wrong merges in large canonical groups)
# ---------------------------------------------------------------------------

_SPLIT_SYSTEM_PROMPT = """You are auditing entity normalization results for a Swiss German news corpus.

You will be shown a canonical entity label and a numbered list of raw label variants that have been grouped under it.
Your task: identify any variants that refer to a DIFFERENT real-world entity and should NOT be grouped here.

For each variant, answer KEEP (same entity, just different phrasing) or SPLIT (clearly different entity).

Rules:
- KEEP if it is the same entity with name, title, language, or role variants, abbreviations, or minor descriptors added
- KEEP if it is genuinely ambiguous — err strongly on the side of keeping
- SPLIT only if it is clearly a different person, organisation, place, or concept
- SPLIT if it belongs to the opposite side of a relationship (e.g. property owners inside a tenants association group)
- SPLIT if it is a different institutional body of the same type (e.g. Finance Commission inside Economic Commission group)
- SPLIT if it is a different country's equivalent body (e.g. Austrian police inside a German citizens group)
- SPLIT if it is the municipal government rather than the affected community (e.g. Municipality of X inside Victims of X group)
- SPLIT if it describes the opposite electoral outcome (e.g. Passage of X inside Rejection of X group)
- SPLIT if it refers to a different country's parallel situation (e.g. Venezuela/Maduro variants inside an Iran regime change group; ceasefire between Iran and Israel inside a Gaza ceasefire group)
- SPLIT if it refers to a different conflict or war (e.g. Iran-Israel ceasefire inside a Gaza/Hamas ceasefire group; Ukraine peace plan inside a Gaza peace plan group). Key signal: if the canonical group is clearly about one conflict (Gaza/Hamas, Ukraine/Russia) but a variant mentions a completely different conflict (Iran-Israel war, Gaza war inside a Ukraine group), always SPLIT it out — even if the type of event (ceasefire, regime change) is similar

Output format — one line per variant, nothing else:
variant_id|KEEP
variant_id|SPLIT"""


def find_split_candidates(
        canonical_map: pd.DataFrame,
        label_counts: pd.DataFrame,
        min_variants: int = 20,
        top_n: int = 50,
) -> dict[str, list[str]]:
    """Find large canonical groups that may contain wrongly merged variants.

    Returns a dict mapping canonical_label -> list of all raw variant labels,
    limited to groups with at least min_variants variants, sorted by size
    descending and capped at top_n groups.
    """
    freq_map = label_counts.set_index("label")["frequency"].to_dict()
    merged   = canonical_map[canonical_map["n_variants"] >= min_variants].copy()

    groups = (
        merged.groupby("canonical_label")["raw_label"]
        .apply(list)
        .reset_index()
    )
    groups["n_variants"] = groups["raw_label"].apply(len)
    groups = groups.sort_values("n_variants", ascending=False).head(top_n)

    print(f"  {len(groups)} canonical groups with ≥{min_variants} variants to audit")
    return dict(zip(groups["canonical_label"], groups["raw_label"]))


def llm_split_canonicals(
        split_candidates: dict[str, list[str]],
        label_counts: pd.DataFrame,
        batch_size: int = 50,
) -> dict[str, list[str]]:
    """Ask LLM to identify variants that don't belong in each canonical group.

    For each canonical group, sends batches of its variants and asks which
    ones refer to a meaningfully different entity. Returns a dict mapping
    canonical_label -> list of raw labels that should be split out.
    """
    import anthropic
    client   = anthropic.Anthropic()
    freq_map = label_counts.set_index("label")["frequency"].to_dict()

    splits_by_canonical: dict[str, list[str]] = {}
    total_splits = 0

    for canonical, variants in split_candidates.items():
        canonical_splits = []

        for batch_start in tqdm(
                range(0, len(variants), batch_size),
                desc=f"  [{canonical[:50]}]",
                unit="batch",
                leave=False,
        ):
            batch = variants[batch_start:batch_start + batch_size]
            lines = [f"{i}| {v} ({freq_map.get(v, 0)})"
                     for i, v in enumerate(batch)]
            content = f"Canonical: {canonical}\n\n" + "\n".join(lines)

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                system=_SPLIT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )

            if not response.content:
                print(f"  Warning: empty response for \'{canonical[:60]}\', skipping batch")
                continue

            if response.stop_reason == "max_tokens":
                print(f"  Warning: truncated response for \'{canonical[:60]}\'")

            for line in response.content[0].text.strip().split("\n"):
                line = line.strip()
                if not line or "|" not in line:
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                try:
                    local_idx = int(parts[0].strip())
                    verdict   = parts[1].strip().upper()
                    if verdict == "SPLIT":
                        canonical_splits.append(batch[local_idx])
                except (ValueError, IndexError):
                    continue

        if canonical_splits:
            splits_by_canonical[canonical] = canonical_splits
            total_splits += len(canonical_splits)
            print(f"  [{canonical[:60]}] → {len(canonical_splits):,} splits")
        else:
            print(f"  [{canonical[:60]}] → clean")

    print(f"\n  Total variants to split out: {total_splits:,}")
    return splits_by_canonical


def apply_splits(
        canonical_map: pd.DataFrame,
        splits_by_canonical: dict[str, list[str]],
) -> pd.DataFrame:
    """Split flagged variants out of their canonical group into singletons.

    Each split variant becomes its own canonical label (raw_label == canonical_label).
    """
    result  = canonical_map.copy()
    total   = 0

    for canonical, variants_to_split in splits_by_canonical.items():
        for v in variants_to_split:
            mask = (
                (result["raw_label"]       == v) &
                (result["canonical_label"] == canonical)
            )
            result.loc[mask, "canonical_label"] = v
            total += int(mask.sum())

    result["n_variants"] = result.groupby(
        "canonical_label")["raw_label"].transform("count")

    print(f"  Split out {total:,} variants as singletons")
    print(f"  Unique canonicals after: {result['canonical_label'].nunique():,}")
    return result

# ---------------------------------------------------------------------------
# 8. Sanity check
# ---------------------------------------------------------------------------

def sanity_check(canonical_map: pd.DataFrame, label_counts: pd.DataFrame,
                 n: int = 50):
    freq_map = label_counts.set_index("label")["frequency"].to_dict()
    merged   = canonical_map[canonical_map["n_variants"] > 1].copy()
    top = (
        merged.groupby("canonical_label")
        .agg(n_variants=("raw_label", "count"),
             variants=("raw_label", list))
        .sort_values("n_variants", ascending=False)
        .head(n)
    )
    print(f"Top {n} normalized entities by number of variants:\n")
    for canonical, row in top.iterrows():
        print(f"  → {canonical} ({freq_map.get(canonical, 0):,} occurrences)")
        for v in row["variants"]:
            if v != canonical:
                print(f"      {v} ({freq_map.get(v, 0):,})")
        print()


# ---------------------------------------------------------------------------
# 9. LLM post-processing
# ---------------------------------------------------------------------------

def build_canonical_type_map(extractions: pd.DataFrame,
                              canonical_map: pd.DataFrame) -> dict[str, str]:
    """Build a mapping from canonical label → most frequent specific_type.

    For each canonical label, collects all specific_type values across its raw
    label variants in the extractions, then returns the most common one.
    """
    raw_to_canon = canonical_map.set_index("raw_label")["canonical_label"].to_dict()
    ext = extractions[extractions["label"].notna() &
                      extractions["specific_type"].notna()].copy()
    ext["canonical"] = ext["label"].map(lambda x: raw_to_canon.get(x, x))
    type_map = (
        ext.groupby("canonical")["specific_type"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    return type_map


def find_candidate_pairs_by_type(
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_counts: pd.DataFrame,
        canonical_map: pd.DataFrame,
        canonical_type_map: dict,
        similarity_threshold: float,
        min_frequency: int = 2,
) -> dict[str, list[tuple]]:
    """Find candidate merge pairs grouped by specific_type.

    For each specific_type, builds a pairwise similarity matrix over all
    canonical labels of that type with frequency >= min_frequency, then
    returns candidate pairs above the similarity threshold.

    Returns a dict mapping specific_type → list of (label_a, label_b, similarity)
    tuples, filtered to exclude already-merged pairs.
    """
    from sklearn.preprocessing import normalize as sk_normalize
    from sklearn.metrics.pairwise import cosine_similarity

    raw_to_canon  = canonical_map.set_index("raw_label")["canonical_label"].to_dict()
    freq_map      = label_counts.set_index("label")["frequency"].to_dict()
    label_to_idx  = {l: i for i, l in enumerate(labels)}
    emb_norm      = sk_normalize(embeddings, norm="l2")

    # group canonical labels by specific_type
    # only include labels that appear in the embedding space and meet min_frequency
    type_to_canonicals: dict[str, list[str]] = {}
    for canon, stype in canonical_type_map.items():
        if canon not in label_to_idx:
            continue
        if freq_map.get(canon, 0) < min_frequency:
            continue
        type_to_canonicals.setdefault(stype, []).append(canon)

    candidates_by_type: dict[str, list[tuple]] = {}
    total_pairs = 0
    total_filtered = 0

    for stype, canon_labels in sorted(type_to_canonicals.items()):
        if len(canon_labels) < 2:
            continue

        idxs      = np.array([label_to_idx[l] for l in canon_labels])
        group_emb = emb_norm[idxs]
        sims      = cosine_similarity(group_emb)
        np.fill_diagonal(sims, 0)

        pairs_idx = np.argwhere(sims > similarity_threshold)
        pairs_idx = pairs_idx[pairs_idx[:, 0] < pairs_idx[:, 1]]

        type_candidates = []
        for i, j in pairs_idx:
            la, lb = canon_labels[i], canon_labels[j]
            # skip if already sharing canonical (shouldn't happen since we
            # iterate over canonicals directly, but guard anyway)
            ca = raw_to_canon.get(la, la)
            cb = raw_to_canon.get(lb, lb)
            if ca == cb:
                total_filtered += 1
                continue
            type_candidates.append((la, lb, float(sims[i, j])))

        if type_candidates:
            candidates_by_type[stype] = type_candidates
            total_pairs += len(type_candidates)

        print(f"  [{stype}] {len(canon_labels):,} labels → "
              f"{len(type_candidates):,} candidate pairs")

    print(f"\n  Total candidate pairs: {total_pairs:,} across "
          f"{len(candidates_by_type):,} specific_types")
    return candidates_by_type


def find_singleton_attachment_pairs_by_type(
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_counts: pd.DataFrame,
        canonical_map: pd.DataFrame,
        canonical_type_map: dict,
        similarity_threshold: float,
        anchor_min_frequency: int = 2,
) -> dict[str, list[tuple]]:
    """Find candidate pairs between freq=1 singletons and established canonicals.

    Unlike find_candidate_pairs_by_type which does full pairwise search within
    a type group, this function only computes cross-similarity between:
      - freq=1 canonical labels (singletons to attach)  [rows]
      - freq>=anchor_min_frequency canonical labels (anchors)  [cols]

    This keeps matrix size manageable for large type groups:
      policy_instrument:       35,738 x 810  = 29M entries (~230MB float32)
      affected_community:      28,178 x 2677 = 75M entries (~600MB float32)
      institutional_framework: 19,985 x 364  =  7M entries (~58MB float32)

    Only pairs where the singleton side has frequency=1 are returned, so merges
    are always singleton -> established canonical. Established canonical ->
    established canonical pairs are handled in the main LLM loop.

    Returns a dict mapping specific_type -> list of (label_a, label_b, similarity)
    where label_a is always the freq=1 singleton.
    """
    from sklearn.preprocessing import normalize as sk_normalize
    from sklearn.metrics.pairwise import cosine_similarity

    raw_to_canon = canonical_map.set_index("raw_label")["canonical_label"].to_dict()
    freq_map     = label_counts.set_index("label")["frequency"].to_dict()
    label_to_idx = {l: i for i, l in enumerate(labels)}
    emb_norm     = sk_normalize(embeddings, norm="l2")

    # group canonical labels by type into singletons and anchors
    type_to_singletons: dict[str, list[str]] = {}
    type_to_anchors:    dict[str, list[str]] = {}

    for canon, stype in canonical_type_map.items():
        if canon not in label_to_idx:
            continue
        f = freq_map.get(canon, 0)
        if f == 1:
            type_to_singletons.setdefault(stype, []).append(canon)
        elif f >= anchor_min_frequency:
            type_to_anchors.setdefault(stype, []).append(canon)

    candidates_by_type: dict[str, list[tuple]] = {}
    total_pairs = 0

    for stype in sorted(set(type_to_singletons) & set(type_to_anchors)):
        singletons = type_to_singletons[stype]
        anchors    = type_to_anchors[stype]

        if not singletons or not anchors:
            continue

        singleton_idxs = np.array([label_to_idx[l] for l in singletons])
        anchor_idxs    = np.array([label_to_idx[l] for l in anchors])

        singleton_emb = emb_norm[singleton_idxs]
        anchor_emb    = emb_norm[anchor_idxs]

        # cross-similarity: singletons x anchors only (not singletons x singletons)
        sims = cosine_similarity(singleton_emb, anchor_emb)

        pairs_idx = np.argwhere(sims > similarity_threshold)

        type_candidates = []
        for si, ai in pairs_idx:
            la = singletons[si]   # freq=1 singleton
            lb = anchors[ai]      # established canonical
            ca = raw_to_canon.get(la, la)
            cb = raw_to_canon.get(lb, lb)
            if ca == cb:
                continue
            type_candidates.append((la, lb, float(sims[si, ai])))

        # deduplicate: keep only best anchor match per singleton
        best: dict[str, tuple] = {}
        for la, lb, sim in type_candidates:
            if la not in best or sim > best[la][2]:
                best[la] = (la, lb, sim)
        type_candidates = list(best.values())

        if type_candidates:
            candidates_by_type[stype] = type_candidates
            total_pairs += len(type_candidates)

        print(f"  [{stype}] {len(singletons):,} singletons x {len(anchors):,} anchors "
              f"-> {len(type_candidates):,} candidate pairs")

    print(f"\n  Total candidate pairs: {total_pairs:,} across "
          f"{len(candidates_by_type):,} specific_types")
    return candidates_by_type


def llm_adjudicate_pairs_by_type(
        candidates_by_type: dict[str, list[tuple]],
        system_prompt: str,
        batch_size: int = 50,
) -> dict[str, dict[int, bool]]:
    """Adjudicate candidate pairs type-by-type, prepending a type header to each batch.

    Returns a dict mapping specific_type → {pair_idx: should_merge}.
    """
    import anthropic
    client = anthropic.Anthropic()

    results_by_type: dict[str, dict[int, bool]] = {}
    total_yes = 0
    total_no  = 0

    for stype, candidates in candidates_by_type.items():
        type_results: dict[int, bool] = {}

        for batch_start in tqdm(
                range(0, len(candidates), batch_size),
                desc=f"  [{stype}]",
                unit="batch",
                leave=False,
        ):
            batch = candidates[batch_start:batch_start + batch_size]
            header = f"# specific_type: {stype}"
            lines  = [f"{i}| A: {la} | B: {lb}"
                      for i, (la, lb, _) in enumerate(batch)]
            content = header + "\n" + "\n".join(lines)

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            )

            if not response.content:
                print(f"  Warning: empty response for [{stype}] batch {batch_start}, skipping")
                continue

            if response.stop_reason == "max_tokens":
                print(f"  Warning: truncated response for [{stype}] batch {batch_start}")

            for line in response.content[0].text.strip().split("\n"):
                line = line.strip()
                if not line or line.startswith("#") or "|" not in line:
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                try:
                    local_idx = int(parts[0].strip())
                    verdict   = parts[1].strip().upper()
                    type_results[batch_start + local_idx] = (verdict == "YES")
                except (ValueError, IndexError):
                    continue

        yes = sum(type_results.values())
        no  = len(type_results) - yes
        total_yes += yes
        total_no  += no
        print(f"  [{stype}] {len(candidates):,} pairs → YES: {yes:,}, NO: {no:,}")
        results_by_type[stype] = type_results

    print(f"\n  Total — YES: {total_yes:,}, NO: {total_no:,}")
    return results_by_type


def apply_llm_merges_by_type(
        candidates_by_type: dict[str, list[tuple]],
        results_by_type: dict[str, dict[int, bool]],
        canonical_map: pd.DataFrame,
        label_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Apply confirmed LLM merge decisions to canonical_map using union-find."""
    freq_map      = label_counts.set_index("label")["frequency"].to_dict()
    raw_to_canon  = canonical_map.set_index("raw_label")["canonical_label"].to_dict()
    all_canonicals = canonical_map["canonical_label"].unique()

    parent = {l: l for l in all_canonicals}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if freq_map.get(ra, 0) >= freq_map.get(rb, 0):
            parent[rb] = ra
        else:
            parent[ra] = rb

    n_merged = 0
    for stype, candidates in candidates_by_type.items():
        type_results = results_by_type.get(stype, {})
        for pair_idx, should_merge in type_results.items():
            if not should_merge:
                continue
            la, lb, _ = candidates[pair_idx]
            ca = find(raw_to_canon.get(la, la))
            cb = find(raw_to_canon.get(lb, lb))
            if ca == cb:
                continue
            union(ca, cb)
            n_merged += 1

    print(f"  Confirmed merges applied: {n_merged}")
    result = canonical_map.copy()
    result["canonical_label"] = result["canonical_label"].map(lambda x: find(x))
    result["n_variants"] = result.groupby("canonical_label")["raw_label"].transform("count")
    return result