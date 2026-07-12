"""Aggregation functions for the visualisation precompute pipeline.

Produces four tidy tables consumed by the Streamlit app:
  - article_table       : one row per article (assignments + metrics + iso_week)
  - cluster_meta        : one row per cluster per level (labels, sizes, avg metrics,
                          primary domain, top entities per slot, representative examples)
  - cluster_publisher   : one row per cluster x publisher per level (counts, both rel
                          normalisations, avg metrics, n for the floor)
  - cluster_week        : one row per cluster x iso_week per level (count, within-week share)
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from config import precompute_config as cfg

_XML_TAG = re.compile(r'<[^>]+')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_preprocessed(path: str) -> pd.DataFrame:
    """Per-article metrics + publisher + pubtime. id as string.

    The source data contains some ids that appear multiple times with different
    pubtimes (republications). Since extractions and cluster assignments are one
    row per id, we deduplicate to one row per id, keeping the earliest pubtime so
    the date marks when the narrative first appeared.
    """
    df = pd.read_csv(path, sep="\t")
    df["id"] = df["id"].astype(str)
    df["_ts"] = pd.to_datetime(df["pubtime"], utc=True, errors="coerce")
    df = (
        df.sort_values("_ts")
        .drop_duplicates(subset="id", keep="first")
        .drop(columns="_ts")
        .reset_index(drop=True)
    )
    return df


def load_assignments(macro_path, meso_path, micro_path) -> pd.DataFrame:
    """Merge the three assignment CSVs into one wide table keyed on id."""
    macro = pd.read_csv(macro_path)
    meso  = pd.read_csv(meso_path)
    micro = pd.read_csv(micro_path)
    for d in (macro, meso, micro):
        d["id"] = d["id"].astype(str)
    out = (
        macro[["id", "macro_cluster"]]
        .merge(meso[["id", "meso_cluster"]],  on="id", how="left")
        .merge(micro[["id", "micro_cluster"]], on="id", how="left")
    )
    return out


def load_labels(macro_path, meso_path, micro_path) -> dict[str, pd.DataFrame]:
    """Return {level: labels_df} with columns cluster,title,description."""
    out = {}
    for level, path in [("macro", macro_path), ("meso", meso_path), ("micro", micro_path)]:
        df = pd.read_csv(path)
        df["cluster"] = df["cluster"].astype(float)
        out[level] = df
    return out


# ---------------------------------------------------------------------------
# Metric transforms
# ---------------------------------------------------------------------------
def add_transformed_metrics(preprocessed: pd.DataFrame) -> pd.DataFrame:
    """Add app-facing metric columns with consistent 'higher = more' direction.

    - complexity = 100 - flesch  (higher = more complex; raw flesch kept as-is)
    - sentiment  = sentiment_score (unchanged, signed)
    - anger/fear/joy/sadness = emotion_* (unchanged, 0..1)
    """
    df = preprocessed.copy()
    df["complexity"] = 100.0 - df[cfg.FLESCH_COL]
    df["sentiment"]  = df[cfg.SENTIMENT_COL]
    df["anger"]      = df["emotion_anger"]
    df["fear"]       = df["emotion_fear"]
    df["joy"]        = df["emotion_joy"]
    df["sadness"]    = df["emotion_sadness"]
    return df


# ---------------------------------------------------------------------------
# Time bucketing
# ---------------------------------------------------------------------------
def add_iso_week(df: pd.DataFrame, trim_partial: bool = True) -> pd.DataFrame:
    """Add an iso_week column (Monday-dated) from pubtime. Optionally trim
    the leading and trailing partial weeks so every bucket is a full 7 days."""
    out = df.copy()
    ts = pd.to_datetime(out["pubtime"], utc=True, errors="coerce")
    # Monday of the ISO week, as a date
    out["iso_week"] = (ts - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.date
    out["iso_week"] = pd.to_datetime(out["iso_week"])

    if trim_partial:
        # Drop the first week if the corpus doesn't start on a Monday, and the
        # last week if it doesn't end on a Sunday — those buckets are partial.
        span_min = ts.min()
        span_max = ts.max()
        weeks = np.sort(out["iso_week"].dropna().unique())
        if len(weeks):
            drop = set()
            if span_min.weekday() != 0:      # 0 = Monday
                drop.add(weeks[0])
            if span_max.weekday() != 6:      # 6 = Sunday
                drop.add(weeks[-1])
            out = out[~out["iso_week"].isin(drop)]
    return out


# ---------------------------------------------------------------------------
# Representative examples (nearest to centroid in narrative embedding space)
# ---------------------------------------------------------------------------
def representative_examples(
    cluster_article_ids: np.ndarray,
    embeddings: np.ndarray,
    embedding_ids: np.ndarray,
    id_to_row: dict,
    n: int,
) -> list:
    """Return up to n article ids nearest to the cluster centroid (cosine)."""
    rows = [id_to_row[a] for a in cluster_article_ids if a in id_to_row]
    if not rows:
        return []
    sub = embeddings[rows]
    # L2-normalise then centroid; cosine similarity = dot with normalised centroid
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sub_n = sub / norms
    centroid = sub_n.mean(axis=0)
    cnorm = np.linalg.norm(centroid)
    if cnorm == 0:
        return []
    centroid /= cnorm
    sims = sub_n @ centroid
    order = np.argsort(-sims)[:n]
    ids_present = [a for a in cluster_article_ids if a in id_to_row]
    return [ids_present[i] for i in order]


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------
def build_article_table(assignments: pd.DataFrame, preprocessed: pd.DataFrame) -> pd.DataFrame:
    """One row per article: assignments + raw + transformed metrics + iso_week + head."""
    keep_pre = ["id", "pubtime", "publisher", "head",
                cfg.SENTIMENT_COL, cfg.FLESCH_COL] + cfg.EMOTION_COLS + \
               ["complexity", "sentiment", "anger", "fear", "joy", "sadness", "iso_week",
                "subhead", "content"]
    keep_pre = [c for c in keep_pre if c in preprocessed.columns]
    art = assignments.merge(preprocessed[keep_pre], on="id", how="inner")

    # Build a plain-text snippet (prefer subhead; fall back to cleaned content)
    has_sub = "subhead" in art.columns
    has_con = "content" in art.columns
    if has_sub or has_con:
        if has_con:
            clean_content = (
                art["content"].fillna("")
                .str.replace(_XML_TAG, " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
        if has_sub:
            sub = art["subhead"].fillna("").str.strip()
            snippet = sub.where(sub != "", clean_content if has_con else "")
        else:
            snippet = clean_content
        art["snippet"] = snippet.str[:300]
        if has_sub:
            art = art.drop(columns=["subhead"])
        if has_con:
            art = art.drop(columns=["content"])

    return art


def _avg_metric_cols(g: pd.DataFrame) -> dict:
    return {m: g[m].mean() for m in cfg.AVG_METRICS}


def build_cluster_meta(
    article_table: pd.DataFrame,
    labels: dict,
    extractions: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_ids: np.ndarray,
) -> pd.DataFrame:
    """One row per cluster per level."""
    n_total = article_table["id"].nunique()
    id_to_row = {str(i): r for r, i in enumerate(embedding_ids)}

    records = []
    for level in cfg.LEVELS:
        col = f"{level}_cluster"
        label_df = labels[level].set_index("cluster")
        parent_col = {"macro": None, "meso": "macro_cluster", "micro": "meso_cluster"}[level]

        for cid, g in article_table.dropna(subset=[col]).groupby(col):
            art_ids = g["id"].values
            row = {
                "level":        level,
                "cluster":      cid,
                "parent":       (int(cid) // 1000) if parent_col else np.nan,
                "n_articles":   len(art_ids),
                "corpus_share": len(art_ids) / n_total,
            }
            # labels
            if cid in label_df.index:
                row["title"]       = label_df.loc[cid, "title"]
                row["description"] = label_df.loc[cid, "description"]
            else:
                row["title"], row["description"] = "", ""
            # averaged metrics
            row.update(_avg_metric_cols(g))
            # primary domain + top entities per slot from extractions
            ext = extractions[extractions["id"].isin(art_ids.astype(str))]
            if not ext.empty and "domain" in ext.columns:
                dom = ext["domain"].dropna()
                row["primary_domain"] = dom.value_counts().idxmax() if not dom.empty else ""
            else:
                row["primary_domain"] = ""
            row["top_entities"]     = _top_entities_per_slot(ext)
            row["slot_type_groups"] = _top_slot_type_groups(ext)
            # representative examples
            reps = representative_examples(
                art_ids.astype(str), embeddings, embedding_ids, id_to_row,
                cfg.N_REPRESENTATIVE_EXAMPLES,
            )
            row["representative_ids"] = ",".join(reps)
            records.append(row)

    return pd.DataFrame(records)


def _top_slot_type_groups(ext: pd.DataFrame) -> str:
    """JSON-encoded slot type groups for the typed card view.
    {slot: [{filler_type, category, specific_type, labels, n_articles}]}
    Groups are sorted by n_articles desc, capped at TOP_TYPE_GROUPS_PER_SLOT.
    Labels within each group are the top TOP_ENTITIES_PER_SLOT by article frequency."""
    import json
    if ext.empty:
        return "{}"

    result = {}
    for slot in cfg.CARD_SLOTS:
        s = ext[ext["slot"] == slot].dropna(subset=["label"]).copy()
        if s.empty:
            continue

        # Ensure type columns exist (guard against older extraction data)
        for col in ("filler_type", "category", "specific_type"):
            if col not in s.columns:
                s[col] = None

        # Fill NA for groupby keys, then restore None in output
        s["_ftype"] = s["filler_type"].fillna("")
        s["_cat"]   = s["category"].fillna("")
        s["_stype"] = s["specific_type"].fillna("")

        groups = []
        for (ftype, cat, stype), g in s.groupby(["_ftype", "_cat", "_stype"]):
            top_labels = (
                g.groupby("label")["id"].nunique()
                .sort_values(ascending=False)
                .head(cfg.TOP_ENTITIES_PER_SLOT)
                .index.tolist()
            )
            if not top_labels:
                continue
            groups.append({
                "filler_type":   ftype or None,
                "category":      cat   or None,
                "specific_type": stype or None,
                "labels":        top_labels,
                "n_articles":    int(g["id"].nunique()),
            })

        groups.sort(key=lambda x: x["n_articles"], reverse=True)
        groups = groups[:cfg.TOP_TYPE_GROUPS_PER_SLOT]

        if groups:
            result[slot] = groups

    return json.dumps(result, ensure_ascii=False)


def _top_entities_per_slot(ext: pd.DataFrame) -> str:
    """Compact slot->entities string for the card, top N per slot by article freq.
    Uses delimiters unlikely to appear in entity labels: slots joined by ' ¦¦ ',
    entities within a slot by ' ¦ ', slot name separated by '::'."""
    if ext.empty:
        return ""
    parts = []
    for slot in cfg.CARD_SLOTS:
        s = ext[ext["slot"] == slot].dropna(subset=["label"])
        if s.empty:
            continue
        top = (
            s.groupby("label")["id"].nunique()
            .sort_values(ascending=False)
            .head(cfg.TOP_ENTITIES_PER_SLOT)
            .index.tolist()
        )
        parts.append(f"{slot}::" + " ¦ ".join(top))
    return " ¦¦ ".join(parts)


def build_slot_fillers(extractions: pd.DataFrame) -> pd.DataFrame:
    """Slim per-filler table the app aggregates at query time into slot-type groups.

    One row per labelled filler in one of the six card slots, carrying only the
    columns the card needs (id, slot, label, filler_type, category, specific_type).
    This replaces the full raw extractions CSV as a runtime dependency of the app.
    """
    keep = ["id", "slot", "label", "filler_type", "category", "specific_type"]
    df = extractions[extractions["slot"].isin(cfg.CARD_SLOTS)].copy()
    df = df.dropna(subset=["label"])
    for c in ("filler_type", "category", "specific_type"):
        if c not in df.columns:
            df[c] = None
    df["id"] = df["id"].astype(str)
    return df[keep].reset_index(drop=True)


def build_entity_ftype(extractions: pd.DataFrame) -> pd.DataFrame:
    """Map each canonical entity label to its most common filler_type.

    Precomputes what the app previously derived with a full groupby+mode over the
    raw extractions at runtime. Returns columns [label, filler_type].
    """
    if extractions.empty or "label" not in extractions.columns \
            or "filler_type" not in extractions.columns:
        return pd.DataFrame(columns=["label", "filler_type"])
    return (
        extractions.dropna(subset=["label", "filler_type"])
        .groupby("label")["filler_type"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )


def build_cluster_publisher(article_table: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster x publisher per level.
    Includes both rel normalisations and averaged metrics with cell n."""
    records = []
    for level in cfg.LEVELS:
        col = f"{level}_cluster"
        sub = article_table.dropna(subset=[col])

        # totals for normalisation
        pub_totals     = sub.groupby("publisher")["id"].nunique()          # within-publisher denom
        cluster_totals = sub.groupby(col)["id"].nunique()                  # within-narrative denom

        for (cid, pub), g in sub.groupby([col, "publisher"]):
            n = len(g)
            rec = {
                "level":            level,
                "cluster":          cid,
                "publisher":        pub,
                "n_articles":       n,
                "rel_by_publisher": n / pub_totals[pub]     if pub_totals[pub]     else 0.0,
                "rel_by_narrative": n / cluster_totals[cid] if cluster_totals[cid] else 0.0,
            }
            rec.update(_avg_metric_cols(g))
            records.append(rec)
    return pd.DataFrame(records)


def build_cluster_week(article_table: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster x iso_week per level: count and within-week share."""
    records = []
    for level in cfg.LEVELS:
        col = f"{level}_cluster"
        sub = article_table.dropna(subset=[col, "iso_week"])
        week_totals = sub.groupby("iso_week")["id"].nunique()
        for (cid, wk), g in sub.groupby([col, "iso_week"]):
            n = len(g)
            records.append({
                "level":           level,
                "cluster":         cid,
                "iso_week":        wk,
                "n_articles":      n,
                "within_week_share": n / week_totals[wk] if week_totals[wk] else 0.0,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Search: entity index, entity vocab + embeddings, cluster centroids
# ---------------------------------------------------------------------------
def build_entity_index(article_table: pd.DataFrame, extractions: pd.DataFrame) -> pd.DataFrame:
    """Inverted index: for each entity label, which clusters it appears in,
    in which slot, at what article-level prominence. One row per
    entity x level x cluster x slot.

    Entities are the raw (possibly un-normalised) slot labels. Semantic search
    over their embeddings sidesteps surface-form fragmentation.
    """
    # article -> cluster maps per level
    level_maps = {lvl: article_table.set_index("id")[f"{lvl}_cluster"] for lvl in cfg.LEVELS}
    cluster_sizes = {
        lvl: article_table.dropna(subset=[f"{lvl}_cluster"]).groupby(f"{lvl}_cluster")["id"].nunique()
        for lvl in cfg.LEVELS
    }

    ext = extractions.dropna(subset=["label"])[["id", "slot", "label"]].copy()
    records = []
    for lvl in cfg.LEVELS:
        cmap = level_maps[lvl]
        e = ext.copy()
        e["cluster"] = e["id"].map(cmap)
        e = e.dropna(subset=["cluster"])
        # article-level counts per entity x cluster x slot
        grp = (
            e.groupby(["label", "cluster", "slot"])["id"]
            .nunique()
            .reset_index(name="n_articles")
        )
        grp["level"] = lvl
        grp["pct_of_cluster"] = grp.apply(
            lambda r: r["n_articles"] / cluster_sizes[lvl].get(r["cluster"], np.nan), axis=1
        )
        records.append(grp)
    out = pd.concat(records, ignore_index=True)
    return out[["label", "level", "cluster", "slot", "n_articles", "pct_of_cluster"]]


def build_entity_vocab(entity_index: pd.DataFrame) -> pd.DataFrame:
    """Distinct entity labels with aggregate stats, parallel to entity_embeddings."""
    agg = (
        entity_index.groupby("label")
        .agg(total_articles=("n_articles", "sum"),
             n_clusters=("cluster", "nunique"),
             slots=("slot", lambda s: ",".join(sorted(set(s)))))
        .reset_index()
        .sort_values("total_articles", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def embed_entities(vocab: pd.DataFrame, model, batch_size: int) -> np.ndarray:
    """Embed each distinct entity label. Matches cluster-pipeline convention:
    normalize_embeddings=True, NO e5 prefix (so query vectors align)."""
    return model.encode(
        vocab["label"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def build_cluster_centroids(
    article_table: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_ids: np.ndarray,
):
    """Centroid of each cluster's articles in the narrative embedding space,
    for thematic 'narratives about X' search. Returns (centroids, keys_df)."""
    id_to_row = {str(i): r for r, i in enumerate(embedding_ids)}
    centroids, keys = [], []
    for lvl in cfg.LEVELS:
        col = f"{lvl}_cluster"
        for cid, g in article_table.dropna(subset=[col]).groupby(col):
            rows = [id_to_row[a] for a in g["id"].astype(str) if a in id_to_row]
            if not rows:
                continue
            sub = embeddings[rows]
            c = sub.mean(axis=0)
            n = np.linalg.norm(c)
            if n == 0:
                continue
            centroids.append(c / n)
            keys.append({"level": lvl, "cluster": cid})
    return np.vstack(centroids), pd.DataFrame(keys)


# ---------------------------------------------------------------------------
# Search: query-time functions (used by the Streamlit app)
# ---------------------------------------------------------------------------
def embed_narratives_for_centroids(
    article_table: pd.DataFrame,
    extractions: pd.DataFrame,
    model,
    batch_size: int,
):
    """Embed each article's dominant_narrative with the SEARCH model, using the
    e5 passage prefix, for thematic centroid search. Returns (emb, ids) aligned.

    Built fresh (not reused from the e5-large cluster embeddings) so the thematic
    space matches the search model queries land in.
    """
    narr = (
        extractions[["id", "dominant_narrative"]]
        .dropna(subset=["dominant_narrative"])
        .drop_duplicates("id")
    )
    narr = narr[narr["id"].isin(article_table["id"])]
    texts = [cfg.E5_PASSAGE_PREFIX + t for t in narr["dominant_narrative"].tolist()]
    emb = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return emb, narr["id"].values.astype(str)


def embed_query_entity(query: str, model) -> np.ndarray:
    """Embed a query for ENTITY search: prefix-free, matching the cluster-pipeline
    convention under which entity labels were embedded."""
    return model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]


def embed_query_thematic(query: str, model) -> np.ndarray:
    """Embed a query for THEMATIC search: e5 query prefix, matching the passage-
    prefixed narrative embeddings the centroids are built from."""
    return model.encode(
        [cfg.E5_QUERY_PREFIX + query],
        convert_to_numpy=True, normalize_embeddings=True,
    )[0]


def embed_query(query: str, model) -> np.ndarray:
    """Deprecated alias — use embed_query_entity / embed_query_thematic.
    Kept for backwards compatibility; behaves as entity (prefix-free)."""
    return embed_query_entity(query, model)


def semantic_entity_search(
    query_vec: np.ndarray,
    entity_embeddings: np.ndarray,
    entity_vocab: pd.DataFrame,
    threshold: float = None,
    max_entities: int = None,
) -> pd.DataFrame:
    """Return entities most semantically similar to the query, above threshold.
    Columns: label, similarity, total_articles, n_clusters, slots."""
    threshold    = cfg.SEARCH_ENTITY_SIM_THRESHOLD if threshold is None else threshold
    max_entities = cfg.SEARCH_MAX_ENTITIES if max_entities is None else max_entities
    sims = entity_embeddings @ query_vec           # both normalised -> cosine
    order = np.argsort(-sims)
    hits = []
    for i in order[:max_entities]:
        if sims[i] < threshold:
            break
        row = entity_vocab.iloc[i]
        hits.append({
            "label": row["label"], "similarity": float(sims[i]),
            "total_articles": int(row["total_articles"]),
            "n_clusters": int(row["n_clusters"]), "slots": row["slots"],
        })
    return pd.DataFrame(hits)


def entities_to_clusters(
    matched_entities: pd.DataFrame,
    entity_index: pd.DataFrame,
    level: str,
    slot_filter: list = None,
) -> pd.DataFrame:
    """Map matched entities to clusters at a level, preserving slot breakdown.
    Returns clusters ranked by summed article prominence, with which slots the
    matched entities fill. slot_filter restricts to e.g. ['subject']."""
    labels = set(matched_entities["label"])
    idx = entity_index[(entity_index["level"] == level) & (entity_index["label"].isin(labels))]
    if slot_filter:
        idx = idx[idx["slot"].isin(slot_filter)]
    if idx.empty:
        return pd.DataFrame(columns=["cluster", "n_articles", "slots", "entities"])
    out = (
        idx.groupby("cluster")
        .agg(n_articles=("n_articles", "sum"),
             slots=("slot", lambda s: ",".join(sorted(set(s)))),
             entities=("label", lambda s: ",".join(sorted(set(s)))))
        .reset_index()
        .sort_values("n_articles", ascending=False)
        .reset_index(drop=True)
    )
    return out


def thematic_cluster_search(
    query_vec: np.ndarray,
    cluster_centroids: np.ndarray,
    centroid_keys: pd.DataFrame,
    level: str,
    top_k: int = 20,
) -> pd.DataFrame:
    """Return clusters whose narrative centroid is most similar to the query.
    Answers 'narratives about X' rather than 'X fills a slot'."""
    mask = (centroid_keys["level"] == level).values
    sims = cluster_centroids[mask] @ query_vec
    keys = centroid_keys[mask].reset_index(drop=True)
    order = np.argsort(-sims)[:top_k]
    return pd.DataFrame({
        "cluster": keys.iloc[order]["cluster"].values,
        "similarity": sims[order],
    })


# ---------------------------------------------------------------------------
# 2D semantic map (UMAP of the e5-large narrative embeddings)
# ---------------------------------------------------------------------------
def build_2d_map(
    article_table: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_ids: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
):
    """Project the e5-large narrative embeddings to 2D for the semantic map.

    Returns (xy_df, centroids_df):
      xy_df:        one row per article with x_2d, y_2d + cluster assignments
      centroids_df: one row per cluster per level with x, y (mean of members), n_articles

    This is a separate 2D projection used ONLY for visualisation — it does not
    affect clustering. Clusters appear as spatial groups because semantic
    similarity drove much of the clustering.
    """
    import umap

    # Align embeddings to the articles we actually have in the table
    id_to_row = {str(i): r for r, i in enumerate(embedding_ids)}
    art_ids = [a for a in article_table["id"].astype(str) if a in id_to_row]
    rows = [id_to_row[a] for a in art_ids]
    sub = embeddings[rows]

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, metric="cosine", random_state=42)
    xy = reducer.fit_transform(sub)

    xy_df = pd.DataFrame({"id": art_ids, "x_2d": xy[:, 0], "y_2d": xy[:, 1]})
    xy_df = xy_df.merge(
        article_table[["id", "macro_cluster", "meso_cluster", "micro_cluster"]],
        on="id", how="left",
    )

    # per-cluster centroids in 2D (mean of member points), per level
    cent_records = []
    for level in cfg.LEVELS:
        col = f"{level}_cluster"
        g = xy_df.dropna(subset=[col]).groupby(col)
        c = g[["x_2d", "y_2d"]].mean()
        n = g["id"].nunique()
        for cid in c.index:
            cent_records.append({
                "level": level, "cluster": cid,
                "x": float(c.loc[cid, "x_2d"]), "y": float(c.loc[cid, "y_2d"]),
                "n_articles": int(n.loc[cid]),
            })
    centroids_df = pd.DataFrame(cent_records)
    return xy_df, centroids_df


def build_entity_article_bridge(article_table: pd.DataFrame, extractions: pd.DataFrame) -> pd.DataFrame:
    """(label, slot, id) rows: which article contains which entity label in which
    slot. Restricted to articles present in the article_table. Enables entity
    search to scope card/stats to the specific matching articles, honouring a
    per-entity slot constraint."""
    ext = extractions.dropna(subset=["label"])[["id", "slot", "label"]].copy()
    ext["id"] = ext["id"].astype(str)
    ext = ext[ext["id"].isin(article_table["id"].astype(str))]
    return ext.drop_duplicates(["label", "slot", "id"]).reset_index(drop=True)