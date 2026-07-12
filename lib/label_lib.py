"""
All functions for the 05_label notebook.
"""

from config import label_config as cfg
from prompts.label_macro_prompt import LABEL_PROMPT_MACRO
from prompts.label_meso_prompt import LABEL_PROMPT_MESO  # updated prompt with anchor requirement
from prompts.label_micro_prompt import LABEL_PROMPT_MICRO

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import anthropic


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_assignments(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    return df

def load_extractions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    return df

def load_preprocessed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=["id", "head"])
    df["id"] = df["id"].astype(str)
    return df.drop_duplicates("id")

def load_labels(path: str) -> pd.DataFrame:
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["cluster", "title", "description"])

def save_labels(labels: list[dict], path: str) -> pd.DataFrame:
    df = pd.DataFrame(labels)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# 2. Feature preparation
# ---------------------------------------------------------------------------

def proportional_n_examples(n: int) -> int:
    return int(np.clip(n * cfg.LABEL_N_EXAMPLES_FRACTION,
                       cfg.LABEL_N_EXAMPLES_MIN,
                       cfg.LABEL_N_EXAMPLES_MAX))

def get_examples(cluster_ids: np.ndarray, extractions: pd.DataFrame,
                 preprocessed: pd.DataFrame, n: int) -> str:
    narratives = (
        extractions[extractions["id"].isin(cluster_ids.astype(str))]
        [["id", "dominant_narrative"]]
        .drop_duplicates("id")
        .dropna(subset=["dominant_narrative"])
        .merge(preprocessed[["id", "head"]], on="id", how="left")
    )
    sample = narratives.sample(min(n, len(narratives)), random_state=42)
    lines  = []
    for _, row in sample.iterrows():
        title = row["head"] if pd.notna(row.get("head")) else ""
        lines.append(f"- [{title}] {row['dominant_narrative']}")
    return "\n".join(lines)

def get_slot_fillers_macro_meso(cluster_ids: np.ndarray, extractions: pd.DataFrame) -> str:
    sub   = extractions[extractions["id"].isin(cluster_ids.astype(str))]
    lines = []
    for slot in cfg.MACRO_MESO_SLOTS:
        slot_sub = sub[sub["slot"] == slot].dropna(subset=["label", "specific_type"])
        if slot_sub.empty:
            continue
        total = len(slot_sub)
        top   = (
            slot_sub.groupby(["label", "specific_type"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(cfg.MACRO_MESO_TOP_N_FILLERS)
        )
        parts = [
            f"{row['label']} ({row['count']/total:.1%}, {row['specific_type']})"
            for _, row in top.iterrows()
        ]
        lines.append(f"{slot.capitalize()}: {', '.join(parts)}")
    return "\n".join(lines)

def get_slot_fillers_micro(cluster_ids: np.ndarray, extractions: pd.DataFrame) -> str:
    sub   = extractions[extractions["id"].isin(cluster_ids.astype(str))]
    lines = []
    for slot in cfg.MICRO_SLOTS:
        slot_sub = sub[sub["slot"] == slot].dropna(subset=["label", "specific_type"])
        if slot_sub.empty:
            continue
        total = len(slot_sub)
        top   = (
            slot_sub.groupby(["label", "specific_type"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(cfg.MICRO_TOP_N_FILLERS)
        )
        parts = [
            f"{row['label']} ({row['count']/total:.1%}, {row['specific_type']})"
            for _, row in top.iterrows()
        ]
        lines.append(f"{slot.capitalize()}: {', '.join(parts)}")
    return "\n".join(lines)

def get_dominant_entities(cluster_ids: np.ndarray, extractions: pd.DataFrame,
                          top_n: int = 50) -> str:
    """Return top-N most frequent named entities for subject and object slots,
    with per-article coverage, so the LLM can reason about what anchor covers 80%+."""
    sub        = extractions[extractions["id"].isin(cluster_ids.astype(str))]
    n_articles = sub["id"].nunique()
    lines      = [f"(Total articles in cluster: {n_articles})"]
    for slot in ["subject", "object"]:
        slot_sub = sub[sub["slot"] == slot].dropna(subset=["label", "specific_type"])
        if slot_sub.empty:
            continue
        top = (
            slot_sub.groupby(["label", "specific_type"])["id"]
            .nunique()
            .reset_index(name="n_articles")
            .sort_values("n_articles", ascending=False)
            .head(top_n)
        )
        lines.append(f"\n{slot.upper()} (top {min(top_n, len(top))} of {slot_sub['label'].nunique()} unique entities):")
        cumulative = 0
        for _, row in top.iterrows():
            pct        = row["n_articles"] / n_articles
            cumulative += pct
            lines.append(
                f"  {row['n_articles']:>4} arts ({pct:.1%}, cumul {cumulative:.1%})  "
                f"{row['label']}  [{row['specific_type']}]"
            )
    return "\n".join(lines)

def format_existing_labels(labels: list[dict], n: int) -> str:
    recent = labels[-n:] if len(labels) > n else labels
    if not recent:
        return "(none yet)"
    return "\n".join(f"- {l['title']}" for l in recent if l.get("title"))


# ---------------------------------------------------------------------------
# 3. LLM labelling
# ---------------------------------------------------------------------------

def call_llm(prompt: str, client: anthropic.Anthropic) -> dict:
    response = client.messages.create(
        model=cfg.MODEL,
        max_tokens=cfg.MAX_TOKENS,
        temperature=cfg.TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    print(f"  RAW RESPONSE: {repr(text)}")
    raise json.JSONDecodeError("No valid JSON found", text, 0)

def label_macro(assignments: pd.DataFrame, extractions: pd.DataFrame,
                preprocessed: pd.DataFrame, client: anthropic.Anthropic,
                existing_labels: pd.DataFrame) -> list[dict]:
    already_done   = set(existing_labels["cluster"].tolist()) if len(existing_labels) else set()
    cluster_ids    = sorted(assignments["macro_cluster"].dropna().unique())
    labels         = existing_labels.to_dict("records")
    sibling_labels = list(existing_labels.to_dict("records"))  # track all macro labels for uniqueness

    for cid in cluster_ids:
        if cid in already_done:
            print(f"  Skipping macro {int(cid)} (already labelled)")
            continue

        ids          = assignments[assignments["macro_cluster"] == cid]["id"].values
        n_examples   = proportional_n_examples(len(ids))
        examples     = get_examples(ids, extractions, preprocessed, n_examples)
        slot_fillers = get_slot_fillers_macro_meso(ids, extractions)
        existing_str = format_existing_labels(sibling_labels, cfg.N_RECENT_LABELS_CONTEXT)

        prompt = LABEL_PROMPT_MACRO.format(
            existing_labels=existing_str,
            slot_fillers=slot_fillers,
            n_examples=n_examples,
            examples=examples,
        )

        try:
            result = call_llm(prompt, client)
            entry = {
                "cluster":     cid,
                "title":       result.get("title", ""),
                "description": result.get("description", ""),
            }
            labels.append(entry)
            sibling_labels.append(entry)
            print(f"  macro {int(cid)}: {result.get('title', '')}")
        except Exception as e:
            print(f"  ERROR macro {int(cid)}: {e}")
            labels.append({"cluster": cid, "title": "", "description": ""})

        time.sleep(0.5)

    return labels

def label_meso(assignments: pd.DataFrame, extractions: pd.DataFrame,
               preprocessed: pd.DataFrame, client: anthropic.Anthropic,
               existing_labels: pd.DataFrame) -> list[dict]:
    already_done = set(existing_labels["cluster"].tolist()) if len(existing_labels) else set()
    cluster_ids  = sorted(assignments["meso_cluster"].dropna().unique())
    labels       = existing_labels.to_dict("records")

    # load macro labels for context and display
    macro_label_map = {}
    if Path(cfg.MACRO_LABELS_CSV).exists():
        _ml = pd.read_csv(cfg.MACRO_LABELS_CSV)
        macro_label_map = _ml.set_index("cluster")[["title", "description"]].to_dict("index")

    current_macro        = None
    current_macro_labels = []  # labels generated so far within current macro group

    for cid in cluster_ids:
        macro_id = int(cid) // 1000

        # reset sibling label context when macro group changes
        if macro_id != current_macro:
            current_macro        = macro_id
            current_macro_labels = []
            macro_info           = macro_label_map.get(float(macro_id), {})
            macro_title          = macro_info.get("title", f"Macro {macro_id}")
            macro_desc           = macro_info.get("description", "")
            print(f"\n── Macro {macro_id}: {macro_title} ──")

        if cid in already_done:
            print(f"  Skipping meso {int(cid)} (already labelled)")
            continue

        ids              = assignments[assignments["meso_cluster"] == cid]["id"].values
        n_examples       = proportional_n_examples(len(ids))
        examples         = get_examples(ids, extractions, preprocessed, n_examples)
        slot_fillers     = get_slot_fillers_macro_meso(ids, extractions)
        dominant_ents    = get_dominant_entities(ids, extractions)
        existing_str     = format_existing_labels(current_macro_labels, cfg.N_RECENT_LABELS_CONTEXT)

        prompt = LABEL_PROMPT_MESO.format(
            parent_title=macro_title,
            parent_description=macro_desc,
            existing_labels=existing_str,
            dominant_entities=dominant_ents,
            slot_fillers=slot_fillers,
            n_examples=n_examples,
            examples=examples,
        )

        try:
            result = call_llm(prompt, client)
            entry = {
                "cluster":     cid,
                "title":       result.get("title", ""),
                "description": result.get("description", ""),
            }
            labels.append(entry)
            current_macro_labels.append(entry)
            print(f"  meso {int(cid)}: {result.get('title', '')}")
        except Exception as e:
            print(f"  ERROR meso {int(cid)}: {e}")
            labels.append({"cluster": cid, "title": "", "description": ""})

        time.sleep(0.5)

    return labels

def label_micro(micro_assignments: pd.DataFrame, extractions: pd.DataFrame,
                preprocessed: pd.DataFrame, meso_labels: pd.DataFrame,
                client: anthropic.Anthropic,
                existing_labels: pd.DataFrame) -> list[dict]:
    already_done   = set(existing_labels["cluster"].tolist()) if len(existing_labels) else set()
    cluster_ids    = sorted(micro_assignments["micro_cluster"].dropna().unique())
    meso_label_map = meso_labels.set_index("cluster")[["title", "description"]].to_dict("index")
    labels         = existing_labels.to_dict("records")

    # load macro labels for context and display
    macro_label_map = {}
    if Path(cfg.MACRO_LABELS_CSV).exists():
        _ml = pd.read_csv(cfg.MACRO_LABELS_CSV)
        macro_label_map = _ml.set_index("cluster")[["title", "description"]].to_dict("index")

    current_macro        = None
    current_meso         = None
    current_meso_labels  = []  # labels generated so far within current meso group

    for cid in cluster_ids:
        meso_id  = int(cid) // 1000
        macro_id = meso_id // 1000

        # update macro context when macro group changes
        if macro_id != current_macro:
            current_macro = macro_id
            macro_info    = macro_label_map.get(float(macro_id), {})
            macro_title   = macro_info.get("title", f"Macro {macro_id}")
            macro_desc    = macro_info.get("description", "")
            print(f"\n── Macro {macro_id}: {macro_title} ──")

        # reset sibling label context when meso group changes
        if meso_id != current_meso:
            current_meso        = meso_id
            current_meso_labels = []
            meso_info           = meso_label_map.get(float(meso_id), {})
            meso_title          = meso_info.get("title", f"Meso {meso_id}")
            meso_desc           = meso_info.get("description", "")
            print(f"  ── Meso {meso_id}: {meso_title} ──")

        if cid in already_done:
            print(f"    Skipping micro {int(cid)} (already labelled)")
            continue

        ids           = micro_assignments[micro_assignments["micro_cluster"] == cid]["id"].values
        n_examples    = proportional_n_examples(len(ids))
        examples      = get_examples(ids, extractions, preprocessed, n_examples)
        slot_fillers  = get_slot_fillers_micro(ids, extractions)
        dominant_ents = get_dominant_entities(ids, extractions)
        existing_str  = format_existing_labels(current_meso_labels, cfg.N_RECENT_LABELS_CONTEXT)

        prompt = LABEL_PROMPT_MICRO.format(
            macro_title=macro_title,
            macro_description=macro_desc,
            meso_title=meso_title,
            meso_description=meso_desc,
            existing_labels=existing_str,
            dominant_entities=dominant_ents,
            slot_fillers=slot_fillers,
            n_examples=n_examples,
            examples=examples,
        )

        try:
            result = call_llm(prompt, client)
            entry = {
                "cluster":     cid,
                "meso_parent": meso_id,
                "title":       result.get("title", ""),
                "description": result.get("description", ""),
            }
            labels.append(entry)
            current_meso_labels.append(entry)
            print(f"    micro {int(cid)}: {result.get('title', '')}")
        except Exception as e:
            print(f"    ERROR micro {int(cid)}: {e}")
            labels.append({"cluster": cid, "meso_parent": meso_id,
                           "title": "", "description": ""})

        time.sleep(0.5)

    return labels


# ---------------------------------------------------------------------------
# 4. Merge labels
# ---------------------------------------------------------------------------

def merge_labels(macro_labels: pd.DataFrame, meso_labels: pd.DataFrame,
                 micro_labels: pd.DataFrame) -> pd.DataFrame:
    macro_labels["level"] = "macro"
    meso_labels["level"]  = "meso"
    micro_labels["level"] = "micro"
    return pd.concat([macro_labels, meso_labels, micro_labels], ignore_index=True)