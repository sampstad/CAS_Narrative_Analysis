"""
All functions for the 03_extract notebook.
"""

from config import extract_config as cfg
from prompts.extract_prompt import SYSTEM_PROMPT

import anthropic
import json
import time
import collections
import jsonschema
import pandas as pd
from pathlib import Path
from tqdm import tqdm





# ---------------------------------------------------------------------------
# 1. Prompt construction
# ---------------------------------------------------------------------------

# concatenate headline + content into a single user message for the model
def make_user_message(head: str, content: str) -> str:
    return f"Headline: {head}\n\n{content}"


# ---------------------------------------------------------------------------
# 2. State tracking
# ---------------------------------------------------------------------------

# load per-article extraction state from disk (dispatched / complete / failed / retry_count)
def load_state() -> dict:
    if Path(cfg.STATE_PATH).exists():
        with open(cfg.STATE_PATH) as f:
            return json.load(f)
    return {}

# save per-article extraction state to disk
def save_state(state: dict):
    with open(cfg.STATE_PATH, "w") as f:
        json.dump(state, f)

# return only articles where narratives have not yet been successfully extracted
def get_pending(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    completed = {k for k, v in state.items() if v["status"] == "complete"}
    return df[~df["id"].isin(completed)].copy()


# ---------------------------------------------------------------------------
# 3. Batch dispatch & collection
# ---------------------------------------------------------------------------


# format a dataframe chunk into the list of request objects the Batches API expects
def build_batch_requests(batch_df: pd.DataFrame) -> list[dict]:
    return [
        {
            "custom_id": str(row["id"]),
            "params": {
                "model":       cfg.MODEL,
                "max_tokens":  cfg.MAX_TOKENS,
                "cache_control": {"type": "ephemeral"},
                "temperature": cfg.TEMPERATURE,
                "system":      SYSTEM_PROMPT,
                "messages":    [{"role": "user", "content": make_user_message(row["head"], row["content"])}],
            }
        }
        for _, row in batch_df.iterrows()
    ]

# send one chunk of articles to the Batches API and record their status as dispatched
def dispatch_batch(client: anthropic.Anthropic, batch_df: pd.DataFrame, state: dict) -> str:
    requests = build_batch_requests(batch_df)
    batch    = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    for article_id in batch_df["id"].astype(str):
        state[article_id] = {
            "status":      "dispatched",
            "batch_id":    batch_id,
            "retry_count": state.get(article_id, {}).get("retry_count", 0),
        }
    save_state(state)
    return batch_id

# dispatch all pending articles in chunks and return the list of batch IDs
def dispatch_all(client: anthropic.Anthropic, df: pd.DataFrame, state: dict) -> list[str]:
    pending  = get_pending(df, state)
    batch_ids = []
    for start in tqdm(range(0, len(pending), cfg.BATCH_SIZE), desc="Dispatching batches"):
        chunk    = pending.iloc[start:start + cfg.BATCH_SIZE]
        batch_id = dispatch_batch(client, chunk, state)
        batch_ids.append(batch_id)
        print(f"  Dispatched {batch_id} ({len(chunk):,} articles)")
    return batch_ids

# block until all batches report ended, polling every cfg.POLL_INTERVAL seconds
def poll_until_complete(client: anthropic.Anthropic, batch_ids: list[str]):
    remaining = set(batch_ids)
    while remaining:
        still_running = set()
        for batch_id in remaining:
            batch  = client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            print(f"  {batch_id}: {batch.processing_status} — "
                  f"processing={counts.processing} "
                  f"succeeded={counts.succeeded} "
                  f"errored={counts.errored}")
            if batch.processing_status != "ended":
                still_running.add(batch_id)
        remaining = still_running
        if remaining:
            time.sleep(cfg.POLL_INTERVAL)

# stream completed results from the API into the raw JSONL file and update state
def collect_results(client: anthropic.Anthropic, batch_ids: list[str], state: dict):
    with open(cfg.RAW_OUTPUT, "a") as out:
        for batch_id in batch_ids:
            for result in client.messages.batches.results(batch_id):
                article_id = result.custom_id
                if result.result.type == "succeeded":
                    raw_text = result.result.message.content[0].text
                    out.write(json.dumps({"article_id": article_id, "raw": raw_text}) + "\n")
                    state[article_id]["status"] = "complete"
                else:
                    state[article_id]["status"] = "failed"
    save_state(state)


# ---------------------------------------------------------------------------
# 5. Parsing & validation
# ---------------------------------------------------------------------------

# extract the outermost JSON object from model output, skipping any reasoning preamble
def _extract_json(raw_text: str) -> dict | None:
    """
    Returns None if no JSON is found or if the JSON is malformed.
    """
    start = raw_text.find("{")
    if start == -1:
        return None
    try:
        depth, end = 0, start
        for i, ch in enumerate(raw_text[start:], start):
            if ch == "{": depth += 1
            elif ch == "}": depth -= 1
            if depth == 0:
                end = i
                break
        return json.loads(raw_text[start:end + 1])
    except json.JSONDecodeError:
        return None

# extract the reasoning preamble that precedes the JSON, for use in semantic clustering
# currently not needed since the model is prompted to output only the JSON
def _extract_reasoning(raw_text: str) -> str | None:
    """
    Uses rfind to locate the last opening brace, so everything before it is reasoning.
    Returns None if the output starts directly with JSON.
    """
    start = raw_text.rfind("{")
    if start == -1:
        return None
    reasoning = raw_text[:start].strip()
    return reasoning if reasoning else None

# parse the raw JSONL file and validate each record against the output schema
def parse_and_validate(raw_output_path: str) -> tuple[list, list, list]:
    """
    Returns (valid_records, malformed_ids, invalid_records).
    valid_records include a 'reasoning' field for semantic clustering.
    malformed_ids: article ids where JSON could not be extracted.
    invalid_records: dicts with 'id' and 'error' where schema validation failed.
    """
    records, malformed, invalid = [], [], []
    with open(raw_output_path) as f:
        for line in f:
            entry  = json.loads(line)
            parsed = _extract_json(entry["raw"])
            if parsed is None:
                malformed.append(entry["article_id"])
                continue
            try:
                jsonschema.validate(parsed, cfg.OUTPUT_SCHEMA)
                records.append({
                    "id": entry["article_id"],
                    "reasoning":  _extract_reasoning(entry["raw"]),
                    **parsed,
                })
            except jsonschema.ValidationError as e:
                invalid.append({"id": entry["article_id"], "error": e.message})

    # deduplicate by id — retries append to the same file so the last occurrence wins
    seen = {}
    for rec in records:
        seen[rec["id"]] = rec
    records = list(seen.values())
    malformed = list(set(malformed))
    invalid_seen = {r["id"]: r for r in invalid}
    invalid = list(invalid_seen.values())

    return records, malformed, invalid

# ---------------------------------------------------------------------------
# 6. Derive additional fields via lookup tables
# ---------------------------------------------------------------------------

# derive actor category from specific_type
def derive_actor_category(specific_type: str | None) -> str | None:
    if specific_type is None:
        return None
    return cfg.SUBTYPE_TO_SECTOR.get(specific_type)

# derive abstract force category from specific_type
def derive_force_category(specific_type: str | None) -> str | None:
    if specific_type is None:
        return None
    return cfg.MECHANISM_TO_CATEGORY.get(specific_type)

# derive ideological axis or legitimation mode from specific_type
def derive_ideology_category(specific_type: str | None) -> str | None:
    if specific_type is None:
        return None
    return cfg.TRADITION_TO_AXIS.get(specific_type)

# ---------------------------------------------------------------------------
# 7. Retry
# ---------------------------------------------------------------------------

# re-dispatch any articles that failed or produced invalid output, up to cfg.MAX_RETRIES times
def retry_failed(client: anthropic.Anthropic, df: pd.DataFrame,
                 state: dict, malformed: list, invalid: list) -> list[str]:
    failed_ids = list(set(
        [aid for aid, v in state.items() if v["status"] == "failed"]
        + malformed
        + [r["id"] for r in invalid]
    ))
    retry_ids = [
        aid for aid in failed_ids
        if state.get(aid, {}).get("retry_count", 0) < cfg.MAX_RETRIES
    ]
    if not retry_ids:
        print("Nothing to retry.")
        return []
    for aid in retry_ids:
        state[aid]["status"]      = "pending"
        state[aid]["retry_count"] = state.get(aid, {}).get("retry_count", 0) + 1
    save_state(state)
    retry_df = df[df["id"].isin(retry_ids)]
    return dispatch_all(client, retry_df, state)


# ---------------------------------------------------------------------------
# 8. Sanity checks
# ---------------------------------------------------------------------------

# print distribution checks on narrative presence, entity types, empty slots, and uncertainty flags
def print_sanity_checks(records: list):
    n = len(records)
    narrative_present = [rec for rec in records if rec.get("narrative_present", True)]
    n_present = len(narrative_present)

    print("=== Narrative presence ===")
    print(f"  Narrative present:  {n_present:,} ({n_present/n:.1%})")
    print(f"  No narrative:       {n - n_present:,} ({(n - n_present)/n:.1%})")

    # Remaining checks only on articles with narratives
    specific_type_by_slot = {s: [] for s in cfg.SLOTS}
    filler_type_by_slot  = {s: [] for s in cfg.SLOTS}
    uncertain_counts     = []
    empty_slots          = []

    for rec in narrative_present:
        n_uncertain = 0
        for slot in cfg.SLOTS:
            fillers = rec["actantial_slots"].get(slot, [])
            if not fillers:
                empty_slots.append(slot)
            for f in fillers:
                specific_type_by_slot[slot].append(f.get("specific_type"))
                filler_type_by_slot[slot].append(f["filler_type"])
                if f.get("uncertain"):
                    n_uncertain += 1
        uncertain_counts.append(n_uncertain)

    if not n_present:
        return

    print("\n=== Entity type distribution by slot ===")
    for slot in cfg.SLOTS:
        c     = collections.Counter(filler_type_by_slot[slot])
        total = sum(c.values())
        if total:
            print(f"  {slot:10s}  actor={c.get('actor',0)/total:.0%}  "
                  f"abstract_force={c.get('abstract_force',0)/total:.0%}  "
                  f"ideology={c.get('ideology',0)/total:.0%}  (n={total:,})")

    print("\n=== Empty slot frequency ===")
    for slot, count in collections.Counter(empty_slots).most_common():
        print(f"  {slot:10s}: {count:,} ({count/n_present:.1%})")

    print("\n=== Uncertain flag usage ===")
    has_uncertain = sum(1 for c in uncertain_counts if c > 0)
    print(f"  Articles with ≥1 uncertain annotation : {has_uncertain:,} ({has_uncertain/n_present:.1%})")
    print(f"  Mean uncertain annotations per article: {sum(uncertain_counts)/n_present:.2f}")


# ---------------------------------------------------------------------------
# 9. Save reasoning text (for clustering)
# ---------------------------------------------------------------------------

# save the model's chain-of-thought reasoning text to CSV for use in semantic embedding at the clustering stage
# currently not needed since the model is prompted to output only the JSON
def save_reasoning(records: list, meta_df: pd.DataFrame):
    """
    The reasoning contains the full slot-by-slot chain of thought, which is richer
    than dominant_narrative alone for embedding at the clustering stage.
    """
    meta = meta_df[["id"]]
    rows = [
        {"id": rec["id"], "reasoning": rec.get("reasoning")}
        for rec in records
        if rec.get("narrative_present", True) and rec.get("reasoning")
    ]
    meta = meta.copy()
    meta["id"] = meta["id"].astype(str)
    df = pd.DataFrame(rows).merge(meta, on="id", how="left")
    df.to_csv(cfg.REASONING_CSV, index=False)
    print(f"Saved {len(df):,} reasoning texts -> {cfg.REASONING_CSV}")


# ---------------------------------------------------------------------------
# 10. Flatten to long-format CSV
# ---------------------------------------------------------------------------

# flatten extraction results to one row per slot filler and join with article metadata
def to_long_format(records: list, meta_df: pd.DataFrame) -> pd.DataFrame:
    meta = meta_df[["id", "pubtime", "medium_code", "medium_name", "char_count"]]
    rows = []
    for rec in records:
        if not rec.get("narrative_present", True):
            continue
        base = {
            "id":                 rec["id"],
            "narrative_present":  True,
            "dominant_narrative": rec["dominant_narrative"],
        }
        for slot in cfg.SLOTS:
            for i, f in enumerate(rec["actantial_slots"].get(slot, [])):
                specific_type = f.get("specific_type") or None
                ftype         = f["filler_type"]
                rows.append({
                    **base,
                    "slot":             slot,
                    "filler_rank":      i,
                    "label":            f["label"],
                    "filler_type":      ftype,
                    "specific_type":    specific_type,
                    "category":         derive_actor_category(specific_type) if ftype == "actor"          else
                                        derive_force_category(specific_type) if ftype == "abstract_force" else
                                        derive_ideology_category(specific_type) if ftype == "ideology"    else None,
                    "domain":           f.get("domain"),
                    "uncertain":        f.get("uncertain", False),
                    "uncertainty_note": f.get("uncertainty_note") or None,
                })
    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype(str)
    meta = meta.copy()
    meta["id"] = meta["id"].astype(str)
    return df.merge(meta, on="id", how="left")