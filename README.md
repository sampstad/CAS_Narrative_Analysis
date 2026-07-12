# Swiss German News · Narrative Analysis

A pipeline and interactive explorer for the **narrative structure** of Swiss German
news coverage. Each article is analysed into a Greimas-style *actantial* structure
— who does what to whom, and why — and articles whose storylines converge are
grouped into **narrative clusters**. The same structured representation powers a
searchable **entity vocabulary** for tracing specific actors, forces, and
ideological frames across the whole corpus.

The repository has two parts:

- **A processing pipeline** (`01`–`07` notebooks + `lib/` + `config/`) that turns a
  raw article dataset into extracted narratives, clusters, labels, and the compact
  artifacts the app reads.
- **A Streamlit app** (`app/`) that visualises the results: an interactive cluster
  map, full narrative cards with publisher/time filters, side-by-side comparison,
  and entity analysis.

## The app

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

The app reads precomputed artifacts from `data/viz/` (override the location with the
`APP_DATA_DIR` environment variable). If those artifacts are missing, run the
pipeline first (see below).

Pages: **Welcome** (overview), **Explore / Inspect / Compare narratives**,
**Explore / Compare entities**, and a **Glossary** defining every role and entity
type. Definitions of the annotation scheme live on the Glossary page.

## The pipeline

Run the notebooks in order. Each stage writes intermediate files into `data/`.

| Notebook | Purpose |
| --- | --- |
| `01_load.ipynb` | Load and filter the raw article dataset |
| `02_preprocess.ipynb` | Clean text; compute sentiment, emotion, and readability metrics |
| `03_extract.ipynb` | LLM extraction of actantial narrative structure per article |
| `04_normalize.ipynb` | Normalise and canonicalise entity labels |
| `05_cluster.ipynb` | Embed narratives and cluster them at macro / meso / micro levels |
| `06_label.ipynb` | LLM-label each cluster (title + description) |
| `07_precompute.ipynb` | Emit the tidy `data/viz/` tables the app consumes |

Pipeline dependencies (GPU-oriented) are in `requirements.txt`; the leaner
app-only set is in `app/requirements.txt`. A conda environment is provided in
`environment.yml`.

## Data & artifacts

Raw and intermediate pipeline data under `data/` is **not** committed (large and
possibly licensed). The small precomputed artifacts the app needs are committed
under `data/viz/`, with one exception:

- **`data/viz/entity_embeddings.npy` (~389 MB)** is not committed because it exceeds
  GitHub's 100 MB file limit. Regenerate it by running `07_precompute.ipynb`.
  Until it is present, semantic entity search is disabled; the rest of the app
  (cluster map, narrative cards, entity vocabulary search, comparison) works from
  the committed artifacts.

## Repository layout

```
01_load … 07_precompute.ipynb   Pipeline notebooks
app/                            Streamlit app (views, components, data layer)
lib/                            Pipeline logic (one module per stage)
config/                         Stage configuration
prompts/                        LLM prompt templates
data/viz/                       Precomputed artifacts read by the app (committed)
report/                         Project report (docx/pdf), figures, and analysis script
```

## Report

The written project report lives in `report/`:

- `report/Narrative_Analysis_Report.docx` / `.pdf` — the report.
- `report/analysis.py` — regenerates every figure and statistic in `report/img/` from
  the data (`python report/analysis.py`).
- `report/build_report.py` — assembles the `.docx` from the figures and `report/stats.json`.

## License

Released under the MIT License — see [LICENSE](LICENSE).
