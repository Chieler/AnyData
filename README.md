# AnyData

A lightweight ingest + audit tool for HuggingFace image datasets. Point it at
a dataset, and it streams the images, embeds them, and produces a compact
JSON report of everything worth flagging: corrupt images, near-duplicates,
label imbalance across splits, image-size outliers, and visually confused
classes (via k-NN over embeddings). The report is sized to drop straight
into an LLM prompt for further analysis.

## How it works

`DataViewer(link, db_link).run()` is the single entry point:

- **Empty DB** (or `force_repopulate=True`): streams the dataset from
  HuggingFace, decodes and hashes each image, embeds it with
  [DINOv2](https://huggingface.co/facebook/dinov2-base), and writes
  everything to a local SQLite database.
- **Non-empty DB**: skips ingest entirely and just summarizes what's
  already stored — no network access, no model load.

During ingest, corrupt images (fail to decode) and duplicates (matching
perceptual hash) are detected and excluded from the embedding/label stats,
but their counts are tracked and reported.

```python
from DataViewer import DataViewer

dv = DataViewer(link="some-org/some-dataset", db_link="dataset.db")
summary = dv.run(primary_label_field="label", k=10)

print(summary.to_prompt())
```

## What's in the summary

`DatasetAuditSummary` bundles:

- `n_images`, `n_corrupt`, `corrupt_fraction`
- `n_duplicates`, `duplicate_fraction`
- `label_summary` — per-split class distributions (Gini, normalized
  entropy, imbalance ratio) plus a cross-split comparison flagging classes
  that are missing from or drifted between splits
- `size_summary` — width/height/aspect-ratio quartiles per split
- `knn_confusion` — per-class neighbor agreement and the top confused
  class pairs, computed over DINOv2 embeddings and normalized by class size

Call `.to_prompt()` on the result to get a compact JSON string ready to
paste into an LLM prompt.

## Requirements

Ingest requires `torch`, `transformers`, `datasets`, `imagehash`, and a
`facebook/dinov2-base` checkpoint (downloaded automatically on first use).
The summary-only path (non-empty DB) only needs `sqlite3`, `pandas`, and
`numpy` from the standard/scientific stack — these heavy deps are imported
lazily so you can read back a report without loading a model.

Embedding currently targets Apple Silicon (`mps` device) — see
`DataViewer._load_models` / `embed_batch` in `DataViewer.py` if you need to
target CUDA or CPU instead.

## Status

Early / single-file prototype. `main.py` is currently empty — the library
is used directly via `DataViewer`.
