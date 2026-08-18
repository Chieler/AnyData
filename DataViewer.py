"""
DataViewer -- ingest + audit for a HuggingFace image dataset.

Entry point: DataViewer(link, db_link).run()
  - empty DB (or force_repopulate=True): downloads + ingests, then summarizes
  - non-empty DB: summarizes only, no download, no model load

Heavy deps (torch, transformers, datasets, imagehash, sklearn) are imported
lazily inside the methods that need them -- the summary-only path only
touches sqlite3/pandas/numpy/json/math, on purpose, since it shouldn't
need a GPU model loaded just to read numbers back out of SQLite.
"""

import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DatasetAuditSummary:
    """Everything the LLM needs for facet discovery / gap analysis, in one object."""
    n_images: int
    n_corrupt: int
    corrupt_fraction: float
    n_duplicates: Optional[int]        # None if the DB predates ingest_stats tracking
    duplicate_fraction: Optional[float]
    label_summary: dict                # cross_split_report output
    size_summary: dict                 # width/height/aspect quartiles, per split
    knn_confusion: dict                # per-class agreement + top confusions, size-normalized

    def to_prompt(self) -> str:
        """Compact JSON block, ready to drop into an LLM prompt."""
        return json.dumps({
            "dataset_stats": {
                "n_images": self.n_images,
                "n_corrupt": self.n_corrupt,
                "corrupt_fraction": round(self.corrupt_fraction, 4),
                "n_duplicates": self.n_duplicates,
                "duplicate_fraction": (round(self.duplicate_fraction, 4)
                                        if self.duplicate_fraction is not None else None),
            },
            "label_distribution": self.label_summary,
            "image_size_distribution": self.size_summary,
            "visual_confusion": self.knn_confusion,
        }, indent=2)


class DataViewer:

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS img_info (
        row_id     INTEGER PRIMARY KEY,
        split      TEXT NOT NULL,
        metadata   TEXT,
        width      INTEGER,
        height     INTEGER,
        aspect     REAL,
        hash       TEXT,
        embedding  BLOB,
        is_corrupt INTEGER NOT NULL DEFAULT 0
    );
    """
    STATS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS ingest_stats (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """
    WRITE_CMD = """INSERT INTO img_info
        (row_id, split, metadata, width, height, aspect, hash, embedding, is_corrupt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(row_id) DO NOTHING"""

    def __init__(self, link, db_link):
        self.link = link
        self.db_link = db_link
        self.conn = sqlite3.connect(db_link)
        self.conn.execute(self.SCHEMA)
        self.conn.execute(self.STATS_SCHEMA)
        self.conn.commit()

        self.dataset = None
        self.processor = None
        self.model = None

        self.hash_set = set()
        self.duplicate = 0
        self.counts = defaultdict(lambda: defaultdict(Counter))
        self.corrupt = 0

    # ---------------------------------------------------------------- entry point
    def run(self, force_repopulate: bool = False,
            primary_label_field: Optional[str] = None,
            k: int = 10) -> DatasetAuditSummary:
        """
        The single call site. Empty DB -> download + ingest, then summarize.
        Non-empty DB -> summarize only, no network, no model load.
        """
        if force_repopulate or self._is_empty():
            self._ingest()
        return self.build_summary(primary_label_field=primary_label_field, k=k)

    def _is_empty(self) -> bool:
        n = self.conn.execute("SELECT COUNT(*) FROM img_info").fetchone()[0]
        return n == 0

    # ---------------------------------------------------------------- ingest-stats persistence
    # Dropped duplicates never get a row in img_info, so that count is only
    # recoverable if we persist it at ingest time -- unlike n_corrupt, which
    # is always recoverable from the is_corrupt column regardless of session.
    def _save_ingest_stat(self, key, value):
        self.conn.execute(
            "INSERT INTO ingest_stats (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def _load_ingest_stat(self, key, default=None):
        row = self.conn.execute("SELECT value FROM ingest_stats WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    # ---------------------------------------------------------------- ingest (heavy deps lazily imported)
    def _load_models(self):
        if self.model is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.model = AutoModel.from_pretrained("facebook/dinov2-base").half().to("mps").eval()
            self._torch = torch

    def embed_batch(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to("mps")
        inputs = {k: (v.half() if v.is_floating_point() else v) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self.model(**inputs).last_hidden_state[:, 0]
        return out.float().cpu().numpy()

    def flush(self, batch_rows, batch_imgs, conn):
        if not batch_imgs:
            return
        vecs = self.embed_batch(batch_imgs)
        conn.executemany(self.WRITE_CMD, [
            (r["row_id"], r["split"], json.dumps(r["metadata"]),
             r["width"], r["height"], r["width"] / r["height"],
             r["hash"], vec.astype(np.float32).tobytes(), 0)
            for r, vec in zip(batch_rows, vecs)
        ])
        conn.commit()

    def _ingest(self):
        import imagehash
        from datasets import load_dataset

        self._load_models()
        self.dataset = load_dataset(self.link, streaming=True)

        BATCH_SIZE = 64
        global_id = 0
        for split_name, split_ds in self.dataset.items():
            batch_imgs, batch_rows = [], []
            col_names = [c for c in split_ds.features.keys() if c != "image"]

            for example in split_ds:
                try:
                    img = example["image"].convert("RGB")
                    img.load()                      # forces full decode; catches truncation
                except Exception:
                    self.corrupt += 1
                    self.conn.execute(self.WRITE_CMD, (global_id, split_name, "{}",
                                                        0, 0, 0, None, None, 1))
                    global_id += 1
                    continue

                meta = {}
                for col in col_names:
                    feat = split_ds.features[col]
                    value = feat.int2str(example[col]) if feat.__class__.__name__ == "ClassLabel" else example[col]
                    self.counts[split_name][col][str(value)] += 1
                    meta[col] = value

                img_hash = str(imagehash.dhash(img))
                if img_hash in self.hash_set:
                    self.duplicate += 1
                    global_id += 1
                    continue
                self.hash_set.add(img_hash)

                w, h = img.size
                batch_rows.append({"row_id": global_id, "split": split_name,
                                    "metadata": meta, "width": w, "height": h, "hash": img_hash})
                batch_imgs.append(img)
                global_id += 1

                if len(batch_imgs) == BATCH_SIZE:
                    self.flush(batch_rows, batch_imgs, self.conn)
                    print(f"{split_name}: {global_id} seen, {self.duplicate} dup, {self.corrupt} corrupt")
                    batch_rows, batch_imgs = [], []

            self.flush(batch_rows, batch_imgs, self.conn)

        # persist -- these are only knowable during ingest, not recoverable from the table alone
        self._save_ingest_stat("n_duplicates_dropped", self.duplicate)
        self._save_ingest_stat("n_corrupt_during_ingest", self.corrupt)

    # ---------------------------------------------------------------- loading
    def load_dataset_from_db(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM img_info", self.conn)

    def load_embeddings(self, df: Optional[pd.DataFrame] = None, dim: int = 768):
        """Returns (row_ids, metas, matrix) for non-corrupt rows with an embedding."""
        if df is None:
            df = self.load_dataset_from_db()
        usable = df[(df["is_corrupt"] == 0) & (df["embedding"].notna())]
        row_ids = usable["row_id"].to_numpy()
        metas = [json.loads(m) for m in usable["metadata"]]
        matrix = np.vstack([np.frombuffer(e, dtype=np.float32) for e in usable["embedding"]])
        assert matrix.shape[1] == dim, f"expected dim {dim}, got {matrix.shape[1]}"
        return row_ids, metas, matrix

    # ---------------------------------------------------------------- distribution stats
    def gini(self, counts):
        vals = sorted(int(c) for c in counts)
        n = len(vals)
        if n == 0:
            return 0.0
        total = sum(vals)
        if total == 0:
            return 0.0
        weighted = sum((2 * i - n - 1) * v for i, v in enumerate(vals, start=1))
        return weighted / (n * total)

    def normalized_entropy(self, counts):
        vals = [int(c) for c in counts if c > 0]
        n = len(vals)
        if n <= 1:
            return 1.0
        total = sum(vals)
        h = -sum((v / total) * math.log(v / total) for v in vals)
        return h / math.log(n)

    def summarize_counter(self, counter, small_cardinality=30, head=15, tail=10,
                          small_class_threshold=10):
        n_classes = len(counter)
        if n_classes == 0:
            return {"n_classes": 0, "n_images": 0}
        mc = counter.most_common()
        vals = [c for _, c in mc]
        total = sum(vals)
        base = {
            "n_classes": n_classes,
            "n_images": total,
            "imbalance_ratio": round(max(vals) / min(vals), 2) if min(vals) > 0 else None,
            "gini": round(self.gini(vals), 3),
            "normalized_entropy": round(self.normalized_entropy(vals), 3),
            "n_classes_below_threshold": sum(1 for v in vals if v < small_class_threshold),
            "small_class_threshold": small_class_threshold,
        }
        if n_classes <= small_cardinality:
            base["truncated"] = False
            base["classes"] = dict(mc)
        else:
            base["truncated"] = True
            base["head"] = dict(mc[:head])
            base["tail"] = dict(mc[-tail:])
            base["n_omitted"] = max(0, n_classes - head - tail)
        return base

    def summarize_counts(self, counts, **kwargs):
        return {
            split_name: {col: self.summarize_counter(counter, **kwargs) for col, counter in columns.items()}
            for split_name, columns in counts.items()
        }

    def cross_split_report(self, counts, **kwargs):
        summary = self.summarize_counts(counts, **kwargs)
        all_cols = {col for cols in counts.values() for col in cols}
        comparisons = {}
        for col in all_cols:
            splits_with_col = {s: cols[col] for s, cols in counts.items() if col in cols}
            if len(splits_with_col) < 2:
                continue
            props = {}
            for split_name, counter in splits_with_col.items():
                total = sum(counter.values()) or 1
                props[split_name] = {k: v / total for k, v in counter.items()}
            all_values = set().union(*(set(c) for c in splits_with_col.values()))
            missing, drifted = {}, {}
            for value in all_values:
                present_in = [s for s in splits_with_col if value in splits_with_col[s]]
                absent_in = [s for s in splits_with_col if value not in splits_with_col[s]]
                if absent_in:
                    missing[value] = {"present_in": present_in, "absent_in": absent_in}
                elif len(props) >= 2:
                    ps = [props[s].get(value, 0.0) for s in props]
                    if max(ps) - min(ps) > 0.05:
                        drifted[value] = {s: round(props[s].get(value, 0.0), 4) for s in props}
            comparisons[col] = {
                "splits": list(splits_with_col),
                "classes_missing_from_some_split": missing,
                "classes_with_proportion_drift": drifted,
            }
        return {"per_split": summary, "cross_split": comparisons}

    def counts_from_db(self, df: pd.DataFrame):
        """
        Rebuild split -> column -> Counter from the DB (the 'already
        populated, just summarize' path). Parses each row's metadata JSON
        once -- not once per column, which was the bug in the original.
        """
        counts = defaultdict(lambda: defaultdict(Counter))
        clean = df[df["is_corrupt"] == 0]
        for split_name, meta_json in zip(clean["split"], clean["metadata"]):
            meta = json.loads(meta_json)
            for col, value in meta.items():
                counts[split_name][col][str(value)] += 1
        return counts

    # ---------------------------------------------------------------- size distribution
    def size_summary(self, df: pd.DataFrame):
        """
        Width/height/aspect quartiles, per split, corrupt rows excluded.
        Corrupt rows store width=height=aspect=0 placeholders -- including
        them would silently skew every quartile toward zero.
        Reuses the aspect column already computed at ingest time rather
        than recomputing width/height division (which is also where the
        original 0/0 -> NaN risk lived).
        """
        clean = df[df["is_corrupt"] == 0]
        out = {}
        for split_name, sub in clean.groupby("split"):
            widths = sub["width"].to_numpy()
            heights = sub["height"].to_numpy()
            aspect = sub["aspect"].to_numpy()
            split_summary = {}
            for name, arr in [("width", widths), ("height", heights), ("aspect", aspect)]:
                q = np.percentile(arr, [0, 25, 50, 75, 100])
                split_summary[name] = {
                    "min": round(float(q[0]), 2), "q1": round(float(q[1]), 2),
                    "median": round(float(q[2]), 2), "q3": round(float(q[3]), 2),
                    "max": round(float(q[4]), 2),
                }
            out[split_name] = split_summary
        return out

    # ---------------------------------------------------------------- k-NN confusion (normalized, directional)
    def normalize(self, matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def knn_confusion(self, matrix, labels, k=10, top_n=15):
        """
        Rates are normalized by class size (count / (n_class * k)), so a
        large class doesn't dominate the ranking just by being large.
        Direction is preserved -- (A -> B) and (B -> A) reported
        separately, since "A absorbed into B" and "B absorbed into A" are
        different findings even when they co-occur.
        """
        labels = np.asarray(labels)
        normed = self.normalize(matrix)
        sim = normed @ normed.T
        np.fill_diagonal(sim, -1.0)
        k = min(k, sim.shape[0] - 1)
        knn_idx = np.argsort(-sim, axis=1)[:, :k]

        class_counts = Counter(labels.tolist())
        classes = sorted(class_counts)

        agreement = {}
        for c in classes:
            idxs = np.where(labels == c)[0]
            own_frac = (labels[knn_idx[idxs]] == c).mean(axis=1)
            agreement[c] = round(float(own_frac.mean()), 3)

        directional = Counter()
        for i in range(len(labels)):
            own = labels[i]
            for lab in labels[knn_idx[i]]:
                if lab != own:
                    directional[(own, lab)] += 1

        confusions = []
        for (a, b), cnt in directional.items():
            n_a = class_counts[a]
            rate = cnt / (n_a * k)   # size-normalized, not a raw count
            confusions.append({"from": a, "to": b, "rate": round(rate, 4), "raw_count": int(cnt)})
        confusions.sort(key=lambda d: -d["rate"])

        return {
            "k": k,
            "per_class_neighbor_agreement": dict(sorted(agreement.items(), key=lambda kv: kv[1])),
            "top_confusions": confusions[:top_n],
        }

    # ---------------------------------------------------------------- optional: clustering (not in default summary)
    # Kept available, but deliberately excluded from build_summary(): both
    # PCA-component-count and HDBSCAN's min_cluster_size proved to swing
    # results significantly on real embeddings, while k-NN confusion needs
    # no such hyperparameter. Use these directly if you want to explore,
    # not as a source of automatic report findings.
    def reduce_dim(self, matrix, n_components=50):
        from sklearn.decomposition import PCA
        n = min(n_components, matrix.shape[0], matrix.shape[1])
        pca = PCA(n_components=n, random_state=0)
        reduced = pca.fit_transform(matrix)
        return reduced, float(pca.explained_variance_ratio_.sum())

    def cluster(self, reduced, min_cluster_size=None, n_images=None):
        from sklearn.cluster import HDBSCAN
        if min_cluster_size is None:
            n = n_images or reduced.shape[0]
            min_cluster_size = max(5, int(n * 0.005))
        model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        label = model.fit_predict(reduced)
        return label, min_cluster_size

    # ---------------------------------------------------------------- the assembly point
    def build_summary(self, primary_label_field: Optional[str] = None, k: int = 10) -> DatasetAuditSummary:
        df = self.load_dataset_from_db()
        total = len(df)
        n_corrupt = int(df["is_corrupt"].sum())

        n_duplicates = self._load_ingest_stat("n_duplicates_dropped")
        duplicate_fraction = (n_duplicates / total) if (n_duplicates is not None and total) else None

        counts = self.counts_from_db(df)
        label_summary = self.cross_split_report(counts)
        size_summary = self.size_summary(df)

        row_ids, metas, matrix = self.load_embeddings(df)

        if primary_label_field is None:
            keys = set()
            for m in metas:
                keys.update(m.keys())
            if len(keys) == 1:
                primary_label_field = next(iter(keys))
            else:
                raise ValueError(
                    f"Multiple metadata fields found {sorted(keys)} -- "
                    f"pass primary_label_field explicitly to run knn_confusion()."
                )
        labels = np.array([m[primary_label_field] for m in metas])
        knn_result = self.knn_confusion(matrix, labels, k=k)

        return DatasetAuditSummary(
            n_images=total,
            n_corrupt=n_corrupt,
            corrupt_fraction=round(n_corrupt / total, 4) if total else 0.0,
            n_duplicates=n_duplicates,
            duplicate_fraction=duplicate_fraction,
            label_summary=label_summary,
            size_summary=size_summary,
            knn_confusion=knn_result,
        )