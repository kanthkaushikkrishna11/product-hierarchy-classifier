# src/scoring.py
# -*- coding: utf-8 -*-
"""
Confidence scoring for product→(group, variant) assignments.

Signals:
- Text embedding cohesion (variant/group centroids) — SBERT preferred, TF-IDF fallback
- Axis presence and within-variant numeric consistency
- Cohort size
- Screen-size sanity penalty

API:
  scorer = ConfidenceScorer(use_sbert=True)
  scorer.fit(df, assignments_df, variant_records, calibration_labels=None)
  out = scorer.score_assignments(df, assignments_df, return_features=False)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Optional DL embeddings
_SBERT_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except Exception:  # pragma: no cover
    _SBERT_AVAILABLE = False

# Optional sklearn stack
_SKLEARN_AVAILABLE = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize as sk_normalize
except Exception:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
    TfidfVectorizer = object  # type: ignore
    LogisticRegression = object  # type: ignore
    def sk_normalize(x, **kwargs):  # type: ignore
        return x

from .normalize import in_valid_screen_range


# ----------------------------- small utilities -----------------------------

def _as_str(x):
    return "" if x is None else str(x)


def _gather_text(row: pd.Series) -> str:
    """Primary text for embeddings: prefer ml_text, else name."""
    return _as_str(row.get("ml_text") or row.get("name"))


def _cosine_dense(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)


def _cosine(a, b) -> float:
    """Cosine for dense arrays or sparse CSR rows."""
    try:
        return _cosine_dense(a, b)
    except Exception:
        num = (a @ b.T).A[0, 0]
        an = math.sqrt((a.multiply(a)).sum())
        bn = math.sqrt((b.multiply(b)).sum())
        return float(num / ((an + 1e-12) * (bn + 1e-12)))


@dataclass
class _CohortStats:
    centroid: Any                   # dense np.ndarray or sparse row
    size: int
    numeric_axes_stats: Dict[str, Dict[str, float]]  # e.g., {"ram_gb": {"mode_frac": .95, "std": 0, "n": 10}}


def _numeric_consistency(values: List[Optional[float]], step: float) -> Dict[str, float]:
    """Mode fraction and std for a numeric list; ignores None. Quantizes by step."""
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"mode_frac": 0.0, "std": float("nan"), "n": 0}
    q = [round(v / step) * step for v in xs]
    vals, counts = np.unique(q, return_counts=True)
    return {"mode_frac": float(np.max(counts)) / float(len(q)), "std": float(np.std(q)), "n": len(q)}


# --------------------------- embedding backend ----------------------------

class _Embedder:
    """SBERT if available; else TF-IDF (if sklearn); else identity vectors."""

    def __init__(self, use_sbert=True, tfidf_min_df=3, tfidf_max_df=0.95, tfidf_ngram=(1, 2)):
        self.use_sbert = bool(use_sbert)
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_ngram = tfidf_ngram
        self._sbert = None
        self._tfidf = None

    def _maybe_load_sbert(self):
        if not self.use_sbert or not _SBERT_AVAILABLE:
            return None
        if self._sbert is None:
            try:
                self._sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                self._sbert = None
        return self._sbert

    def fit_transform(self, texts: List[str]):
        model = self._maybe_load_sbert()
        if model is not None:
            try:
                vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return np.asarray(vecs, dtype=np.float32)
            except Exception:
                pass

        if not _SKLEARN_AVAILABLE:
            n = len(texts)
            return np.eye(n, dtype=np.float32)

        self._tfidf = TfidfVectorizer(
            min_df=self.tfidf_min_df,
            max_df=self.tfidf_max_df,
            ngram_range=self.tfidf_ngram,
            sublinear_tf=True,
        )
        X = self._tfidf.fit_transform(texts)
        return sk_normalize(X, norm="l2", copy=False)

    def transform(self, texts: List[str]):
        model = self._maybe_load_sbert()
        if model is not None:
            try:
                vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return np.asarray(vecs, dtype=np.float32)
            except Exception:
                pass

        if not _SKLEARN_AVAILABLE:
            n = len(texts)
            return np.eye(n, dtype=np.float32)

        if self._tfidf is None:
            # Should not happen in normal flow; return zeros
            n = len(texts)
            return np.zeros((n, 1), dtype=np.float32)
        X = self._tfidf.transform(texts)
        return sk_normalize(X, norm="l2", copy=False)


# ------------------------------ scorer class ------------------------------

class ConfidenceScorer:
    """
    Compute confidence and evidence for assignments.

    If no calibrator is fit, uses deterministic weighted features → logistic.
    If labels are passed to fit(..., calibration_labels=...), trains a
    LogisticRegression calibrator.
    """

    def __init__(
        self,
        use_sbert=True,
        w_variant_sim=0.45,
        w_group_sim=0.20,
        w_axes_presence=0.15,
        w_axes_consistency=0.10,
        w_variant_size=0.06,
        w_group_size=0.04,
        penalty_screen_invalid=0.10,
        seed=42,
    ):
        self.embedder = _Embedder(use_sbert=use_sbert)
        self.w_variant_sim = w_variant_sim
        self.w_group_sim = w_group_sim
        self.w_axes_presence = w_axes_presence
        self.w_axes_consistency = w_axes_consistency
        self.w_variant_size = w_variant_size
        self.w_group_size = w_group_size
        self.penalty_screen_invalid = penalty_screen_invalid
        self.seed = seed

        self._X = None
        self._group_stats: Dict[str, _CohortStats] = {}
        self._variant_stats: Dict[str, _CohortStats] = {}
        self._calibrator: Optional[LogisticRegression] = None  # type: ignore
        self._feature_names = [
            "variant_sim", "group_sim", "axes_presence", "axes_consistency",
            "variant_size_norm", "group_size_norm", "screen_invalid_flag"
        ]

    # -------------------------------- fit ---------------------------------

    def fit(self, df: pd.DataFrame, assignments_df: pd.DataFrame, variant_records: List[Dict[str, Any]], calibration_labels=None):
        """
        Prepare embeddings, centroids, per-variant numeric stats, and optional calibrator.

        df               : main dataframe (must include 'product_id', 'name'/'ml_text', 'axes', 'sub_category')
        assignments_df   : columns [product_id, group_id, variant_id]
        variant_records  : output of build_variants (list of dicts)
        calibration_labels : optional list/array aligned with assignments_df rows (0/1 or floats)
        """
        # 1) Embed all rows (deterministic)
        texts = [_gather_text(row) for _, row in df.iterrows()]
        self._X = self.embedder.fit_transform(texts)

        # 2) Row index by product_id
        idx_by_pid = {str(pid): i for i, pid in enumerate(df["product_id"].astype(str).values)}

        # 3) Build cohorts (indices per group/variant)
        rows_by_group: Dict[str, List[int]] = {}
        rows_by_variant: Dict[str, List[int]] = {}
        for _, r in assignments_df.iterrows():
            i = idx_by_pid.get(str(r["product_id"]))
            if i is None:
                continue
            rows_by_group.setdefault(str(r["group_id"]), []).append(i)
            rows_by_variant.setdefault(str(r["variant_id"]), []).append(i)

        # 4) Centroids
        def _centroid(row_indices: List[int]):
            if hasattr(self._X, "toarray"):  # sparse
                sub = self._X[row_indices]
                c = sk_normalize(sub.mean(axis=0), norm="l2", copy=False)
                return c
            sub = self._X[row_indices]
            c = np.mean(sub, axis=0, dtype=np.float32)
            n = np.linalg.norm(c) + 1e-12
            return c / n

        # 5) Fill group stats
        for gid, rows in rows_by_group.items():
            self._group_stats[gid] = _CohortStats(centroid=_centroid(rows), size=len(rows), numeric_axes_stats={})

        # 6) Fill variant stats (numeric consistency for RAM/Storage/Screen)
        def _axis_vals(rows: List[int], key: str, field: str) -> List[Optional[float]]:
            vs = []
            for i in rows:
                axes = df.iloc[i].get("axes") or {}
                ax = axes.get(key) or {}
                v = ax.get(field)
                if v is None:
                    vs.append(None)
                else:
                    try:
                        vs.append(float(v))
                    except Exception:
                        vs.append(None)
            return vs

        for vr in variant_records:
            vid = vr["variant_id"]
            rows = rows_by_variant.get(vid, [])
            if not rows:
                continue
            stats = {
                "ram_gb": _numeric_consistency(_axis_vals(rows, "config", "ram_gb"), step=8.0),
                "storage_gb": _numeric_consistency(_axis_vals(rows, "config", "storage_gb"), step=128.0),
                "screen_inches": _numeric_consistency(_axis_vals(rows, "size", "screen_inches"), step=0.5),
            }
            self._variant_stats[vid] = _CohortStats(centroid=_centroid(rows), size=len(rows), numeric_axes_stats=stats)

        # 7) Optional calibrator
        if _SKLEARN_AVAILABLE and calibration_labels is not None:
            feats = [self._features_for(str(r["product_id"]), str(r["group_id"]), str(r["variant_id"]), df)
                     for _, r in assignments_df.iterrows()]
            Xcal = np.array([[f[k] for k in self._feature_names] for f in feats], dtype=float)
            ycal = (np.array(calibration_labels, dtype=float).ravel() > 0.5).astype(int)
            self._calibrator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.seed)  # type: ignore
            self._calibrator.fit(Xcal, ycal)  # type: ignore

        return self

    # --------------------------- per-row features --------------------------

    def _features_for(self, product_id: str, group_id: str, variant_id: str, df: pd.DataFrame) -> Dict[str, float]:
        """Compute scalar features for one assignment."""
        # row index by product_id (first match)
        try:
            i = int(df.index[df["product_id"].astype(str) == str(product_id)][0])
        except Exception:
            i = 0

        xi = self._X[i]

        # Text cohesion
        gstats = self._group_stats.get(group_id)
        vstats = self._variant_stats.get(variant_id)
        group_sim = _cosine(xi, gstats.centroid) if gstats else 0.0
        variant_sim = _cosine(xi, vstats.centroid) if vstats else 0.0

        # Axis presence across 5 simple signals
        axes = df.iloc[i].get("axes") or {}
        present = 0
        total = 5
        if (axes.get("config") or {}).get("ram_gb") is not None: present += 1
        if (axes.get("config") or {}).get("storage_gb") is not None: present += 1
        if (axes.get("config") or {}).get("color") is not None: present += 1
        if (axes.get("size") or {}).get("screen_inches") is not None: present += 1
        if (axes.get("silicon") or {}).get("cpu") is not None: present += 1
        axes_presence = present / float(total)

        # Variant numeric consistency (mode fraction across RAM/Storage/Screen)
        if vstats and vstats.numeric_axes_stats:
            ms = [vstats.numeric_axes_stats[k]["mode_frac"] for k in ("ram_gb", "storage_gb", "screen_inches")]
            axes_consistency = float(np.mean(ms))
        else:
            axes_consistency = 0.0

        # Cohort sizes (simple saturating normalizers)
        group_size_norm = math.tanh((gstats.size if gstats else 0) / 10.0)
        variant_size_norm = math.tanh((vstats.size if vstats else 0) / 5.0)

        # Screen sanity
        subcat = _as_str(df.iloc[i].get("sub_category"))
        title = _as_str(df.iloc[i].get("name"))
        screen = (axes.get("size") or {}).get("screen_inches")
        screen_invalid_flag = 0.0
        if screen is not None and not in_valid_screen_range(subcat, float(screen), title):
            screen_invalid_flag = 1.0

        return {
            "variant_sim": float(max(0.0, min(1.0, variant_sim))),
            "group_sim": float(max(0.0, min(1.0, group_sim))),
            "axes_presence": float(max(0.0, min(1.0, axes_presence))),
            "axes_consistency": float(max(0.0, min(1.0, axes_consistency))),
            "variant_size_norm": float(max(0.0, min(1.0, variant_size_norm))),
            "group_size_norm": float(max(0.0, min(1.0, group_size_norm))),
            "screen_invalid_flag": float(screen_invalid_flag),
        }

    # ------------------------------- scoring ------------------------------

    def _score_det(self, f: Dict[str, float]) -> float:
        """Deterministic weighted logistic mapping."""
        z = (
            self.w_variant_sim * f["variant_sim"]
            + self.w_group_sim * f["group_sim"]
            + self.w_axes_presence * f["axes_presence"]
            + self.w_axes_consistency * f["axes_consistency"]
            + self.w_variant_size * f["variant_size_norm"]
            + self.w_group_size * f["group_size_norm"]
            - self.penalty_screen_invalid * f["screen_invalid_flag"]
        )
        # map roughly [0,~1.9] into [0,1]
        return float(1.0 / (1.0 + math.exp(-4.0 * (z - 0.5))))

    def _score(self, f: Dict[str, float]) -> float:
        if self._calibrator is None:
            return self._score_det(f)
        X = np.array([[f[k] for k in self._feature_names]], dtype=float)
        p = self._calibrator.predict_proba(X)[0, 1]  # type: ignore
        return float(max(0.0, min(1.0, p)))

    # ------------------------------- evidence -----------------------------

    def _evidence(self, f: Dict[str, float]) -> List[str]:
        tags: List[str] = []
        # text cohesion
        tags.append("variant_text_strong" if f["variant_sim"] >= 0.85 else ("variant_text_good" if f["variant_sim"] >= 0.70 else "variant_text_weak"))
        if f["group_sim"] >= 0.80:
            tags.append("group_text_strong")
        elif f["group_sim"] >= 0.65:
            tags.append("group_text_good")
        # axes
        tags.append("axes_complete" if f["axes_presence"] >= 0.8 else ("axes_partial" if f["axes_presence"] >= 0.4 else "axes_sparse"))
        if f["axes_consistency"] >= 0.9:
            tags.append("variant_axes_consistent")
        elif f["axes_consistency"] >= 0.7:
            tags.append("variant_axes_mostly_consistent")
        # cohort sizes
        if f["variant_size_norm"] >= 0.7:
            tags.append("variant_popular")
        if f["group_size_norm"] >= 0.7:
            tags.append("group_popular")
        # sanity
        if f["screen_invalid_flag"] > 0.5:
            tags.append("screen_out_of_range")
        return tags

    # ------------------------------- public API ---------------------------

    def score_assignments(self, df: pd.DataFrame, assignments_df: pd.DataFrame, *, return_features=False) -> pd.DataFrame:
        """
        Return a new dataframe with [product_id, group_id, variant_id, confidence, evidence].
        If return_features=True, includes per-feature columns used in scoring.
        """
        records: List[Dict[str, Any]] = []
        for _, r in assignments_df.iterrows():
            pid, gid, vid = str(r["product_id"]), str(r["group_id"]), str(r["variant_id"])
            feats = self._features_for(pid, gid, vid, df)
            conf = round(self._score(feats), 4)
            ev = ",".join(self._evidence(feats))
            out = dict(r)
            out["confidence"] = conf
            out["evidence"] = ev
            if return_features:
                for k in self._feature_names:
                    out[k] = feats[k]
            records.append(out)
        return pd.DataFrame.from_records(records)