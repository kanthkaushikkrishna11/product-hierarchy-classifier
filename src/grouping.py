# src/grouping.py
# -*- coding: utf-8 -*-
"""
Embedding-based product grouping with readable, hash-free IDs.

Steps
-----
1) Brand-aware blocking: (sub_category, normalized brand, screen-size bucket)
2) Text embeddings: SBERT (preferred) or TF-IDF fallback
3) Cosine radius neighbors -> connected components
4) Deterministic, readable group_id:
     brand_family_slug[_year][-2|-3|...]

This module returns only group assignments and minimal group metadata.
"""

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from collections import defaultdict

# Optional SBERT
_HAS_SBERT = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

# Optional sklearn
_HAS_SK = True
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_normalize
except Exception:
    _HAS_SK = False

from .normalize import slugify, normalize_brand, canonical_title

logger = logging.getLogger(__name__)

__all__ = ["ProductGrouper", "GroupingResult"]


# ------------------------------- data model -------------------------------

@dataclass
class GroupingResult:
    """Output of the grouping stage."""
    group_ids: list          # list[str], aligned with df.index
    groups: list             # list[dict], per-group metadata


# ------------------------------- small utils ------------------------------

def _as_str(x):
    """Safe string conversion."""
    return "" if x is None else str(x)


def _clean_family_phrase(title, brand_norm):
    """
    Build a short family phrase from canonicalized title:
      - drop leading brand token
      - drop trivial refurb/open-box tokens
      - keep first ~5 tokens for readability
    """
    t = canonical_title(title)
    tokens = t.split()
    b = (brand_norm or "").strip().lower()
    if tokens and b and tokens[0] == b:
        tokens = tokens[1:]
    while tokens and tokens[0] in {"renewed", "refurbished", "restored", "certified", "open", "box"}:
        tokens = tokens[1:]
    head = " ".join(tokens[:5]).strip()
    return head or t


def _extract_year(text):
    """Return 2010–2039 if present, else None."""
    m = re.search(r"\b(20[1-3]\d)\b", _as_str(text))
    return m.group(1) if m else None


def _size_bucket(sub, screen_inches):
    """Half-inch buckets for laptops; 5-inch buckets for TVs; 'u' if unknown."""
    if screen_inches is None:
        return "u"
    try:
        s = float(screen_inches)
    except Exception:
        return "u"
    if (sub or "").lower() == "laptop":
        return f"{round(s * 2) / 2:.1f}"
    return f"{int(round(s / 5.0) * 5)}"


class _UF:
    """Tiny union-find for connected components."""
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


# --------------------------------- grouper ---------------------------------

class ProductGrouper:
    """
    Brand-aware blocking + embedding similarity with cosine radius neighbors.
    Deterministic processing; readable, hash-free group IDs.

    Usage:
      grouper = ProductGrouper(threshold=0.82, use_sbert=True)
      result = grouper.build(df)  # result: GroupingResult
    """

    def __init__(
        self,
        threshold=0.82,
        use_sbert=True,
        min_block_size=2,
        random_state=42,
        tfidf_min_df=3,
        tfidf_max_df=0.95,
        tfidf_ngram=(1, 2),
        title_field="name",
    ):
        self.threshold = float(threshold)
        self.use_sbert = bool(use_sbert and _HAS_SBERT)
        self.min_block_size = int(min_block_size)
        self.random_state = int(random_state)
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_ngram = tfidf_ngram
        self.title_field = title_field

        self._sbert = None
        self._tfidf = None  # TfidfVectorizer

    # ------------------------------ embeddings --------------------------- #

    def _maybe_load_sbert(self):
        """Lazily load SBERT if requested and available."""
        if self._sbert is None and self.use_sbert:
            try:
                self._sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Grouping: using SBERT all-MiniLM-L6-v2")
            except Exception:
                self._sbert = None
                logger.warning("Grouping: SBERT unavailable; falling back to TF-IDF.")
        return self._sbert

    def _embed(self, texts):
        """Return L2-normalized embeddings (np.array or sparse matrix)."""
        model = self._maybe_load_sbert()
        if model is not None:
            try:
                vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return np.asarray(vecs, dtype=np.float32)
            except Exception:
                logger.warning("Grouping: SBERT encode failed; falling back to TF-IDF.")

        if not _HAS_SK:
            # No sklearn: identity (isolates items)
            return np.eye(len(texts), dtype=np.float32)

        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(
                min_df=self.tfidf_min_df,
                max_df=self.tfidf_max_df,
                ngram_range=self.tfidf_ngram,
                sublinear_tf=True,
            )
            X = self._tfidf.fit_transform(texts)
        else:
            X = self._tfidf.transform(texts)

        return sk_normalize(X, norm="l2", copy=False)

    # ------------------------------- helpers ----------------------------- #

    def _block_key(self, row):
        """
        Blocking key: (sub_category, normalized brand, size bucket).
        Uses axes.size.screen_inches if present; 'u' otherwise.
        """
        sub = _as_str(row.get("sub_category")).strip().lower() or "unknown"
        brand_norm = normalize_brand(_as_str(row.get("brand"))) or "_unknown"
        axes = row.get("axes") or {}
        size = (axes.get("size") or {}).get("screen_inches")
        bucket = _size_bucket(sub, size)
        return (sub, brand_norm, bucket)

    def _radius_neighbors(self, X, radius):
        """Cosine radius neighbors list for each row index."""
        if not _HAS_SK:
            return [[i] for i in range(X.shape[0])]
        nn = NearestNeighbors(radius=radius, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(X)
        _, idxs = nn.radius_neighbors(X, radius=radius, return_distance=True)
        out = []
        for i, js in enumerate(idxs):
            # ensure self is present
            if i not in js:
                js = np.append(js, i)
            out.append(list(js))
        return out

    def _components(self, neighbors):
        """Connected components via union-find."""
        n = len(neighbors)
        uf = _UF(n)
        for i, js in enumerate(neighbors):
            for j in js:
                if i != j:
                    uf.union(i, j)
        root2members = defaultdict(list)
        for i in range(n):
            root2members[uf.find(i)].append(i)
        return [sorted(m) for m in root2members.values()]

    def _medoid(self, comp, X):
        """Pick representative index (highest similarity to centroid)."""
        if len(comp) == 1:
            return comp[0]
        if hasattr(X, "toarray"):  # sparse
            sub = X[comp]
            c = sk_normalize(sub.mean(axis=0), norm="l2", copy=False)
            sims = (sub @ c.T).A.ravel()
        else:  # dense
            sub = X[comp]
            c = np.mean(sub, axis=0, dtype=np.float32)
            c /= (np.linalg.norm(c) + 1e-12)
            sims = sub @ c
        best = int(np.argmax(sims))
        max_sim = float(sims[best])
        ties = [k for k, s in enumerate(sims) if abs(float(s) - max_sim) < 1e-9]
        return comp[min(ties)] if len(ties) > 1 else comp[best]

    def _base_group_id(self, brand_norm, medoid_title):
        """brand_family_slug[_year] from medoid title; returns (id, family_phrase, year?)."""
        brand_token = (brand_norm or "unknown").lower()
        family_phrase = _clean_family_phrase(medoid_title, brand_token)
        family_slug = slugify(family_phrase) or "product"
        generation = _extract_year(medoid_title)
        base = f"{brand_token}_{family_slug}"
        if generation:
            base = f"{base}_{generation}"
        return base, family_phrase, generation

    # --------------------------------- API -------------------------------- #

    def build(self, df):
        """
        Group products and return GroupingResult.

        Inputs
        ------
        df : pandas.DataFrame
          Must include columns: 'name', 'brand', 'sub_category'.
          If 'axes.size.screen_inches' is present, it improves blocking.

        Returns
        -------
        GroupingResult
          .group_ids: list[str] aligned with df.index
          .groups: list[dict] with group metadata
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("ProductGrouper.build expects a pandas DataFrame")

        n = len(df)
        if n == 0:
            return GroupingResult(group_ids=[], groups=[])

        # Build blocks
        blocks = defaultdict(list)
        for i, row in df.iterrows():
            blocks[self._block_key(row)].append(i)

        group_ids = [""] * n
        group_records = []
        radius = 1.0 - self.threshold

        logger.info(
            "Grouping: cosine threshold=%.3f (radius=%.3f); SBERT=%s",
            self.threshold, radius, str(self.use_sbert),
        )

        # global ID collision map
        id_counts = {}

        # Process blocks deterministically
        for (sub, brand_norm, bucket) in sorted(blocks.keys()):
            idxs = blocks[(sub, brand_norm, bucket)]
            titles = [_as_str(df.loc[i, self.title_field]) for i in idxs]
            texts = [canonical_title(t) for t in titles]

            X = self._embed(texts)

            # Build components
            if len(idxs) < self.min_block_size or not _HAS_SK:
                comps = [[k] for k in range(len(idxs))]
            else:
                neighbors = self._radius_neighbors(X, radius=radius)
                comps = self._components(neighbors)

            pending = []
            for comp in comps:
                members = [idxs[j] for j in comp]
                med_local = self._medoid(comp, X)
                med_global = idxs[med_local]
                med_title = titles[med_local]
                base_id, fam, gen = self._base_group_id(
                    None if brand_norm == "_unknown" else brand_norm,
                    med_title,
                )
                pending.append((base_id, med_global, members, brand_norm, fam, gen))

            # Deterministic assignment and suffixing
            pending.sort(key=lambda t: (t[0], t[1]))
            for base_id, medoid_idx, members, bnorm, fam, gen in pending:
                id_counts[base_id] = id_counts.get(base_id, 0) + 1
                count = id_counts[base_id]
                gid = base_id if count == 1 else f"{base_id}-{count}"

                for gi in members:
                    group_ids[gi] = gid

                group_records.append(
                    {
                        "group_id": gid,
                        "brand": None if bnorm == "_unknown" else bnorm,
                        "family": fam,
                        "generation": gen,
                        "member_indices": sorted(members),
                    }
                )

        # Safety: assign any missing (should not happen)
        for i in range(n):
            if not group_ids[i]:
                title = _as_str(df.loc[i, self.title_field])
                brand_norm = normalize_brand(_as_str(df.loc[i, "brand"])) or "_unknown"
                base_id, fam, gen = self._base_group_id(
                    None if brand_norm == "_unknown" else brand_norm, title
                )
                id_counts[base_id] = id_counts.get(base_id, 0) + 1
                gid = base_id if id_counts[base_id] == 1 else f"{base_id}-{id_counts[base_id]}"
                group_ids[i] = gid
                group_records.append(
                    {
                        "group_id": gid,
                        "brand": None if brand_norm == "_unknown" else brand_norm,
                        "family": fam,
                        "generation": gen,
                        "member_indices": [i],
                    }
                )

        return GroupingResult(group_ids=group_ids, groups=group_records)

    # Backwards-compat name used in some earlier snippets
    def fit_predict(self, df):
        """Alias for build(df) for compatibility."""
        return self.build(df)