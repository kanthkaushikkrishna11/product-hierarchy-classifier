# src/bundle_detection.py
# -*- coding: utf-8 -*-
"""
Bundle and multi-pack detection with weak supervision + lightweight ML.

Usage
-----
detector = BundleDetector()
detector.fit(df)         # trains on weak labels derived from text/specs
out = detector.detect(df)  # returns a new DataFrame with bundle columns

Optional
--------
df_axes = apply_to_axes(df, out)  # sets axes['packaging']['bundle']=True when bundled
"""

from __future__ import annotations

import re
from typing import Any, List, Dict

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SK_AVAILABLE = True
except Exception:
    SK_AVAILABLE = False


# --------------------------- small utilities --------------------------- #

def _as_str(x):
    """Safe str cast."""
    return "" if x is None else str(x)


def _get_text(row):
    """Primary text for patterns and ML."""
    return _as_str(row.get("ml_text") or row.get("name"))


def _get_specs(row):
    return row.get("specs") or {}


def _get_specs_lc(row):
    return row.get("specs_lc") or {}


def _get_details(row):
    return row.get("details_parsed") or {}


# -------------------------- pattern dictionaries ----------------------- #

# High-precision bundle cues in titles/descriptions.
# (Kept conservative to minimize false positives; the model will generalize.)
BUNDLE_PHRASES = [
    r"\bbundle\b",
    r"\bcombo\b",
    r"\bkit\b",
    r"\bwith\b",          # use with caution; rely on context + model
    r"\bincludes?\b",
    r"\b\+\b",            # "TV + Soundbar"
    r"\b&\b",
]

# Multi-pack cues
PACK_PATTERNS = [
    r"\b(\d+)\s*pack\b",
    r"\b(\d+)\s*pk\b",
    r"\b(\d+)\s*-\s*pack\b",
    r"\bx\s*(\d+)\b",     # "x2"
    r"\b(\d+)\s*count\b",
]

# Common accessories to extract (small, general list; order matters for matching).
ACCESSORY_LEXICON = [
    "soundbar", "subwoofer", "mount", "wall mount", "hdmi cable", "cable", "dock",
    "bag", "backpack", "sleeve", "case", "cover", "mouse", "keyboard", "stylus",
    "headset", "earbuds", "headphones", "webcam", "mic", "microphone",
    "micro sd", "sd card", "memory card", "usb drive", "flash drive",
    "adapter", "charger", "power adapter", "pen", "stand",
]


# ------------------------------ weak labels ---------------------------- #

def _weak_label(row):
    """
    Return 1 for high-confidence bundle, 0 for high-confidence standalone, None otherwise.
    """
    name = _get_text(row).lower()
    specs_lc = _get_specs_lc(row)
    details = _get_details(row)

    # Positive seeds: obvious bundle/kit/combo patterns.
    if re.search(r"\b(bundle|kit|combo)\b", name):
        return 1
    if re.search(r"\bwith\b", name) and re.search(r"\b(mouse|bag|soundbar|case|keyboard|micro sd|sd card|dock|mount)\b", name):
        return 1
    if re.search(r"\b\+\b", name) and re.search(r"\b(mouse|bag|soundbar|case|keyboard|micro sd|sd card|dock|mount)\b", name):
        return 1

    # Positive seeds via specs / variations (e.g., Edition: "... + 256GB Micro SD Card")
    acc_val = specs_lc.get("accessories included")
    if isinstance(acc_val, str) and len(acc_val) >= 3:
        return 1

    # productVariations frequently encode accessory bundles in 'Edition'
    pvars = details.get("productVariations") or []
    if isinstance(pvars, list):
        for pv in pvars:
            opts = (pv or {}).get("options") or {}
            for v in opts.values():
                if isinstance(v, str) and re.search(r"\b\+\s*\d* ?(micro sd|sd card|bag|mouse|case|keyboard|dock|mount)\b", v.lower()):
                    return 1

    # Negative seeds: explicit "laptop only" / "unit only" phrasing
    if re.search(r"\b(laptop|tv)\s+only\b", name) or "unit only" in name:
        return 0

    # No strong evidence → unlabeled (model will skip).
    return None


# ------------------------------ extraction ----------------------------- #

def _extract_pack_qty(text):
    """Return integer pack quantity if detected; else None."""
    s = text.lower()
    for pat in PACK_PATTERNS:
        m = re.search(pat, s)
        if m:
            try:
                q = int(m.group(1))
                if 2 <= q <= 50:
                    return q
            except Exception:
                pass
    return None


def _extract_accessories_from_text(text):
    """Scan text for accessory lexicon (dedup, order preserved)."""
    found = []
    s = text.lower()
    for acc in ACCESSORY_LEXICON:
        # simple containment; prefer longest phrases first to avoid duplicates
        if acc in s and acc not in found:
            found.append(acc)
    return found


def _extract_accessories_from_specs(specs_lc):
    """Parse 'Accessories Included' value (comma/pipe separated)."""
    val = specs_lc.get("accessories included")
    if not isinstance(val, str):
        return []
    # split on common separators
    parts = re.split(r"[,\|;/]+", val.lower())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(p)
    return out


def _extract_accessories_from_variations(details):
    """Look in productVariations -> options for accessory hints."""
    acc = []
    pvars = details.get("productVariations") or []
    if not isinstance(pvars, list):
        return acc
    for pv in pvars:
        opts = (pv or {}).get("options") or {}
        for v in opts.values():
            if not isinstance(v, str):
                continue
            s = v.lower()
            if re.search(r"\b\+\s*\d* ?(micro sd|sd card|bag|mouse|case|keyboard|dock|mount)\b", s):
                # canonicalize a bit
                if "micro sd" in s or "sd card" in s or "memory card" in s:
                    acc.append("micro sd")
                if "bag" in s:
                    acc.append("bag")
                if "mouse" in s:
                    acc.append("mouse")
                if "case" in s or "sleeve" in s:
                    acc.append("case")
                if "keyboard" in s:
                    acc.append("keyboard")
                if "dock" in s:
                    acc.append("dock")
                if "mount" in s:
                    acc.append("mount")
    # dedup keep order
    out = []
    for a in acc:
        if a not in out:
            out.append(a)
    return out


def _guess_main_product(row):
    """Anchor product type; fall back to sub_category or a simple guess from name."""
    sub = _as_str(row.get("sub_category")).strip().lower()
    if sub:
        return sub
    name = _get_text(row).lower()
    if "laptop" in name or "chromebook" in name or "macbook" in name:
        return "laptop"
    if "tv" in name or "television" in name:
        return "tv"
    if "monitor" in name:
        return "monitor"
    return "product"


# ------------------------------ main class ------------------------------ #

class BundleDetector:
    """
    Weakly-supervised bundle detector with TF-IDF + Logistic Regression.

    fit(df):
      - Builds weak labels from high-precision rules.
      - Trains a tiny classifier on df['ml_text'].

    detect(df):
      - Uses the classifier (if trained) plus extraction helpers
        to return a DataFrame with bundle annotations.
    """

    def __init__(self, min_positive=40, min_negative=60, min_df=3, max_df=0.95):
        self.min_positive = int(min_positive)
        self.min_negative = int(min_negative)
        self.min_df = min_df
        self.max_df = max_df
        self._pipe = None  # sklearn pipeline

    def fit(self, df: pd.DataFrame):
        if not SK_AVAILABLE:
            return self

        labels = []
        idxs = []
        texts = []
        for i, row in df.iterrows():
            y = _weak_label(row)
            if y is None:
                continue
            labels.append(int(y))
            idxs.append(i)
            texts.append(_get_text(row))

        pos = sum(labels)
        neg = len(labels) - pos
        if pos < self.min_positive or neg < self.min_negative:
            # Not enough seeds -> skip training (still do rule-based detect)
            return self

        self._pipe = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=self.min_df, max_df=self.max_df, ngram_range=(1, 2), sublinear_tf=True)),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        self._pipe.fit(texts, labels)
        return self

    def _predict_proba(self, texts: List[str]):
        if self._pipe is None or not SK_AVAILABLE:
            # no model -> neutral probability
            return np.array([0.5] * len(texts), dtype=float)
        try:
            p = self._pipe.predict_proba(texts)[:, 1]
            return p
        except Exception:
            return np.array([0.5] * len(texts), dtype=float)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = [_get_text(r) for _, r in df.iterrows()]
        probs = self._predict_proba(texts)

        records: List[Dict[str, Any]] = []
        for (i, row), proba in zip(df.iterrows(), probs):
            name = _get_text(row)
            specs_lc = _get_specs_lc(row)
            details = _get_details(row)

            # rules (high-precision) to upgrade probability
            is_pack = _extract_pack_qty(name) is not None
            has_bundle_word = bool(re.search(r"\b(bundle|kit|combo)\b", name.lower()))
            with_plus = (" with " in name.lower()) or (" + " in name.lower()) or (" includes " in name.lower())

            # combine model & rules
            bundle_p = proba
            if has_bundle_word:
                bundle_p = max(bundle_p, 0.9)
            if with_plus:
                bundle_p = max(bundle_p, 0.7)
            if is_pack:
                bundle_p = max(bundle_p, 0.85)

            pack_qty = _extract_pack_qty(name)
            accessories = []
            accessories += _extract_accessories_from_text(name)
            accessories += _extract_accessories_from_specs(specs_lc)
            accessories += _extract_accessories_from_variations(details)
            # dedup while preserving order
            seen = set()
            accessories = [a for a in accessories if not (a in seen or seen.add(a))]

            bundle_type = "standalone"
            if pack_qty and pack_qty >= 2:
                bundle_type = "multi_pack"
            elif bundle_p >= 0.6 and len(accessories) >= 1:
                bundle_type = "accessory_bundle"

            is_bundle = bundle_type != "standalone"
            main_product = _guess_main_product(row)

            evidence = []
            if has_bundle_word: evidence.append("word:bundle/kit/combo")
            if with_plus:       evidence.append("with/plus/includes")
            if pack_qty:        evidence.append(f"pack:{pack_qty}")
            if accessories:     evidence.append(f"acc:{len(accessories)}")
            if bundle_p >= 0.8: evidence.append("model:strong")
            elif bundle_p >= 0.6: evidence.append("model:weak")

            records.append({
                "product_id": row.get("product_id"),
                "is_bundle": bool(is_bundle),
                "bundle_type": bundle_type,
                "bundle_confidence": float(round(min(1.0, max(0.0, bundle_p)), 4)),
                "main_product": main_product,
                "accessories": accessories,
                "pack_qty": int(pack_qty) if pack_qty else None,
                "bundle_evidence": ",".join(evidence),
            })

        return pd.DataFrame.from_records(records)


# --------------------------- axes application --------------------------- #

def apply_to_axes(df: pd.DataFrame, bundle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge bundle annotations back into the 'axes.packaging' dict, setting bundle=True
    where is_bundle. Non-destructive: copies the axes column.
    """
    if df.empty or bundle_df.empty:
        return df
    df = df.copy()
    # align by product_id
    ann = {str(r["product_id"]): bool(r["is_bundle"]) for _, r in bundle_df.iterrows()}
    new_axes = []
    for _, row in df.iterrows():
        pid = str(row.get("product_id"))
        axes = (row.get("axes") or {}).copy()
        packaging = dict(axes.get("packaging") or {})
        if ann.get(pid, False):
            packaging["bundle"] = True
        if packaging:
            axes["packaging"] = packaging
        new_axes.append(axes)
    df["axes"] = new_axes
    return df