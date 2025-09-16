# src/extractors.py
# -*- coding: utf-8 -*-
"""
Axis extraction using weak labels + small ML models.

Targets learned (multiclass):
  - RAM (GB)
  - Storage (GB)
  - Screen size (rounded to 0.5")
  - Color (normalized token)
  - CPU token (e.g., 'intel_i5_1235u', 'apple_m2', 'amd_ryzen7_7730u')

GPU, condition, and bundle are derived via gentle normalizers (not modeled yet).
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Optional ML stack (graceful fallback if missing)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False

from .specs_map import SpecIndexer
from .normalize import (
    parse_ram_gb,
    parse_storage_to_gb,
    parse_screen_inches,
    in_valid_screen_range,
    normalize_color,
    normalize_cpu,
    normalize_gpu,
    normalize_condition,
)


# ------------------------------- small helpers -------------------------------

def _round_half_inch(x):
    """Round a float to the nearest 0.5."""
    try:
        return round(float(x) * 2.0) / 2.0
    except Exception:
        return None


def _as_str(x):
    """Safe string conversion."""
    return "" if x is None else str(x)


def _collect_text(row):
    """
    Primary text for models. Start with loader's `ml_text`,
    then add a few high-signal spec snippets when present.
    """
    parts = [
        _as_str(row.get("ml_text")),
        _as_str(row.get("name")),
        _as_str(row.get("brand")),
    ]

    specs = row.get("specs") or {}
    for k in (
        "Processor Brand",
        "Processor Type",
        "Processor Model",
        "RAM Memory",
        "Hard Drive Capacity",
        "Solid State Drive Capacity",
        "Screen Size",
        "Color",
        "Condition",
    ):
        v = specs.get(k) or specs.get(k.lower())
        if v:
            parts.append(str(v))

    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()


def _weak_labels(df):
    """
    Build weak labels with gentle parsers + spec mapping.
    Returns a dict of lists aligned to df rows.
    """
    y_ram, y_storage, y_screen, y_color, y_cpu = [], [], [], [], []

    for _, row in df.iterrows():
        name = _as_str(row.get("name"))
        subcat = _as_str(row.get("sub_category"))
        specs = row.get("specs") or {}
        specs_lc = row.get("specs_lc") or {}
        idx = SpecIndexer(specs, specs_lc)

        # RAM
        ram = None
        for t in (_as_str(idx.first("ram_gb")), name):
            ram = ram or parse_ram_gb(t)
        y_ram.append(ram)

        # Storage
        storage = None
        for t in (_as_str(idx.first("storage_gb")), name):
            storage = storage or parse_storage_to_gb(t)
        y_storage.append(storage)

        # Screen inches
        screen = None
        for t in (_as_str(idx.first("screen_inches")), name):
            screen = screen or parse_screen_inches(t)
        if screen and not in_valid_screen_range(subcat, screen, name):
            screen = None
        y_screen.append(screen)

        # Color
        color = None
        for t in (_as_str(idx.first("color")), name):
            c = normalize_color(t)
            if c:
                color = c
                break
        y_color.append(color)

        # CPU token
        cpu_text = " ".join(
            [
                name,
                _as_str(idx.first("cpu_brand")),
                _as_str(idx.first("cpu_family")),
                _as_str(idx.first("cpu_model")),
            ]
        )
        y_cpu.append(normalize_cpu(cpu_text))

    return {
        "ram_gb": y_ram,
        "storage_gb": y_storage,
        "screen_inches": y_screen,
        "color": y_color,
        "cpu": y_cpu,
    }


def _prep_labels(values, transform=None):
    """
    Convert raw values into string classes; return:
      labels_str, labeled_row_idx, str2idx, idx2str
    """
    labels_str, labeled_idx = [], []
    for i, v in enumerate(values):
        if v is None:
            continue
        u = transform(v) if transform else v
        if u is None:
            continue
        labels_str.append(str(u))
        labeled_idx.append(i)

    uniq = sorted(set(labels_str), key=lambda s: (len(s), s))
    str2idx = {s: j for j, s in enumerate(uniq)}
    idx2str = {j: s for s, j in str2idx.items()}
    return labels_str, labeled_idx, str2idx, idx2str


# -------------------------------- main class --------------------------------

@dataclass
class AxisPrediction:
    value: object | None
    proba: float | None
    source: str  # 'parse' | 'model' | 'none'


class AxisModeler:
    """
    Train small classifiers per-attribute using weak labels, then infer axes for rows.

    Usage:
      mdl = AxisModeler()
      mdl.fit(df)
      axes = mdl.infer_axes(row)
    """

    def __init__(self, random_state=42, min_class_support=10, tfidf_min_df=3, tfidf_max_df=0.95):
        self.random_state = random_state
        self.min_class_support = min_class_support
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df

        self.vectorizer = None  # shared TF-IDF
        self.models = {}        # target -> fitted clf
        self.idx2str = {}       # target -> {class_idx: label_str}
        self.str2idx = {}       # target -> {label_str: class_idx}

    # ------------------------------- training -------------------------------

    def fit(self, df):
        """Fit TF-IDF + LogisticRegression per target using weak labels."""
        if not SKLEARN_AVAILABLE:
            # Parse-only mode if sklearn isn't available
            return self

        corpus = [_collect_text(r) for _, r in df.iterrows()]
        self.vectorizer = TfidfVectorizer(
            min_df=self.tfidf_min_df,
            max_df=self.tfidf_max_df,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X_all = self.vectorizer.fit_transform(corpus)

        weak = _weak_labels(df)

        self._fit_one("ram_gb", X_all, weak["ram_gb"], transform=lambda v: int(v))
        self._fit_one("storage_gb", X_all, weak["storage_gb"], transform=lambda v: int(v))
        self._fit_one("screen_inches", X_all, weak["screen_inches"], transform=lambda v: _round_half_inch(float(v)))
        self._fit_one("color", X_all, weak["color"], transform=lambda v: str(v))
        self._fit_one("cpu", X_all, weak["cpu"], transform=lambda v: str(v))

        return self

    def _fit_one(self, target, X_all, labels, transform):
        """Fit one classifier if enough labeled examples/classes exist."""
        if not SKLEARN_AVAILABLE or self.vectorizer is None:
            return

        labels_str, labeled_idx, str2idx, idx2str = _prep_labels(labels, transform=transform)
        if len(labeled_idx) < max(self.min_class_support, 8) or len(str2idx) < 2:
            return

        y = np.array([str2idx[s] for s in labels_str], dtype=int)
        X = X_all[labeled_idx]

        # balance classes
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight = {c: w for c, w in zip(classes, weights)}

        clf = LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            class_weight=class_weight,
            random_state=self.random_state,
        )
        clf.fit(X, y)

        self.models[target] = clf
        self.idx2str[target] = idx2str
        self.str2idx[target] = str2idx

    # -------------------------------- inference --------------------------------

    def _predict_class(self, target, text):
        """Predict label for one target from text using trained model."""
        model = self.models.get(target)
        if not SKLEARN_AVAILABLE or model is None or self.vectorizer is None:
            return AxisPrediction(None, None, "none")

        X = self.vectorizer.transform([text])
        proba = None
        try:
            probs = model.predict_proba(X)[0]
            j = int(np.argmax(probs))
            proba = float(probs[j])
        except Exception:
            j = int(model.predict(X)[0])

        label = self.idx2str[target].get(j)

        # Cast back to numeric when appropriate
        if target == "screen_inches" and label is not None:
            try:
                return AxisPrediction(float(label), proba, "model")
            except Exception:
                pass
        if target in ("ram_gb", "storage_gb") and label is not None:
            try:
                return AxisPrediction(int(float(label)), proba, "model")
            except Exception:
                pass

        return AxisPrediction(label, proba, "model")

    def infer_axes(self, row):
        """
        Infer axes for a single product row (pd.Series or dict).
        Prefers parsed values; falls back to model predictions.
        """
        r = row if isinstance(row, pd.Series) else pd.Series(row)

        name = _as_str(r.get("name"))
        subcat = _as_str(r.get("sub_category"))
        specs = r.get("specs") or {}
        specs_lc = r.get("specs_lc") or {}
        text = _collect_text(r)

        idx = SpecIndexer(specs, specs_lc)

        # ------------------- CONFIG: RAM / storage / color -------------------
        # RAM
        ram = None
        for t in (idx.first("ram_gb"), name):
            ram = ram or parse_ram_gb(_as_str(t))
        ram_src = "parse" if ram is not None else "none"
        if ram is None:
            pred = self._predict_class("ram_gb", text)
            if pred.value is not None:
                ram, ram_src = pred.value, pred.source

        # Storage
        storage = None
        for t in (idx.first("storage_gb"), name):
            storage = storage or parse_storage_to_gb(_as_str(t))
        storage_src = "parse" if storage is not None else "none"
        if storage is None:
            pred = self._predict_class("storage_gb", text)
            if pred.value is not None:
                storage, storage_src = pred.value, pred.source

        # Color
        color = None
        for t in (idx.first("color"), name):
            c = normalize_color(_as_str(t))
            if c:
                color = c
                break
        color_src = "parse" if color is not None else "none"
        if color is None:
            pred = self._predict_class("color", text)
            # modest threshold to avoid over-coloring laptops
            if pred.value is not None and (pred.proba is None or pred.proba >= 0.55):
                color, color_src = str(pred.value), pred.source

        config_axis = {}
        if ram is not None:
            config_axis["ram_gb"] = ram
        if storage is not None:
            config_axis["storage_gb"] = storage
        if color is not None:
            config_axis["color"] = color
        if not config_axis:
            config_axis = None

        # ------------------------- SIZE: screen inches ------------------------
        screen = None
        for t in (idx.first("screen_inches"), name):
            screen = screen or parse_screen_inches(_as_str(t))
        if screen and not in_valid_screen_range(subcat, screen, name):
            screen = None
        screen_src = "parse" if screen is not None else "none"
        if screen is None:
            pred = self._predict_class("screen_inches", text)
            if pred.value is not None and in_valid_screen_range(subcat, float(pred.value), name):
                screen, screen_src = float(pred.value), pred.source

        size_axis = {"screen_inches": float(screen)} if screen is not None else None

        # ------------------------- SILICON: cpu/gpu --------------------------
        cpu_tok = normalize_cpu(
            " ".join(
                [
                    name,
                    _as_str(idx.first("cpu_brand")),
                    _as_str(idx.first("cpu_family")),
                    _as_str(idx.first("cpu_model")),
                ]
            )
        )
        cpu_src = "parse" if cpu_tok else "none"
        if not cpu_tok:
            pred = self._predict_class("cpu", text)
            if pred.value:
                cpu_tok, cpu_src = str(pred.value), pred.source

        gpu_tok = normalize_gpu(" ".join([name, _as_str(idx.first("gpu_model"))]))

        silicon_axis = {}
        if cpu_tok:
            silicon_axis["cpu"] = cpu_tok
        if gpu_tok:
            silicon_axis["gpu"] = gpu_tok
        if not silicon_axis:
            silicon_axis = None

        # ---------------------- PACKAGING: condition/bundle ------------------
        cond_text = " ".join([name, _as_str(idx.first("condition"))])
        condition = normalize_condition(cond_text)

        name_l = name.lower()
        bundle = any(k in name_l for k in (" bundle", " kit", " combo", " with ", "+", " includes "))
        packaging_axis = {"bundle": bool(bundle)}
        if condition:
            packaging_axis["condition"] = condition

        # ---------------------- region / carrier (unused) --------------------
        region_axis = None
        carrier_axis = None

        return {
            "config": config_axis,
            "size": size_axis,
            "silicon": silicon_axis,
            "region": region_axis,
            "carrier": carrier_axis,
            "packaging": packaging_axis,
        }