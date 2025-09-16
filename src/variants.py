# src/variants.py
# -*- coding: utf-8 -*-
"""
Deterministic, readable Variant ID generation and aggregation.

Inputs per row (df columns)
- product_id : str
- group_id   : str
- axes       : dict from extractors (config/size/silicon/packaging/...)

Outputs
- variant_ids       : list aligned with df.index
- variant_records   : list of dicts for variants.json
- assignments_df    : DataFrame [product_id, group_id, variant_id]

Notes
- No hashes. IDs are human-readable and stable.
- Axis order is fixed for determinism.
- packaging.bundle is included only when True.
"""

from __future__ import annotations

import re
import pandas as pd


# ------------------------------- small helpers -------------------------------

def _fmt_gb(x):
    """Return '16gb' from 16; None if unavailable."""
    if x is None:
        return None
    try:
        return f"{int(x)}gb"
    except Exception:
        return None


def _fmt_inches(x):
    """Return inches as string with at most one decimal (e.g., '15.6', '65')."""
    if x is None:
        return None
    try:
        q = round(float(x) + 1e-8, 1)
        s = f"{q:.1f}"
        return re.sub(r"\.0$", "", s)
    except Exception:
        return None


def _join(parts, sep="_"):
    """Join non-empty parts with `sep`; return None if all empty."""
    vals = [p for p in parts if p not in (None, "", "none")]
    return sep.join(vals) if vals else None


# ------------------------------ axis → segment ------------------------------

def _config_segment(config):
    """
    Compose 'config:...' from (ram_gb, storage_gb, color), in that order.
    Examples:
      config:8gb_256gb_black
      config:16gb_512gb
      config:silver
    """
    if not isinstance(config, dict) or not config:
        return None
    ram = _fmt_gb(config.get("ram_gb"))
    storage = _fmt_gb(config.get("storage_gb"))
    color = config.get("color")
    payload = _join([ram, storage, color])
    return f"config:{payload}" if payload else None


def _size_segment(size):
    """Compose 'size:...' from screen_inches. Examples: size:13.6, size:65."""
    if not isinstance(size, dict) or not size:
        return None
    scr = _fmt_inches(size.get("screen_inches"))
    return f"size:{scr}" if scr else None


def _silicon_segment(silicon):
    """
    Compose 'silicon:...' from cpu and gpu tokens.
    Examples:
      silicon:intel_i5_1235u_iris_xe
      silicon:apple_m2
    """
    if not isinstance(silicon, dict) or not silicon:
        return None
    cpu = silicon.get("cpu")
    gpu = silicon.get("gpu")
    payload = _join([cpu, gpu])
    return f"silicon:{payload}" if payload else None


def _packaging_segment(packaging):
    """
    Compose 'packaging:...' from condition and bundle flag.
    Examples:
      packaging:refurbished
      packaging:open_box_bundle
      packaging:bundle         (when only bundle=True)
    """
    if not isinstance(packaging, dict) or not packaging:
        return None
    cond = packaging.get("condition")
    has_bundle = bool(packaging.get("bundle", False))
    parts = []
    if cond:
        parts.append(str(cond))
    if has_bundle:
        parts.append("bundle")
    if not parts:
        return None
    return f"packaging:{'_'.join(parts)}"


# ----------------------------- variant id builder -----------------------------

def serialize_variant_id(group_id, axes):
    """
    Compose a variant_id from group_id and axes.
    Only include non-empty axes, in this order: config / size / silicon / packaging.
    If nothing is present, append 'config:unk'.
    """
    axes = axes or {}
    segments = []

    s = _config_segment(axes.get("config"))
    if s:
        segments.append(s)

    s = _size_segment(axes.get("size"))
    if s:
        segments.append(s)

    s = _silicon_segment(axes.get("silicon"))
    if s:
        segments.append(s)

    # region/carrier intentionally omitted unless populated elsewhere

    s = _packaging_segment(axes.get("packaging"))
    if s:
        segments.append(s)

    if not segments:
        segments = ["config:unk"]

    return "/".join([str(group_id)] + segments)


# --------------------------------- pruning ----------------------------------

def _prune_empty_axes(axes):
    """Remove empty axis dicts/keys so records mirror the serialization."""
    out = {}
    for key in ("config", "size", "silicon", "region", "carrier", "packaging"):
        v = (axes or {}).get(key)
        if isinstance(v, dict):
            vv = {kk: vv for kk, vv in v.items() if vv not in (None, "", [], {})}
            # drop packaging.bundle=False
            if key == "packaging" and vv.get("bundle") is False:
                vv.pop("bundle", None)
            if vv:
                out[key] = vv
        elif v not in (None, "", [], {}):
            out[key] = v
    return out


# ------------------------------- aggregation --------------------------------

def build_variants(
    df,
    *,
    product_id_col="product_id",
    group_id_col="group_id",
    axes_col="axes",
):
    """
    Build per-row variant_id and aggregate variant records.

    Returns
    -------
    (variant_ids, variant_records, assignments_df)
      - variant_ids: list[str] aligned with df.index
      - variant_records: list[dict] with keys [variant_id, group_id, axes, product_count]
      - assignments_df: DataFrame with [product_id, group_id, variant_id]
    """
    n = len(df)
    if n == 0:
        cols = [product_id_col, group_id_col, "variant_id"]
        return [], [], pd.DataFrame(columns=cols)

    # 1) Per-row variant_id
    variant_ids = []
    for _, row in df.iterrows():
        gid = row.get(group_id_col)
        axes = row.get(axes_col) or {}
        vid = serialize_variant_id(str(gid), axes)
        variant_ids.append(vid)

    # 2) Aggregate variant records (first snapshot + counts)
    variant_axes = {}
    variant_counts = {}

    for idx, vid in enumerate(variant_ids):
        if vid not in variant_axes:
            axes = df.iloc[idx][axes_col] or {}
            # mirror serialization: remove empty keys and bundle=False
            axes_clean = dict(axes)
            if isinstance(axes_clean.get("packaging"), dict):
                pack = dict(axes_clean["packaging"])
                if not pack.get("bundle", False):
                    pack.pop("bundle", None)
                if not pack:
                    axes_clean["packaging"] = None
            variant_axes[vid] = _prune_empty_axes(axes_clean)
        variant_counts[vid] = variant_counts.get(vid, 0) + 1

    variant_records = []
    for vid in sorted(variant_axes.keys()):
        gid = vid.split("/")[0]
        variant_records.append({
            "variant_id": vid,
            "group_id": gid,
            "axes": variant_axes[vid],
            "product_count": int(variant_counts[vid]),
        })

    # 3) Assignments table
    assignments_df = pd.DataFrame({
        product_id_col: df[product_id_col].astype(str).values,
        group_id_col: df[group_id_col].astype(str).values,
        "variant_id": variant_ids,
    })

    return variant_ids, variant_records, assignments_df