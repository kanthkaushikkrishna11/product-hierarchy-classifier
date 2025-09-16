# src/pipeline.py
# -*- coding: utf-8 -*-
"""
End-to-end Product Hierarchy Classifier.

Flow:
  1) Load
  2) Extract axes
  3) Group (readable IDs)
  4) Build variants (readable IDs)
  5) Score confidence
  6) Export artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Modules
from .loaders import load_products
from .extractors import AxisModeler
from .grouping import ProductGrouper
from .variants import build_variants
from .scoring import ConfidenceScorer
from .normalize import normalize_panel_type


# ----------------------------- small helpers -----------------------------

def _safe_mode(values):
    """Most frequent non-empty value; deterministic tie-breaker."""
    vals = [v for v in values if v not in (None, "", [], {}, "none")]
    if not vals:
        return None
    c = Counter(vals)
    # sort by (-count, string) for stable ties
    return sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def _cpu_readable(cpu_token):
    """Readable CPU family label from compact token (for JSON readability only)."""
    if not cpu_token:
        return None
    s = str(cpu_token)
    if s.startswith("apple_m"):
        return "Apple " + s.split("_")[1].upper()
    if s.startswith("intel_i"):
        gen = s.split("_")[1]
        return f"Intel Core {gen.upper()}"
    if s.startswith("amd_ryzen"):
        parts = s.split("_")
        for p in parts:
            if p.isdigit():
                return f"AMD Ryzen {p}"
        # fallback if no numeric part isolated
        if len(parts) > 1:
            digits = "".join(ch for ch in parts[1] if ch.isdigit())
            if digits:
                return f"AMD Ryzen {digits}"
        return "AMD Ryzen"
    return None


def _derive_group_base_specs(df, group_id):
    """
    Build a small, human-friendly 'base_specs' block for product_groups.json.
    This is for reporting only; it does not influence grouping or variants.
    """
    rows = df[df["group_id"] == group_id]
    if rows.empty:
        return {}

    base = {}

    # form factor / display type (lightweight)
    sub_mode = _safe_mode(rows["sub_category"].astype(str).tolist())
    if sub_mode and sub_mode.lower() == "laptop":
        base["form_factor"] = "laptop"
    elif sub_mode and sub_mode.lower() == "tv":
        # try to infer a common display token from names
        panel_tokens = [normalize_panel_type(r.get("name")) for _, r in rows.iterrows()]
        disp = _safe_mode(panel_tokens)
        if disp:
            base["display_type"] = disp

    # screen size mode (0.1")
    sizes = []
    for _, r in rows.iterrows():
        ax = (r.get("axes") or {}).get("size") or {}
        v = ax.get("screen_inches")
        if v is not None:
            try:
                sizes.append(round(float(v), 1))
            except Exception:
                pass
    if sizes:
        base["screen_size_inches"] = _safe_mode(sizes)

    # processor family (readable label)
    cpus = []
    for _, r in rows.iterrows():
        ax = (r.get("axes") or {}).get("silicon") or {}
        cpus.append(_cpu_readable(ax.get("cpu")))
    fam = _safe_mode(cpus)
    if fam:
        base["processor_family"] = fam

    return base


def _export_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _export_csv(path: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# --------------------------------- pipeline --------------------------------

def run_pipeline(
    *,
    input_csv: str,
    output_dir: str = "output",
    sample = None,
    seed: int = 42,
    use_sbert: bool = True,
    group_threshold: float = 0.82,
    return_scoring_features: bool = False):
    """Run the full pipeline and return a summary dict."""
    t0 = time.time()

    # 1) Load
    df = load_products(input_csv, sample=sample, seed=seed)
    n_rows = len(df)

    # 2) Axis extraction
    modeler = AxisModeler(random_state=seed)
    modeler.fit(df)
    df = df.copy()
    df["axes"] = [modeler.infer_axes(row) for _, row in df.iterrows()]

    # 3) Grouping (human-readable IDs)
    grouper = ProductGrouper(threshold=group_threshold, use_sbert=use_sbert, random_state=seed)
    gresult = grouper.fit_predict(df)
    df["group_id"] = gresult.group_ids

    # 4) Variant building
    variant_ids, variant_records, assignments_df = build_variants(
        df, product_id_col="product_id", group_id_col="group_id", axes_col="axes"
    )
    df["variant_id"] = variant_ids

    # 5) Confidence scoring
    scorer = ConfidenceScorer(use_sbert=use_sbert)
    scorer.fit(df, assignments_df, variant_records)
    enriched = scorer.score_assignments(df, assignments_df, return_features=return_scoring_features)

    # 6) Exports
    os.makedirs(output_dir, exist_ok=True)

    # product_groups.json
    prod_count = enriched.groupby("group_id")["product_id"].nunique().to_dict()
    variant_count = Counter([vr["group_id"] for vr in variant_records])

    # collect brand/family/generation from grouping output
    gen_by_gid, brand_by_gid, family_by_gid = {}, {}, {}
    for rec in gresult.groups:
        gid = rec["group_id"]
        gen_by_gid[gid] = rec.get("generation")
        brand_by_gid[gid] = rec.get("brand")
        family_by_gid[gid] = rec.get("family") or ""

    groups_payload = {"product_groups": []}
    for gid in sorted(prod_count.keys()):
        groups_payload["product_groups"].append({
            "group_id": gid,
            "brand": brand_by_gid.get(gid),
            "family": family_by_gid.get(gid),
            "generation": gen_by_gid.get(gid),
            "base_specs": _derive_group_base_specs(df, gid),
            "variant_count": int(variant_count.get(gid, 0)),
            "product_count": int(prod_count.get(gid, 0)),
        })
    _export_json(os.path.join(output_dir, "product_groups.json"), groups_payload)

    # variants.json
    _export_json(os.path.join(output_dir, "variants.json"), {"variants": variant_records})

    # assignments.csv
    _export_csv(os.path.join(output_dir, "assignments.csv"), enriched)

    # summary.json
    avg_conf = float(np.mean(enriched["confidence"])) if not enriched.empty else 0.0
    summary = {
        "total_products": int(n_rows),
        "total_groups": int(len(prod_count)),
        "total_variants": int(len(variant_records)),
        "products_assigned": int(enriched["product_id"].nunique()),
        "products_unassigned": int(n_rows - enriched["product_id"].nunique()),
        "average_confidence": round(avg_conf, 4),
        "processing_time_seconds": round(time.time() - t0, 2),
        "sbert_used": bool(use_sbert),
        "group_threshold": float(group_threshold),
    }
    _export_json(os.path.join(output_dir, "summary.json"), summary)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Product Hierarchy Classifier")
    parser.add_argument("--input", type=str, required=True, help="Path to input products CSV")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--sample", type=int, default=None, help="Optional N for deterministic sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-sbert", action="store_true", help="Disable SBERT even if available")
    parser.add_argument("--threshold", type=float, default=0.82, help="Grouping cosine threshold in [0,1]")
    parser.add_argument("--return-scoring-features", action="store_true", help="Include per-feature columns in assignments.csv")
    args = parser.parse_args()

    res = run_pipeline(
        input_csv=args.input,
        output_dir=args.output,
        sample=args.sample,
        seed=args.seed,
        use_sbert=(not args.no_sbert),
        group_threshold=args.threshold,
        return_scoring_features=args.return_scoring_features,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()