# analysis/brand_seller_quality_report.py
# -*- coding: utf-8 -*-
"""
Brand/seller data quality scoring + anomaly surfacing.

Run (example)
-------------
python -m analysis.brand_seller_quality_report --input products-export.csv --assign output/assignments.csv --outdir output/analysis

Or import and call build_quality_report(df, assignments_df) from a notebook/script.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.normalize import in_valid_screen_range


# ------------------------------- helpers ------------------------------- #

def _as_str(x):
    return "" if x is None else str(x)


def _axis_val(axes, path):
    """axes path helper: ('config','ram_gb') -> value or None."""
    if not isinstance(axes, dict):
        return None
    a = axes.get(path[0]) or {}
    return a.get(path[1])


def _implausible_ram_gb(v):
    try:
        x = float(v)
        return x < 2 or x > 128
    except Exception:
        return False


def _implausible_storage_gb(v):
    try:
        x = float(v)
        return x < 16 or x > 8192   # flag <16 GB or >8 TB
    except Exception:
        return False


def _screen_bad(subcat, inches, title):
    if inches is None:
        return True
    return not in_valid_screen_range(_as_str(subcat), float(inches), _as_str(title))


def _has_weird_chars(s):
    # crude check for unusual artifacts after loader cleaning
    if not s:
        return False
    return bool(re.search(r"[^\w\s\-\+\.:,/#\(\)\'\"]", s))


# ------------------------------- scoring ------------------------------- #

def _row_issues(row) -> Dict[str, int]:
    """Return binary issue flags for a single row."""
    axes = row.get("axes") or {}
    sub = row.get("sub_category")
    title = row.get("name")

    ram = _axis_val(axes, ("config", "ram_gb"))
    storage = _axis_val(axes, ("config", "storage_gb"))
    screen = _axis_val(axes, ("size", "screen_inches"))

    flags = {
        "missing_brand": int(_as_str(row.get("brand")) == ""),
        "missing_axes_config": int(ram is None and storage is None),
        "missing_screen": int(screen is None),
        "weird_title_chars": int(_has_weird_chars(_as_str(title))),
        "implausible_ram": int(_implausible_ram_gb(ram)),
        "implausible_storage": int(_implausible_storage_gb(storage)),
        "screen_out_of_range": int(_screen_bad(sub, screen, title)),
    }
    return flags


def _score_rows(df: pd.DataFrame, assignments_df: pd.DataFrame) -> pd.DataFrame:
    """Attach issue flags and per-row quality score (0–100)."""
    df = df.copy()
    # join confidence by product_id
    conf_map = {str(r["product_id"]): float(r.get("confidence", 0.0)) for _, r in assignments_df.iterrows()}

    issues = []
    scores = []
    confs = []
    for _, row in df.iterrows():
        pid = str(row.get("product_id"))
        flags = _row_issues(row)
        issues.append(flags)
        c = conf_map.get(pid, 0.0)
        confs.append(c)

        # Start from 100, subtract weighted penalties
        score = 100.0
        score -= 15 * flags["missing_brand"]
        score -= 10 * flags["missing_axes_config"]
        score -= 5 * flags["missing_screen"]
        score -= 5 * flags["weird_title_chars"]
        score -= 20 * flags["implausible_ram"]
        score -= 20 * flags["implausible_storage"]
        score -= 10 * flags["screen_out_of_range"]

        # Confidence adjustment (low conf → reduce)
        if c < 0.5: score -= 10
        elif c < 0.7: score -= 5

        scores.append(max(0.0, min(100.0, score)))

    issue_df = pd.DataFrame.from_records(issues)
    df["dq_score"] = scores
    df["confidence"] = confs
    df = pd.concat([df, issue_df], axis=1)
    return df


def _aggregate(df_scored: pd.DataFrame, by: str) -> pd.DataFrame:
    """Aggregate mean score, confidence, and issue rates by a cohort key."""
    grp = df_scored.groupby(by, dropna=False)
    out = grp.agg({
        "dq_score": "mean",
        "confidence": "mean",
        "missing_brand": "mean",
        "missing_axes_config": "mean",
        "missing_screen": "mean",
        "weird_title_chars": "mean",
        "implausible_ram": "mean",
        "implausible_storage": "mean",
        "screen_out_of_range": "mean",
    }).reset_index()
    out = out.sort_values("dq_score", ascending=True)
    return out


def build_quality_report(df: pd.DataFrame, assignments_df: pd.DataFrame):
    """
    Compute row-level data quality, brand-level, and seller-level summaries.

    Returns
    -------
    (df_scored, brand_report, seller_report, anomalies)
    """
    scored = _score_rows(df, assignments_df)

    brand_report = _aggregate(scored, by="brand")
    seller_report = _aggregate(scored, by="seller_id")

    # anomalies: rows with any severe flags or dq_score < 60
    severe = (
        (scored["implausible_ram"] == 1)
        | (scored["implausible_storage"] == 1)
        | (scored["screen_out_of_range"] == 1)
        | (scored["dq_score"] < 60)
    )
    anomalies = scored.loc[severe, ["product_id", "brand", "seller_id", "sub_category", "name",
                                    "dq_score", "confidence",
                                    "implausible_ram", "implausible_storage", "screen_out_of_range",
                                    "missing_brand", "missing_axes_config", "missing_screen"]].copy()

    return scored, brand_report, seller_report, anomalies


# ------------------------------- CLI entry ------------------------------ #

def _save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to the products CSV (raw or enriched)")
    p.add_argument("--assign", required=True, help="Path to assignments.csv")
    p.add_argument("--outdir", default="output/analysis", help="Where to write CSV reports")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    asn = pd.read_csv(args.assign)

    # If axes/details columns are JSON strings, try to eval safely
    for col in ("axes", "details_parsed", "specs", "specs_lc"):
        if col in df.columns and df[col].dtype == object:
            # very gentle parse
            try:
                df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith("{") else x)
            except Exception:
                pass

    scored, brand_report, seller_report, anomalies = build_quality_report(df, asn)

    _save_csv(scored, os.path.join(args.outdir, "rows_scored.csv"))
    _save_csv(brand_report, os.path.join(args.outdir, "brand_quality.csv"))
    _save_csv(seller_report, os.path.join(args.outdir, "seller_quality.csv"))
    _save_csv(anomalies, os.path.join(args.outdir, "anomalies.csv"))


if __name__ == "__main__":
    main()