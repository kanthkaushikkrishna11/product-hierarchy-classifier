#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_bundles_min.py

Quick probe to find likely bundles vs. non-bundles and to extract accessory term frequencies.

What it does
------------
1) Scores each row's "bundle-ness" using:
   - Trigger phrases (with, includes, bundle, combo, kit, package, set, +, w/, etc.)
   - Accessory lexicon hits (bag, mouse, soundbar, sleeve, dock, ...)
   - Multi-pack patterns (2-pack, set of 3, x2, 3pk)
   - Specs hints (e.g., 'Accessories Included', 'Package Contents') when available

2) Writes three CSVs to --outdir:
   - bundle_candidates.csv — top-N rows by score (with evidence columns)
   - non_bundle_sample.csv — N rows sampled from the low-score region
   - accessory_term_counts.csv — frequency of matched accessory terms

3) Prints a short preview to stdout.

Usage
-----
python analysis/probe_bundles_min.py \
  --input path/to/products.csv \
  --outdir output/probe_min

Optional:
  --extra-lexicon "dock,stylus,stand,pen,sleeve,screen protector"
  --top 60 --sample-non 60
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd


# ------------------------------ helpers ------------------------------ #

def _clean_text(s):
    """Light, defensive text normalize: NFKC + lowercase + squeeze whitespace."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\x00", " ")
    s = re.sub(r"[\u200B-\u200D\u2060\uFEFF\u00AD]", "", s)  # zero-widths
    s = s.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ").replace("\\", " ")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _safe_json_dict(x):
    """Parse JSON string to dict; return {} if it fails or isn't dict."""
    if isinstance(x, dict):
        return x
    if not isinstance(x, str) or not x.strip():
        return {}
    try:
        obj = json.loads(x)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        # extremely common: stray trailing commas or bad escapes
        try:
            # soften a little: drop control chars
            y = re.sub(r"[\x00-\x1F\x7F]", " ", x)
            obj = json.loads(y)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def _specs_from_details(details):
    """Return details['specifications'] if dict; else {}."""
    d = _safe_json_dict(details)
    sp = d.get("specifications")
    return sp if isinstance(sp, dict) else {}


# ------------------------------ config ------------------------------ #

# Bundle trigger phrases (regex-safe, all lowercase)
TRIGGERS = [
    r"\bwith\b",
    r"\bincludes?\b",
    r"\bbundle\b",
    r"\bcombo\b",
    r"\bkit\b",
    r"\bpackage\b",
    r"\bset\b",
    r"\bw\/\b",               # "w/"
]

# Simple symbol triggers (treated separately)
SYMBOL_TRIGGERS = [
    r"\+",                    # "TV + Soundbar"
    r"\&",                    # "Keyboard & Mouse"
]

# Multi-pack / quantity patterns
MULTIPACK_PATTERNS = [
    r"\b(\d+)[-\s]?(?:pack|pk)\b",   # 2-pack, 3pk
    r"\bset\s+of\s+(\d+)\b",         # set of 3
    r"\bx\s?(\d+)\b",                 # x2, x3
]

# Accessory lexicon (lowercase tokens). Add your frequent terms via --extra-lexicon.
ACCESSORY_LEXICON = {
    "bag", "backpack", "sleeve", "case", "cover",
    "mouse", "mousepad", "mouse pad",
    "keyboard", "combo", "dock", "docking", "hub", "stand", "mount", "stylus", "pen",
    "charger", "adapter", "power adapter", "ac adapter", "charger cable", "cable",
    "hdmi", "usb", "displayport", "vga",
    "headset", "headphones", "earbuds", "earphones", "mic", "microphone",
    "webcam", "camera", "tripod",
    "sd card", "micro sd", "memory card", "flash drive", "usb drive",
    "cleaning kit", "cloth", "screen protector",
    "soundbar", "subwoofer", "speakers", "remote", "antenna",
    "controller", "gamepad",
}


# ---------------------------- scoring logic ---------------------------- #

def score_bundle_like(text, specs, extra_terms):
    """
    Return (score, evidence) where:
      score: numeric bundle-likeness
      evidence: dict of matched triggers/accessories/multipack/spec_hits
    """
    t = _clean_text(text)
    sp = {str(k).lower(): str(v) for k, v in specs.items()} if isinstance(specs, dict) else {}

    # 1) Trigger phrase hits
    trig_hits = []
    for pat in TRIGGERS:
        if re.search(pat, t):
            trig_hits.append(pat.strip("\\b"))
    for pat in SYMBOL_TRIGGERS:
        # consider '+' only if it appears between words/numbers to avoid math noise
        if pat == r"\+":
            if re.search(r"\w\s*\+\s*\w", t):
                trig_hits.append("+")
        else:
            if re.search(pat, t):
                trig_hits.append("&")

    # 2) Accessory lexicon hits (title + a couple of common spec fields)
    lex = set(ACCESSORY_LEXICON) | set(extra_terms)
    # From specs text fields likely to mention included items
    spec_blobs = []
    for key in ("accessories included", "package contents", "included", "in the box", "items included"):
        v = sp.get(key)
        if v:
            spec_blobs.append(_clean_text(v))
    hay = " ".join([t] + spec_blobs)

    acc_hits = []
    # match whole words or common bigrams
    for term in sorted(lex, key=len, reverse=True):
        # turn "mouse pad" (bigram) into a safe regex
        term_pat = r"\b" + re.escape(term) + r"\b"
        if re.search(term_pat, hay):
            acc_hits.append(term)

    # 3) Multipack patterns
    mp_hits = []
    for mp in MULTIPACK_PATTERNS:
        m = re.findall(mp, t)
        if m:
            mp_hits.extend([f"{mp}:{n}" for n in m])

    # 4) Spec hints
    spec_hint = any(k in sp for k in (
        "accessories included", "package contents", "included", "in the box"
    ))

    # Score (simple, transparent)
    score = 0.0
    score += 2.0 * len(set(trig_hits))          # each unique trigger
    score += 1.0 * min(len(set(acc_hits)), 6)   # each unique accessory (cap to avoid runaway)
    score += 2.0 * len(mp_hits)                 # multipack is strong signal
    if spec_hint:
        score += 2.0

    evidence = {
        "triggers": sorted(set(trig_hits)),
        "accessories": sorted(set(acc_hits)),
        "multipack": mp_hits,
        "spec_hint": bool(spec_hint),
        "score": round(score, 2),
    }
    return score, evidence


def gather_text(row):
    """Collect name + brief_description + a couple spec strings."""
    name = _clean_text(row.get("name"))
    desc = _clean_text(row.get("brief_description"))
    details = row.get("details")
    specs = _specs_from_details(details)
    # pull a couple of useful spec strings
    parts = [name, desc]
    for k in ("Accessories Included", "Package Contents", "Included", "In the Box"):
        v = specs.get(k)
        if v:
            parts.append(_clean_text(v))
    return " ".join([p for p in parts if p]), specs


# ------------------------------ main flow ------------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Probe likely bundle rows and accessory term frequencies.")
    ap.add_argument("--input", required=True, help="Path to products CSV.")
    ap.add_argument("--outdir", required=True, help="Output directory for CSVs.")
    ap.add_argument("--top", type=int, default=50, help="How many bundle candidates to keep.")
    ap.add_argument("--sample-non", type=int, default=50, help="How many non-bundles to sample.")
    ap.add_argument("--extra-lexicon", type=str, default="", help="Comma-separated extra accessory terms.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Parse extra terms
    extra_terms = set()
    if args.extra_lexicon.strip():
        extra_terms = {t.strip().lower() for t in args.extra_lexicon.split(",") if t.strip()}

    # Load CSV
    df = pd.read_csv(args.input)
    # Ensure minimum columns exist
    for col in ("product_id", "name", "brief_description", "details"):
        if col not in df.columns:
            df[col] = ""

    # Score rows
    scores = []
    evidences = []
    accessory_counter = Counter()

    for _, row in df.iterrows():
        text, specs = gather_text(row)
        s, ev = score_bundle_like(text, specs, extra_terms)
        scores.append(s)
        evidences.append(ev)
        # Count accessory terms from evidence
        for term in ev["accessories"]:
            accessory_counter[term] += 1

    df = df.copy()
    df["bundle_score"] = scores
    # Flatten a couple of evidence fields for readability/export
    df["evidence_triggers"] = [", ".join(ev["triggers"]) for ev in evidences]
    df["evidence_accessories"] = [", ".join(ev["accessories"]) for ev in evidences]
    df["evidence_multipack"] = [", ".join(ev["multipack"]) for ev in evidences]
    df["evidence_spec_hint"] = [ev["spec_hint"] for ev in evidences]

    # Split candidates vs non-bundles
    df_sorted = df.sort_values("bundle_score", ascending=False)
    bundle_candidates = df_sorted.head(max(1, args.top)).copy()

    # Choose non-bundle sample from the bottom 40% (safer negatives)
    cutoff_idx = int(0.6 * len(df_sorted))
    tail = df_sorted.iloc[cutoff_idx:]
    non_bundle_sample = tail.sample(n=min(len(tail), max(1, args.sample_non)), random_state=42).copy()

    # Export
    path_candidates = os.path.join(args.outdir, "bundle_candidates.csv")
    path_nonbundles = os.path.join(args.outdir, "non_bundle_sample.csv")
    path_terms = os.path.join(args.outdir, "accessory_term_counts.csv")

    cols_export = [
        "product_id", "name", "brief_description", "bundle_score",
        "evidence_triggers", "evidence_accessories", "evidence_multipack", "evidence_spec_hint"
    ]
    (bundle_candidates[cols_export]).to_csv(path_candidates, index=False)
    (non_bundle_sample[cols_export]).to_csv(path_nonbundles, index=False)

    term_rows = [{"term": k, "count": v} for k, v in accessory_counter.most_common()]
    pd.DataFrame(term_rows, columns=["term", "count"]).to_csv(path_terms, index=False)

    # Preview to stdout
    print("\n=== Bundle Candidates (top {}) ===".format(len(bundle_candidates)))
    for _, r in bundle_candidates.head(10).iterrows():
        print(f"- [{r['bundle_score']:.1f}] {str(r['product_id'])[:24]} :: {r['name'][:100]}")

    print("\n=== Non-bundle Sample ({} rows) ===".format(len(non_bundle_sample)))
    for _, r in non_bundle_sample.head(10).iterrows():
        print(f"- [{r['bundle_score']:.1f}] {str(r['product_id'])[:24]} :: {r['name'][:100]}")

    print("\n=== Accessory Term Counts (top 25) ===")
    for term, cnt in accessory_counter.most_common(25):
        print(f"{term:20s}  {cnt}")

    print("\nWrote:")
    print(" -", path_candidates)
    print(" -", path_nonbundles)
    print(" -", path_terms)
    print("\nNext:")
    print("  • Share the top bundle candidates and the accessory_term_counts.csv back with me.")
    print("  • I’ll fold the strong terms into the bundle lexicon and tighten thresholds.")
    print("  • We’ll then wire bundle flags into the pipeline (kept out of variant IDs per your directive).")


if __name__ == "__main__":
    main()