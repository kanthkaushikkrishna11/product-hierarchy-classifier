# src/loaders.py
# -*- coding: utf-8 -*-
"""
Load product CSV(s), parse JSON specs, and clean text for downstream ML.

Adds columns:
  - details_parsed : dict
  - specs          : dict
  - specs_lc       : dict
  - model_code     : str | None  (reference-only; NOT used by logic)
  - ml_text        : str         (clean, embedding-friendly)
"""

from __future__ import annotations

import html
import json
import logging
import pathlib
import re
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)

# ----------------------------- path utils ----------------------------- #

def _as_paths(source):
    """
    Accepts: a string path, a Path object, a glob string (e.g., 'data/*.csv'),
             or a list/tuple mixing these.
    Returns: list[pathlib.Path] of existing files (absolute), de-duplicated, stable order.
    """
    def _expand_one(item):
        p = item if isinstance(item, pathlib.Path) else pathlib.Path(str(item))
        s = str(p)

        # Glob: expand within its parent; include only files
        if any(ch in s for ch in "*?[]"):
            base = p.parent if str(p.parent) != "" else pathlib.Path(".")
            return [pp.resolve() for pp in sorted(base.glob(p.name)) if pp.is_file()]

        # Regular path: include only if it's a file
        return [p.resolve()] if p.is_file() else []

    if isinstance(source, (list, tuple)):
        all_paths = []
        for item in source:
            all_paths.extend(_expand_one(item))
    else:
        all_paths = _expand_one(source)

    out, seen = [], set()
    for p in all_paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


# --------------------------- text cleaning ---------------------------- #

# Remove zero-width/invisible and control characters
_ZERO_WIDTH = re.compile(r"[\u200B-\u200D\u2060\uFEFF\u00AD]")
_CTRL = re.compile(r"[\u0000-\u001F\u007F-\u009F]")

# Strip HTML tags after decoding entities
_HTML_TAG = re.compile(r"<[^>]+>")

# Whitespace and punctuation normalizers
_WS = re.compile(r"\s+")
_MULTI_PUNCT = re.compile(r"([!?.,;:])\1{1,}")
_DASHES = dict.fromkeys(map(ord, "–—−"), ord("-"))  # map en/em/minus to hyphen

# Common acronyms to preserve uppercase
_ACRONYMS = {
    "TV", "OLED", "QLED", "IPS", "HDR", "UHD", "FHD", "HD", "MICRO", "QNED",
    "SSD", "HDD", "NVME", "USB", "HDMI", "CPU", "GPU", "RAM", "DDR",
    "GB", "TB", "MHZ", "GHZ", "M1", "M2", "M3", "RTX", "GTX", "RX",
    "WI-FI", "WIFI", "4K", "8K", "2K", "1440P", "1080P", "720P"
}

def _strip_invisible_ctrl(s: str) -> str:
    """Remove zero-width and control characters."""
    s = _ZERO_WIDTH.sub("", s)
    s = _CTRL.sub(" ", s)
    return s

def _normalize_unicode(s: str) -> str:
    """Fold Unicode to NFKC (curly quotes, fullwidth chars, etc.)."""
    return unicodedata.normalize("NFKC", s)

def _dehtml(s: str) -> str:
    """Decode HTML entities and strip tags."""
    s = html.unescape(s)
    s = _HTML_TAG.sub(" ", s)
    return s

def _normalize_punct(s: str) -> str:
    """Unify dashes, collapse repeated punctuation, normalize quotes, drop stray backslashes."""
    s = s.translate(_DASHES)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"').replace("`", "'")
    s = _MULTI_PUNCT.sub(r"\1", s)
    s = s.replace("\\", " ")  # neutralize lone backslashes from feeds
    return s

def _smart_case_token(tok: str) -> str:
    """Title-case or preserve uppercase for acronyms; keep mixed-case tokens as-is."""
    if tok.upper() in _ACRONYMS:
        return tok.upper()
    if any(c.islower() for c in tok) and any(c.isupper() for c in tok):
        return tok  # mixed case (e.g., MacBook)
    letters = [c for c in tok if c.isalpha()]
    if not letters:
        return tok
    if all(c.isupper() for c in letters) or all(c.islower() for c in letters):
        if len(letters) <= 3 and tok.isalpha():
            return tok.upper()  # keep short brands like LG uppercase
        return tok.title()
    return tok

def _prettify_sentence(s: str) -> str:
    """
    Clean and normalize for readability + embeddings:
      NFKC → strip invisibles/controls → de-HTML → normalize punctuation
      → collapse whitespace → smart-case if mostly screaming/lowercase → trim
    """
    if not s:
        return ""
    s = _normalize_unicode(str(s))
    s = _strip_invisible_ctrl(s)
    s = _dehtml(s)
    s = _normalize_punct(s)
    s = _WS.sub(" ", s).strip()

    letters = [c for c in s if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
        lower_ratio = sum(1 for c in letters if c.islower()) / max(1, len(letters))
        if upper_ratio > 0.8 or lower_ratio > 0.8:
            s = " ".join(_smart_case_token(t) for t in s.split(" "))

    return _WS.sub(" ", s).strip(" -_/|")

def _clean_text_field(x) -> str:
    """Convert any value to a cleaned string."""
    if x is None:
        return ""
    return _prettify_sentence(str(x))


# -------------------------- JSON/spec helpers --------------------------- #

def _safe_json_loads(x):
    """
    Parse JSON string into dict. Return {} on failure or if not a dict.
    Do not pre-clean JSON strings (avoid corrupting valid JSON).
    """
    if isinstance(x, dict):
        return x
    if not isinstance(x, str) or x.strip() == "":
        return {}
    try:
        obj = json.loads(x)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _flatten_specs(details):
    """Return details['specifications'] if it is a dict, else {}."""
    sp = details.get("specifications")
    return sp if isinstance(sp, dict) else {}

def _lowercase_keys(d):
    """Lowercase dict keys; keep values as-is. Return {} on error."""
    try:
        return {str(k).lower(): v for k, v in d.items()}
    except Exception:
        return {}

def _clean_model_code(x):
    """Light normalization for reference-only model code."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    return re.sub(r"\s+", " ", s)


# --------------------------- ML text builder ---------------------------- #

_INTERESTING_SPECS = [
    "Processor Brand", "Processor Type", "Processor Model", "Processor Speed",
    "RAM Memory", "System Memory (RAM)", "Hard Drive Capacity", "Solid State Drive Capacity",
    "Screen Size", "Screen Resolution", "Display Technology", "Color", "Condition",
    # lower-case duplicates sometimes seen in feeds
    "processor brand", "processor type", "processor model", "processor speed",
    "ram memory", "system memory (ram)", "hard drive capacity", "solid state drive capacity",
    "screen size", "screen resolution", "display technology", "color", "condition",
]

def _clean_for_text(v) -> str:
    """Clean a value for inclusion in ml_text."""
    return _clean_text_field(v)

def _build_ml_text(row: dict) -> str:
    """Assemble a compact, clean text signal for embeddings/weak supervision."""
    name = _clean_text_field(row.get("name", ""))
    details = row.get("details_parsed") or {}
    top_brand = _clean_text_field(details.get("brand") or row.get("brand") or "")
    top_color = _clean_text_field(details.get("color") or "")
    model_code = _clean_text_field(row.get("model_code") or "")

    specs = row.get("specs") or {}
    parts = [name, top_brand, top_color, model_code]

    for k in _INTERESTING_SPECS:
        v = specs.get(k)
        if v is not None:
            parts.append(_clean_for_text(v))

    text = " ".join(p for p in parts if p).strip()
    return _WS.sub(" ", text)


# ----------------------------- public API ------------------------------ #

def load_products(source, *, sample=None, seed=42, usecols=None, dtype_overrides=None) -> pd.DataFrame:
    """
    Read one or more CSVs and return a cleaned DataFrame.

    Args:
      source: string/Path/glob or a list/tuple of these.
      sample: optional int; take a deterministic random sample of N rows.
      seed: random seed for sampling.
      usecols: optional list of CSV columns to read.
      dtype_overrides: optional dict of pandas dtype overrides.

    Returns:
      DataFrame with added columns: details_parsed, specs, specs_lc, model_code, ml_text.
    """
    paths = _as_paths(source)
    if not paths:
        raise FileNotFoundError(f"No files matched: {source}")

    frames = []
    parse_errors = 0

    for p in paths:
        df = pd.read_csv(p, usecols=usecols, dtype=dtype_overrides)
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # Deterministic sample if requested
    if isinstance(sample, int) and 0 < sample < len(df_all):
        df_all = df_all.sample(n=sample, random_state=seed).reset_index(drop=True)

    # Ensure expected columns exist (avoid KeyErrors downstream)
    for col in ("product_id", "name", "brief_description", "details", "brand", "model", "category", "sub_category"):
        if col not in df_all.columns:
            df_all[col] = ""

    # Parse details JSON (do not pre-clean JSON string)
    details_parsed = []
    for raw in df_all["details"].tolist():
        obj = _safe_json_loads(raw)
        details_parsed.append(obj)
        if not obj and isinstance(raw, str) and raw.strip():
            parse_errors += 1
    df_all["details_parsed"] = details_parsed

    # Extract specs + lowercase view
    df_all["specs"] = df_all["details_parsed"].apply(_flatten_specs)
    df_all["specs_lc"] = df_all["specs"].apply(_lowercase_keys)

    # Clean key text columns
    for col in ("name", "brief_description", "brand", "model", "category", "sub_category"):
        if col in df_all.columns:
            df_all[col] = df_all[col].astype(str).map(_clean_text_field)

    # Reference-only model code
    df_all["model_code"] = df_all["model"].apply(_clean_model_code)

    # Build clean ml_text
    df_all["ml_text"] = df_all.apply(lambda r: _build_ml_text(r.to_dict()), axis=1)

    # Logging (not fatal)
    total = len(df_all)
    if total:
        pct_err = 100.0 * parse_errors / max(1, total)
        logger.info(
            "Loaded %d rows from %d file(s). details JSON parse soft-failures: %d (%.2f%%).",
            total, len(paths), parse_errors, pct_err,
        )

    return df_all