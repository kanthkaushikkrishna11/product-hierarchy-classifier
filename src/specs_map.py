# src/specs_map.py
# -*- coding: utf-8 -*-
"""
Canonicalize noisy spec keys and provide a thin index over vendor specs.

This file maps raw keys to a small set of canonical names.
It does NOT parse or normalize values (that belongs in normalize/extractors).
"""

# ---------------------------- Canonical keys ---------------------------- #
# Keep this list tight and stable; values are left as-is.
CANONICAL_KEYS = [
    # Config axis
    "ram_gb",
    "storage_gb",
    "color",
    # Size axis
    "screen_inches",
    # Silicon axis
    "cpu_brand",      # e.g., intel/amd/apple (raw text)
    "cpu_family",     # e.g., "Intel Core i5", "Ryzen 7", "M2"
    "cpu_model",      # e.g., "i5-1235U" (raw if present)
    "gpu_model",      # e.g., "RTX 4050", "Iris Xe" (raw)
    # TV traits (collected for completeness; do not drive grouping)
    "panel_type",     # OLED/QLED/LED/LCD (raw)
    "refresh_hz",     # "120 Hz" → still raw here
    "resolution",     # "3840x2160" / "4K" (raw)
    # Packaging axis
    "condition",      # New/Refurbished/Renewed (raw)
]

# --------------------------- Synonym dictionary ------------------------- #
# Keys are LOWER-CASED variants as they appear in vendor feeds.
# Values are our canonical names. We only unify names here, not values.
SPEC_KEY_MAP = {
    # ---- RAM ----
    "ram memory": "ram_gb",
    "system memory (ram)": "ram_gb",
    "system memory": "ram_gb",
    "ram": "ram_gb",
    "ram memory installed size": "ram_gb",
    "ram installed size": "ram_gb",
    "ram size": "ram_gb",

    # ---- Storage (total capacity; SSD/HDD variants) ----
    "hard drive capacity": "storage_gb",
    "solid state drive capacity": "storage_gb",
    "hard disk size": "storage_gb",
    "total storage capacity": "storage_gb",
    # some feeds ship snake_case already
    "hard_drive_capacity": "storage_gb",
    "solid_state_drive_capacity": "storage_gb",

    # ---- Screen size ----
    "screen size": "screen_inches",
    "standing_screen_display_size": "screen_inches",
    "display size": "screen_inches",
    "product diagonal": "screen_inches",

    # ---- CPU ----
    "processor brand": "cpu_brand",
    "processor_brand": "cpu_brand",
    "chipset_brand": "cpu_brand",
    "processor type": "cpu_family",
    "processor": "cpu_family",
    "cpu model": "cpu_model",
    "processor model": "cpu_model",
    "processor model number": "cpu_model",
    "processor_model": "cpu_model",

    # ---- GPU ----
    "graphics": "gpu_model",
    "gpu brand": "gpu_model",
    "gpu brand:": "gpu_model",
    "graphics type": "gpu_model",
    "graphics card description": "gpu_model",
    "graphics_coprocessor": "gpu_model",
    "card_description": "gpu_model",

    # ---- Color ----
    "color": "color",

    # ---- Packaging / Condition ----
    "condition": "condition",

    # ---- TV traits (completeness only) ----
    "display technology": "panel_type",
    "display type": "panel_type",
    "television type": "panel_type",
    "refresh rate": "refresh_hz",
    "screen resolution": "resolution",
    "resolution": "resolution",
    "display resolution": "resolution",
    "max_screen_resolution": "resolution",
}

# Build reverse map for diagnostics: canonical -> [synonyms...]
CANON_TO_SYNONYMS = {}
for raw_key, canon in SPEC_KEY_MAP.items():
    CANON_TO_SYNONYMS.setdefault(canon, []).append(raw_key)


# ------------------------------ API ------------------------------------ #

def canonicalize_spec_keys(specs_lc):
    """
    Convert a lowercased spec dict into {canonical_key: [values,...]}.
    Keeps all values for each canonical key (no parsing here).
    """
    if not isinstance(specs_lc, dict):
        return {}
    out = {}
    for raw_key, value in specs_lc.items():
        canon = SPEC_KEY_MAP.get(raw_key)
        if canon is None:
            continue
        out.setdefault(canon, []).append(value)
    return out


class SpecIndexer:
    """
    Read-only view over vendor specs, accessed via canonical names.

    Usage:
      idx = SpecIndexer(specs_raw=row["specs"], specs_lc=row["specs_lc"])
      ram = idx.first("ram_gb")
      all_res = idx.all("resolution")
    """

    __slots__ = ("_raw", "_lc", "_canon_map")

    def __init__(self, specs_raw, specs_lc):
        self._raw = specs_raw or {}
        self._lc = specs_lc or {}
        self._canon_map = canonicalize_spec_keys(self._lc)

    def synonyms(self, canonical_key):
        """List of raw-key synonyms that map to this canonical key."""
        return list(CANON_TO_SYNONYMS.get(canonical_key, []))

    def all(self, canonical_key):
        """All raw values under a canonical key (order follows specs_lc iteration)."""
        return list(self._canon_map.get(canonical_key, []))

    def first(self, canonical_key):
        """First non-empty value for a canonical key, or None."""
        vals = self._canon_map.get(canonical_key, [])
        for v in vals:
            if v not in (None, "", [], {}):
                return v
        return None

    def has(self, canonical_key):
        """True if any value exists for the canonical key."""
        return bool(self._canon_map.get(canonical_key))

    def as_map(self):
        """Entire canonical multi-map (shallow copy)."""
        return dict(self._canon_map)