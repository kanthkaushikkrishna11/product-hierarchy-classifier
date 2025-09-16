# src/normalize.py
# -*- coding: utf-8 -*-
"""
Lightweight normalization helpers for weak labels and feature hygiene.
These are not business rules—just gentle, conservative standardizers.
"""

import re
import unicodedata
from decimal import Decimal

# ----------------------------- brand / color ----------------------------- #

# Minimal alias map—avoid overfitting.
BRAND_ALIASES = {
    "hewlett-packard": "hp",
    "hp inc.": "hp",
    "hp (refurbished)": "hp",
    "dell (refurbished)": "dell",
    "samsung®": "samsung",
    "lg electronics": "lg",
    "lgelectronics": "lg",
    "insignia™": "insignia",
    "amazon renewed": "amazon",
}

COLOR_MAP = {
    "space grey": "space_gray",
    "space gray": "space_gray",
    "midnight": "midnight_blue",
    "grey": "gray",
    "rose": "pink",
}


def normalize_brand(raw):
    """Return canonical brand (lowercase) or None."""
    if not raw:
        return None
    s = str(raw).strip().lower()
    s = re.sub(r"^brand:\s*", "", s)
    s = re.sub(r"\s+\(refurbished\)$", "", s)
    return BRAND_ALIASES.get(s, s) or None


def normalize_color(raw):
    """Return underscore color token (e.g., 'space_gray') or None."""
    if not raw:
        return None
    s = str(raw).strip().lower()
    s = COLOR_MAP.get(s, s)
    s = s.replace(" ", "_")
    return s or None


# ----------------------------- text helpers ------------------------------ #

def canonical_title(name):
    """Lowercase, strip noisy punctuation; keep quotes for inch parsing."""
    s = (name or "").lower()
    s = re.sub(r'[^a-z0-9\" ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ----------------------------- numeric parsing --------------------------- #

def parse_storage_to_gb(text):
    """Extract storage in GB (supports TB→GB). Return int or None."""
    if not text:
        return None
    m = re.search(r'(\d+(?:\.\d+)?)\s*(tb|gb)\b', str(text), re.I)
    if not m:
        return None
    val = Decimal(m.group(1))
    unit = m.group(2).lower()
    if unit == "tb":
        val *= Decimal(1024)
    try:
        return int(round(val))
    except Exception:
        return None


def parse_ram_gb(text):
    """Extract RAM in GB. Return int or None."""
    if not text:
        return None
    s = str(text)
    m = re.search(r'(\d+)\s*gb\s*(?:ram|memory|ddr\d)?', s, re.I)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m2 = re.search(r'ram[^0-9]{0,6}(\d+)', s, re.I)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    return None


def parse_screen_inches(text):
    """Extract screen diagonal in inches as float. Return float or None."""
    if not text:
        return None
    s = str(text)
    m = re.search(r'(\d{2,3}|\d{1,2}\.\d)\s*(?:\"|inch|in|-inch)\b', s, re.I)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def in_valid_screen_range(sub_category, inches, title):
    """Soft sanity range: laptops 10–18, TVs 18–130, projector/laser up to 150."""
    if inches is None:
        return False
    sub = (sub_category or "").lower()
    title_low = (title or "").lower()
    if "projector" in title_low or "laser tv" in title_low:
        return 10.0 <= inches <= 150.0
    if sub == "laptop":
        return 10.0 <= inches <= 18.0
    if sub == "tv":
        return 18.0 <= inches <= 130.0
    return 10.0 <= inches <= 130.0


# ----------------------------- silicon tokens ---------------------------- #

# Minimal patterns to seed CPU family tokens (weak labels).
CPU_PATTERNS = [
    # Apple M-series: m1/m2/m3 (+ pro/max)
    (r'\b(m[123])\b(?:\s*(pro|max))?', lambda g: f"apple_{g[0]}{('_'+g[1]) if g[1] else ''}"),
    # Intel Core i-series with optional model (e.g., 1235U)
    (r'\bintel\s*core\s*i([3579])[- ]?(\d{4,5}[A-Za-z]{0,2})?', lambda g: f"intel_i{g[0]}{'_'+g[1].lower() if g[1] else ''}"),
    (r'\bi([3579])[- ]?(\d{4,5}[A-Za-z]{0,2})\b', lambda g: f"intel_i{g[0]}{'_'+g[1].lower() if g[1] else ''}"),
    # AMD Ryzen N with optional model (e.g., 7840U)
    (r'\bryzen\s*([3579])\b(?:\s*(\d{4,5}[A-Za-z]{0,2}))?', lambda g: f"amd_ryzen{g[0]}{'_'+g[1].lower() if g[1] else ''}"),
]


def normalize_cpu(text):
    """Return compact CPU token (e.g., 'intel_i5_1235u', 'apple_m2') or None."""
    if not text:
        return None
    s = str(text).lower()
    for pat, fmt in CPU_PATTERNS:
        m = re.search(pat, s, re.I)
        if m:
            return fmt(m.groups())
    return None


def normalize_gpu(text):
    """Return compact GPU token (e.g., 'nvidia_rtx_4050', 'intel_iris_xe') or None."""
    if not text:
        return None
    s = str(text).lower()
    if "iris xe" in s:
        return "intel_iris_xe"
    m = re.search(r'\brtx\s*(\d{4})\b', s)
    if m:
        return f"nvidia_rtx_{m.group(1)}"
    m = re.search(r'\bgtx\s*(\d{3,4})\b', s)
    if m:
        return f"nvidia_gtx_{m.group(1)}"
    if "radeon" in s:
        m = re.search(r'\bradeon\s*([a-z0-9]+)\b', s)
        if m:
            return f"amd_radeon_{m.group(1)}"
        return "amd_radeon"
    if "10core" in s and "apple" in s:
        return "apple_10core"
    return None


def slugify(text, max_len=48):
    """Lowercase ASCII slug with underscores; max_len enforced; never empty."""
    if text is None:
        return "product"
    s = unicodedata.normalize("NFKD", str(text))
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if max_len and len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "product"


# ------------------------ display-related helpers ------------------------ #

def normalize_resolution(text):
    """
    Map resolution to: '8k' | '4k' | '1440p' | '1080p' | '720p' or None.
    Recognizes tokens and pixel grids (e.g., 3840x2160 → 4k).
    """
    if not text:
        return None
    s = str(text).lower()

    # Direct tokens
    if re.search(r'\b8k\b', s) or re.search(r'\b4320p\b', s):
        return "8k"
    if re.search(r'\b4k\b', s) or re.search(r'\buhd\b', s) or re.search(r'\bultra\s*hd\b', s) or re.search(r'\b2160p\b', s):
        return "4k"
    if re.search(r'\b1440p\b', s) or re.search(r'\bqhd\b', s) or re.search(r'\bwqhd\b', s) or re.search(r'\buwqhd\b', s) or re.search(r'\bultrawide\s*qhd\b', s) or re.search(r'\b2k\b', s):
        return "1440p"
    if re.search(r'\b1080p\b', s) or re.search(r'\bfhd\b', s) or re.search(r'\bfull\s*hd\b', s):
        return "1080p"
    if re.search(r'\b720p\b', s) or re.search(r'\bhd\s*ready\b', s):
        return "720p"

    # Pixel grids (x or ×)
    mm = re.search(r'(\d{3,5})\s*[x×]\s*(\d{3,5})', s)
    if mm:
        try:
            w, h = int(mm.group(1)), int(mm.group(2))
            lo, hi = min(w, h), max(w, h)
            if hi >= 7600 and lo >= 4300:
                return "8k"
            if hi >= 3800 and lo >= 2100:
                return "4k"
            if (1410 <= lo <= 1470) and (2500 <= hi <= 3600):
                return "1440p"  # includes 2560×1440, 3440×1440
            if (1910 <= hi <= 1940) and (1070 <= lo <= 1090):
                return "1080p"
            if (1270 <= hi <= 1295) and (710 <= lo <= 730):
                return "720p"
        except Exception:
            pass
    return None


def normalize_refresh_hz(text):
    """Extract refresh rate in Hz (e.g., 60, 120, 144, 240). Return int or None."""
    if not text:
        return None
    s = str(text).lower()

    m = re.search(r'(\d{2,3})\s*hz\b', s)
    if m:
        try:
            hz = int(m.group(1))
            if 24 <= hz <= 300:
                return hz
        except Exception:
            pass

    m2 = re.search(r'refresh[^0-9]{0,6}(\d{2,3})', s)
    if m2:
        try:
            hz = int(m2.group(1))
            if 24 <= hz <= 300:
                return hz
        except Exception:
            pass
    return None


def normalize_panel_type(text):
    """Return compact panel token (e.g., 'oled', 'mini_led', 'ips_lcd') or None."""
    if not text:
        return None
    s = str(text).lower()

    if "qd-oled" in s or "qd oled" in s or "qdoled" in s:
        return "qd_oled"
    if "oled" in s:
        return "oled"
    if "mini led" in s or "mini-led" in s or "miniled" in s:
        return "mini_led"
    if "micro led" in s or "micro-led" in s or "microled" in s:
        return "micro_led"
    if "qled" in s:
        return "qled"
    if "ips" in s:
        return "ips_lcd"
    if "va" in s and "lava" not in s:
        return "va_lcd"
    if "lcd" in s:
        return "lcd"
    if "led" in s:
        return "led"
    return None


def normalize_condition(text):
    """Return item condition: 'new' | 'refurbished' | 'renewed' | 'open_box' | 'used' or None."""
    if not text:
        return None
    s = str(text).lower()
    if re.search(r'\bnew\b', s):
        return "new"
    if "renewed" in s or "renew" in s:
        return "renewed"
    if "refurb" in s or "remanufactur" in s or "restored" in s or "certified refurbished" in s:
        return "refurbished"
    if "open box" in s or "open-box" in s:
        return "open_box"
    if "used" in s or "pre-owned" in s or "preowned" in s:
        return "used"
    return None