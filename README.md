# Product Hierarchy Classifier

ML-first system that turns noisy retail feeds into readable product groups, deterministic variants, and per-assignment confidence scores—robust to missing fields, ambiguous descriptions, and vendor quirks.

---

## 📚 Table of Contents

- [🔭 Overview](#-overview)
- [🏗️ Architecture](#-architecture)
  - [Module Interaction](#module-interaction-left--right)
  - [Architecture Diagram](#architecture-diagram)
  - [Design Tenets](#design-tenets)
- [📦 Repository Layout](#-repository-layout)
- [📥 Input & 📤 Output](#-input--output)
  - [Input: CSV Schema](#input-csv-full-schema)
  - [`details` JSON Schema](#details-json-observed-keys-and-handling)
  - [Field Priority & Fallbacks](#field-priority--fallbacks)
  - [Output Artifacts](#output-artifacts-created-in---output)
- [🧱 Pipeline Architecture](#pipeline-architecture)
  - [`src/loaders.py`](#srcloaderspy)
  - [`src/normalize.py`](#srcnormalizepy)
  - [`src/specs_map.py`](#srcspecs_mappy)
  - [`src/extractors.py`](#srcextractorspy)
  - [`src/grouping.py`](#srcgroupingpy)
  - [`src/variants.py`](#srcvariantspy)
  - [`src/scoring.py`](#srcscoringpy)
  - [`src/pipeline.py`](#srcpipelinepy)
- [🚀 Getting Started](#-getting-started)
  - [1) Environment Setup](#1-environment)
  - [2) Run the Pipeline](#2-run-the-pipeline)
- [📈 Confidence Scoring & Thresholding](#-confidence-scoring--thresholding)
- [🧪 Final Performance Snapshot](#-final-performance-snapshot)
- [🧪 Data Quality & Insights](#-data-quality--insights)
- [🌟 Bonus Challenges](#-bonus-challenges)
  - [1) ML-Powered Matching](#1-ml-powered-matching-implemented)
  - [2) Bundle Detection](#2-bundle-detection-implemented)
  - [3) Advanced Analysis](#3-advanced-analysis-implemented)
  - [Reviewer Notes](#reviewer-notes)
  - [References (Code & Docs)](#references-code--docs)
- [🔁 Reproducibility & Determinism](#-reproducibility--determinism)
- [🛠️ Troubleshooting](#-troubleshooting)
- [📊 Additional Analysis](#additional-analysis)
  - [Probe Bundles – Minimal Inspector](#probe-bundles--minimal-inspector)
  - [What This Script Does](#what-this-script-does)
  - [Quick Start](#quick-start)
  - [Input Assumptions](#input-assumptions)
  - [Outputs](#outputs)
  - [Scoring Logic (Transparent)](#scoring-logic-transparent)
  - [CLI Options](#cli-options)
  - [What the Evidence Fields Mean](#what-the-evidence-fields-mean)
  - [Workflow Recommendation](#workflow-recommendation)
  - [Known Limitations](#known-limitations)
  - [Extending to Production](#extending-to-production)
  - [Example: Interpreting a Candidate Row](#example-interpreting-a-candidate-row)
  - [Reproducibility](#reproducibility)

## 🔭 Overview

**Problem.** Retail catalogs mix duplicates, bundles, inconsistent specs, and uneven text quality. Search, merchandising, and analytics need:
- **Product groups** (brand + family + optional generation)
- **Variants** (configuration/size/silicon/packaging)
- **Confidence** to prioritize actions downstream

**Approach.** Favor ML/DL-powered methods (sentence embeddings + small classifiers) with gentle normalizers for weak labels. Avoid brittle, vendor-specific rules. Always assign every product (low-confidence allowed in rare edge cases).

**Dataset.** CSV with the columns described in **Input** below.

---

## 🏗️ Architecture

**Module interaction (left → right):**
1. `loaders.py` – deep text cleaning + JSON/specs parsing → `ml_text`
2. `extractors.py` – weak labels + small ML models → per-row axes
3. `grouping.py` – SBERT/TF-IDF + radius neighbors → readable `group_id`
4. `variants.py` – deterministic serialization → readable `variant_id`
5. `scoring.py` – text cohesion + axis features → confidence & evidence
6. `pipeline.py` – orchestration & exports (JSON/CSV summaries)

**Architecture Diagram**

![Architecture Diagram](assets/architecture.png)

**Design tenets**
- **ML-first:** embeddings for similarity; TF-IDF fallback if SBERT isn’t available.
- **Readable IDs:** no hashes; stable slugs and numeric suffixes (`-2`, `-3`, …) on collisions.
- **Separation of concerns:** each file does one job; the pipeline only wires them together.
- **Determinism:** fixed seeds, stable sorting, predictable suffixing.

---

## 📦 Repository Layout

```
product-hierarchy-classifier/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ __init__.py
│  ├─ loaders.py          # ingestion + deep text cleaning + ml_text
│  ├─ normalize.py        # gentle parsers & tokens (GB/TB→GB, inches, color, cpu/gpu, 1440p, brand, slugify)
│  ├─ specs_map.py        # canonical spec keys (no value parsing)
│  ├─ extractors.py       # weak-labels + TF-IDF/LogReg → axes inference
│  ├─ grouping.py         # embedding-based grouping → readable group_id
│  ├─ variants.py         # deterministic variant_id serialization
│  ├─ scoring.py          # confidence + evidence (cohesion, axes, sizes)
│  └─ pipeline.py         # CLI orchestrator and exporters
└─ output/                # created at runtime
   ├─ product_groups.json
   ├─ variants.json
   ├─ assignments.csv
   └─ summary.json
```

---

## 📥 Input & 📤 Output

### Input: CSV (Full Schema)

The pipeline expects a single CSV containing the following columns. 

Characteristics: noisy text, Unicode artifacts, partial/malformed JSON specs, vendor idiosyncrasies (e.g., “Refurbished”, “Renewed”, “Open Box”). 

Types are **expected** types after ingestion; raw values may be strings and will be normalized in `loaders.py`. 

| Column | Type | Required | Used by | Notes |
|---|---|---:|---|---|
| `product_id` | string | ✅ | all | Unique row id; emitted in outputs. If missing, fallback to `vendor_sku_id` when configured. |
| `seller_id` | string | – | analytics (future) | Kept for quality scoring/diagnostics; not used for grouping in v1. |
| `category` | string | – | grouping / sanity | If blank, derived from `primary_category_path`. |
| `sub_category` | string | – | grouping / sanity | Used to constrain neighborhoods (e.g., laptops vs TVs). |
| `name` | string | ✅ | grouping/scoring | Primary title; contributes heavily to `ml_text`. |
| `brief_description` | string | – | grouping/scoring | Cleaned, HTML-unescaped (e.g., `&nbsp;`), merged into `ml_text`. |
| `details` | **JSON string** | – | loaders/extractors | Parsed to dict. See **details schema** below. Source of most specs. |
| `shipping_time` | string | – | – | Free text (e.g., `", Shipping, Arrives Jun 25, Free"`). Currently ignored. |
| `review_rating` | float | – | analysis (future) | 0–5 scale if present. Not used in grouping. |
| `reviews` | JSON array or string | – | – | Often `"[]"`. Count can be used for analysis; currently ignored. |
| `vendor_sku_id` | string | – | ids/joins | External id; stored for traceability. |
| `data_refreshed_at` | ISO datetime | – | logs/summary | Ingestion freshness; not used in modeling. |
| `created_at` | ISO datetime | – | analysis (future) | Product creation time. |
| `is_active` | bool/str | – | analysis (future) | `"TRUE"/"FALSE"` normalized to boolean. We still assign inactive rows. |
| `out_of_stock` | bool/str | – | analysis (future) | Stock status; not used in grouping. |
| `updated_at` | ISO datetime | – | logs/summary | Last product update time. |
| `brand` | string | – | grouping/extractors | Conservatively normalized (`normalize_brand`). If blank, fallback to `details.brand`. |
| `model` | string | – | metadata | Carried through as metadata only; not used for decisions. |
| `upc` | string | – | tie-break (future) | Optional unique code; reserved for stricter dedupe. |
| `category_paths` | JSON array of strings | – | grouping/sanity | Alternative taxonomy paths. Parsed if provided. |
| `primary_category_path` | string | – | grouping/sanity | Main taxonomy string like `electronics/computers/laptops`.|

**Important ingestion notes**

- Column names are trimmed; trailing spaces (e.g., `primary_category_path `) are normalized.
- Booleans like `"TRUE"/"FALSE"` are coerced to true booleans.
- `details` is parsed safely; failures leave `details_parsed=None` without crashing the run.
- All text contributing to `ml_text` undergoes: Unicode NFKC, control-char stripping, HTML unescape, dash/quote normalization, whitespace squeeze, acronym-preserving casing.

#### `details` JSON (observed keys and handling)

Example (truncated): 

```json
{
  "url": "https://www.walmart.com/ip/...",
  "brand": "Apple",
  "color": "Silver",
  "image": "https://...jpeg",
  "model": "MD760LL/A",
  "images": ["https://...117x117.jpeg", "..."],
  "pricing": {
    "sale": 186,
    "onSale": false,
    "regular": 186,
    "savings": {"amount": 0, "percentage": 0},
    "currency": "USD"
  },
  "shipping": {"info": ", Shipping, Arrives Jun 20, Free", "available": true, "freeShipping": false},
  "specifications": {
    "Brand": "Apple",
    "Model": "MD760LL/A",
    "Edition": "Air",
    "Features": "Backlit Keyboard",
    "Condition": "Restored: Like New",
    "RAM Memory": "4 GB",
    "Screen Size": "3 in",
    "Battery Life": "12 h",
    "Processor Type": "core_i5",
    "Processor Brand": "Intel",
    "Processor Speed": "1.3 GHz",
    "Hard Drive Capacity": "128 GB",
    "Solid State Drive Capacity": "128 TB",
    "Operating System": "Mac OS"
  },
  "sellerProductId": "984065346",
  "customerTopRated": true,
  "productVariations": [
    {"price": 186, "options": {"Hard Drive Capacity": "128 GB"}, "product_id": "3BWQGFVID9KM"},
    {"price": 265, "options": {"Hard Drive Capacity": "256 GB"}, "product_id": "5DJ7UO4ZPJO0"}
  ],
  "customerReviewCount": 682
}
```

- `specifications` is mapped to canonical keys via `specs_map.py` (e.g., `"RAM Memory"` → `ram_gb`, `"Screen Size"` → `screen_inches`).  
- Value parsers in `normalize.py` handle units and tokens: GB/TB → GB, inches, CPU/GPU tokens, 1440p, panel type, condition, color.  
- Outliers are tolerated and penalized downstream (e.g., `"Screen Size": "3 in"`, `"SSD Capacity": "128 TB"`).  
- Pricing/shipping/images are not used for grouping; they remain available for analytics/exports.

**Field priority & fallbacks**

- **Brand:** `brand` column → `details.brand` → inferred from title tokens (conservative).  
- **Model:** carried as metadata; never used to **force** a merge.  
- **Category/Sub:** `sub_category` preferred; else parsed from `primary_category_path`.  
- **ID:** `product_id` preferred; `vendor_sku_id` retained as external reference.

---

### Output artifacts (created in `--output`)

#### `product_groups.json`

```json
{
  "product_groups": [
    {
      "group_id": "brand_family_slug_2024",
      "brand": "brand",
      "family": "family phrase",
      "generation": "2024",
      "base_specs": {"display_type": "oled", "screen_size_inches": 65.0},
      "variant_count": 7,
      "product_count": 19
    }
  ]
}
```

#### `variants.json`

```json
{
  "variants": [
    {
      "variant_id": "brand_family_2024/config:16gb_512gb_silver/size:15.6/silicon:intel_i7_13700h/packaging:open_box",
      "group_id": "brand_family_2024",
      "axes": {
        "config": {"ram_gb": 16, "storage_gb": 512, "color": "silver"},
        "size": {"screen_inches": 15.6},
        "silicon": {"cpu": "intel_i7_13700h"},
        "packaging": {"condition": "open_box"}
      },
      "product_count": 3
    }
  ]
}
```

#### `assignments.csv`

```
product_id,group_id,variant_id,confidence,evidence[,feature columns...]
```

- `evidence` is a comma-separated tag set (e.g., `variant_text_good,axes_partial,group_popular`).

#### `summary.json`
- Aggregate metrics: counts, average confidence, elapsed time, SBERT usage, threshold.


---

# Pipeline Architecture

File-by-file responsibilities and the rationale behind each component.

---

### `src/loaders.py`

**What**  
Robust ingestion and deep text cleaning.

**How**  
- Unicode normalization, control-character stripping  
- HTML unescape / tag scrub, dash/quote unification, whitespace squeeze  
- Smart casing for “shouty” strings  
- Safe JSON parsing of `details` → `details_parsed`  
- Flatten vendor specs into `specs` and lowercase `specs_lc`  
- Construct `ml_text` for downstream ML tasks

**Why**  
Cleaner text → stronger embeddings → more reliable grouping and inference.

**Contributes**  
High-quality text features and a consistent schema for downstream modules.

---

### `src/normalize.py`

**What**  
Gentle normalizers (not business rules).

**How**  
- Parse RAM/Storage to **GB**  
- Parse inches with sanity guards  
- Normalize color tokens, CPU/GPU tokens, resolution (incl. 1440p)  
- Brand canonicalization; readable `slugify`

**Why**  
Provide weak labels and interpretable tokens while keeping logic minimal to avoid overfitting.

**Contributes**  
Seeds for ML extractors and readable IDs.

---

### `src/specs_map.py`

**What**  
Canonical key mapping for noisy vendor specification keys.

**How**  
Map variants like “system memory (ram)” → `ram_gb` without parsing values.

**Why**  
Unify signal sources for weak labels and diagnostics.

**Contributes**  
Stable lookup layer (used by extractors).

---

### `src/extractors.py`

**What**  
Weak supervision + small ML models for axis inference.

**How**  
- Build weak labels from `normalize.py` + `specs_map.py`  
- Train per-attribute `LogisticRegression` over TF-IDF  
- Infer missing axes from `ml_text` with graceful fallback to parsers

**Targets**  
RAM (GB), Storage (GB), Screen size (0.5″ bins), Color, CPU token (GPU via gentle parse for now).

**Why**  
Fill missing fields robustly and reduce reliance on brittle regex.

**Contributes**  
`axes` dict per row: `config` / `size` / `silicon` / `packaging`.

---

### `src/grouping.py`

**What**  
Embedding-based product-family grouping with human-readable IDs.

**How**  
- Brand / sub-category / size blocking  
- SBERT (MiniLM) or TF-IDF embeddings  
- Cosine radius neighbors (`threshold = 0.82`) → connected components  
- Medoid title → `{brand}_{family_slug}_{year?}` with stable numeric suffixes

**Why**  
Resilient to noise; avoids over-merging; deterministic, readable IDs.

**Contributes**  
`group_id` per row and group metadata (brand, family, generation).

---

### `src/variants.py`

**What**  
Deterministic, readable `variant_id` composition.

**How**  
Serialize only present axes in a fixed order; skip empties and `bundle=False`:

```
{group_id}/[config:...]/[size:...]/[silicon:...]/[packaging:...]
```

**Why**  
Stable IDs for analytics/exports with zero hashes and minimal proliferation.

**Contributes**  
`variant_id` per row, `variants.json`, and the base assignments table.

---

### `src/scoring.py`

**What**  
Confidence scoring and evidence tags.

**How**  
- Text cohesion to variant/group centroids (SBERT/TF-IDF)  
- Axis presence and within-variant consistency  
- Cohort sizes with saturating transforms  
- Sanity penalties; optional LR calibrator when labels exist

**Why**  
Provide interpretable confidence for gating downstream usage.

**Contributes**  
`assignments.csv` with `confidence` and `evidence`.

---

### `src/pipeline.py`

**What**  
CLI orchestrator: **load → extract → group → variants → score → export**.

**Exports**
- `output/product_groups.json` — groups with base specs & counts  
- `output/variants.json` — variant records  
- `output/assignments.csv` — product → (`group_id`, `variant_id`) + confidence & evidence  
- `output/summary.json` — metrics obtained after running the pipeline


## 🚀 Getting Started

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Core deps: `pandas`, `numpy`, `scikit-learn`, `sentence-transformers` (optional; TF-IDF fallback if unavailable).

### 2) Run the Pipeline

```bash
python -m src.pipeline --input /path/to/products-export.csv --output ./output
```

**Useful flags**
- `--sample N` – deterministic subsample for development.
- `--no-sbert` – disable SBERT and use TF-IDF embeddings.
- `--threshold 0.82` – grouping cosine threshold(see below)
- `--return-scoring-features` – add feature columns to `assignments.csv`.

**Input expectations**
- CSV with at least: product_id, name, details (JSON), brand, model, category, sub_category.
-	details may include {"specifications": { ... }} (nested dict is parsed if present).
-	Other columns are preserved and safely ignored by core logic.


---

## 📈 Confidence Scoring & Thresholding

How confidence is computed

**Features per assignment**
-	Variant similarity (text): cosine to variant centroid
-	Group similarity (text): cosine to group centroid
-	Axis presence: fraction present among {ram, storage, color, screen, cpu}
-	Axis consistency: within-variant cohesion (mode_frac on numeric axes)
-	Cohort sizes: saturating transforms of variant/group sizes
-	Sanity penalties: e.g., out-of-range screen size

**Default score** = deterministic weighted sigmoid; optional LR calibrator can learn weights from labels.

**`group_threshold = 0.82` (grouping).**
-	Where applied: cosine radius neighbors on embeddings within brand/sub/size blocks; components become groups.
-	Why 0.82: tuned to balance precision/recall over validation batches for laptops and TVs:
-	Lowers over-grouping risk (families don’t collapse together),
-	Maintains high coverage (few isolated singles),
-	Empirically supports average confidence ≈ 0.8463.
-	Changing the threshold:
-	Higher → stricter grouping (more groups, fewer members/group),
-	Lower → looser grouping (risk of merging families).
-	Fallback: Even when embeddings fall back to TF-IDF, the same thresholding logic applies; blocks protect against catastrophic merges.


---

## 🧪 Final Performance Snapshot

```json
{
  "total_products": 6529,
  "total_groups": 1489,
  "total_variants": 6062,
  "products_assigned": 6529,
  "products_unassigned": 0,
  "average_confidence": 0.8463,
  "processing_time_seconds": 84.83,
  "sbert_used": true,
  "group_threshold": 0.82
}
```

**Interpretation**
- 0 unassigned → honors “assign everything” directive.
- Avg confidence ~0.85 → reliable matching for decisioning/gating.
- Groups vs variants → strong compression without excessive merging.

---

## 🧪 Data Quality & Insights

**Common issues handled**
- Random unicode/control characters, stray backslashes, HTML remnants → normalized in loader.
- Noisy or inconsistent spec keys → mapped to canonical names (`specs_map.py`).
- Ambiguous titles or missing fields → recovered via ML inference on `ml_text`.

**Recommendations to vendors**
- Normalize brand strings; avoid embedding condition in brand.
- Provide valid JSON for `details.specifications`.
- Use consistent units (GB/TB, explicit inch values).

**Recommendations for pipeline extensions**
- Expand canonical key map as new vendors appear.
- Add supervised calibration labels to fine-tune confidence weighting.
- Introduce accessory/bundle awareness (see Bonus).

---

## 🌟 Bonus Challenges

Optional, production-oriented extensions that complement the core pipeline. Each item links to its code location and summarizes rationale, approach, and integration points.

---

### 1) ML-Powered Matching (✅ Implemented)

**Purpose.** Robust product grouping using text embeddings; tolerant to noisy vendor feeds.

**Approach.**
- SBERT or TF-IDF embeddings
- Cosine radius-graph with connected components
- Blocking by brand/sub/size to keep neighborhoods relevant
- Deterministic medoid-based naming

**Code.** [`src/grouping.py`](src/grouping.py)  
**Integration.** Already part of the main pipeline; tunable via `--no-sbert` and `--threshold`.

**Future refinement.** Optional supervised pairwise model for hard negatives.

---

### 2) Bundle Detection (🕓 Implemented)

Flag “main item + accessory” kits and multi-packs so variants don’t over-merge.

**Signals.**
- Title phrases: `bundle`, `kit`, `combo`, `with`, `includes`, `+`, `x2`, `2-pack`
- Specs: `Accessories Included`
- Variations: edition/options like “+ 256GB Micro SD Card”
- Context: `sub_category` anchors the main item (e.g., laptop, TV)

**Method.**
- High-precision regex seeds → weak labels  
- Tiny TF-IDF + LogisticRegression classifier (falls back to rules if sklearn unavailable)  
- Extract accessories and `pack_qty`; infer `main_product`

**Outputs.**
- Columns: `is_bundle` (bool), `bundle_type` (`accessory_bundle|multi_pack|standalone`),  
  `bundle_confidence` (0–1), `main_product`, `accessories` (list), `pack_qty`, `bundle_evidence`
- Optional: set `axes["packaging"]["bundle"]=True` for downstream variant serialization

**Code.** [`src/bundle_detection.py`](src/bundle_detection.py)  
**Integration.**
```python
# after axis extraction, before variants
from src.bundle_detection import BundleDetector, apply_to_axes
det = BundleDetector()
det.fit(df)                     # trains from weak labels (optional if small data)
bundles = det.detect(df)        # -> columns: is_bundle, bundle_type, accessories, ...
df = apply_to_axes(df, bundles) # toggles axes.packaging.bundle=True when bundled
```

---

### 3) Advanced Analysis (🕓 Implemented)

Brand/seller data-quality scoring and anomaly surfacing to triage vendor issues.

**What it checks.**
- Axes completeness (RAM/storage/screen)
- Sanity: plausible RAM/SSD bounds; valid screen range per sub-category
- Text hygiene (residual artifacts)
- Assignment confidence (as a soft penalty)

**Method.**
- Row score starts at 100; subtract additive penalties (transparent rules)
- Aggregate by brand and seller
- Emit anomaly slice (rows with severe flags or low DQ score)

**Artifacts.**
- `rows_scored.csv`, `brand_quality.csv`, `seller_quality.csv`, `anomalies.csv`

**Code.** [`analysis/analysis.brand_seller_quality_report.py`](analysis/analysis.brand_seller_quality_report.py)
**CLI.**
```bash
python -m analysis.analysis.brand_seller_quality_report   --input path/to/products.csv \         # enriched CSV preferred (axes/specs parsed)
  --assign output/assignments.csv   --outdir output/analysis
```

---

### Reviewer Notes

- **Why this matters.** Bundles/multi-packs distort grouping and variant IDs; DQ scoring highlights where low confidence and implausible specs originate (specific brands/sellers).  
- **Safety valves.** All modules are optional; the main pipeline behavior is unchanged if it is skipped.  
- **Reproducibility.** Deterministic fallbacks (rule-only paths) and fixed seeds keep outputs stable across runs.

---

### References (Code & Docs)

- Grouping (embeddings + radius graph): [`src/grouping.py`](src/grouping.py)  
- Bundle detection: [`src/bundle_detection.py`](src/bundle_detection.py)  
- Advanced analysis report: [`analysis/analysis.brand_seller_quality_report.py`](analysis/analysis.brand_seller_quality_report.py)  
- Probe Bundles – Minimal Inspector: [`analysis/analysis.probe_bundles_min.py`](analysis/analysis.probe_bundles_min.py)  
- (Optional) Exploratory notebooks: `notebooks/` (pattern mining, data statistics)

---

*Note:* The files `bundle_detection.py` and `analysis.brand_seller_quality_report.py` are referenced above and should live in `src/` and `analysis/` respectively.

---

## 🔁 Reproducibility & Determinism

- Fixed seeds for models and sampling.
- Sorted processing; stable medoid selection; predictable `-2`, `-3` suffixing.
- SBERT unavailability triggers TF-IDF with identical decision style.

---

## 🛠️ Troubleshooting

- SBERT not installed → pipeline logs warning; TF-IDF path activates automatically.
- Weird casing/unicode visible → add acronyms/tokens to loader if needed.


# ADDITIONAL ANALYSIS


## Probe Bundles – Minimal Inspector

This README documents **`analysis/probe_bundles_min.py`**, a small, dependency-light script to **surface likely bundle listings** and **summarize accessory terms** from  products CSV. It’s designed for **quick dataset triage** before we wire a full bundle-detection module into the main pipeline.

---

## What this script does

1. **Scores each row’s “bundle-ness”** using transparent signals:
   - Trigger phrases: `with`, `includes`, `bundle`, `combo`, `kit`, `package`, `set`, `w/`, `+`, `&`
   - Accessory lexicon hits: `bag`, `mouse`, `soundbar`, `sleeve`, `dock`, `hub`, `keyboard`, …
   - Multi-pack patterns: `2-pack`, `3pk`, `set of 2`, `x2`
   - Spec hints (if present): `Accessories Included`, `Package Contents`, `Included`, `In the Box`

2. **Outputs three CSVs** to an output folder:
   - `bundle_candidates.csv` – top-N rows by score (with evidence columns)
   - `non_bundle_sample.csv` – N sampled rows from the **low-score** region
   - `accessory_term_counts.csv` – frequency table of matched accessory terms

3. **Prints a concise preview** of the top candidates, a sample of non-bundles, and the top accessory terms.

> ⚠️ This script is for **inspection & sampling**, not final labeling.

---

## Quick start

```bash
# From repo root (ensure pandas is installed)
python analysis/probe_bundles_min.py \
  --input path/to/products.csv \
  --outdir output/probe_min \
  --top 60 \
  --sample-non 60 \
  --extra-lexicon "dock,stylus,stand,pen,sleeve,screen protector"
```

**Requirements:** Python 3.8+ and `pandas`.

---

## Input assumptions

 CSV should include (or the script will create empty fallbacks):

- `product_id`
- `name`
- `brief_description`
- `details` (stringified JSON; if present, we read `details["specifications"]`)

Example `details` structure (subset):

```json
{
  "brand": "ASUS",
  "specifications": {
    "Accessories Included": "Adapter",
    "Screen Size": "15.6 in",
    "...": "..."
  }
}
```

---

## Outputs

### 1) `bundle_candidates.csv` (top-N by score)

Columns:
- `product_id`, `name`, `brief_description`
- `bundle_score` – numeric score (see formula below)
- `evidence_triggers` – which bundle phrases matched
- `evidence_accessories` – accessory terms matched
- `evidence_multipack` – multipack regex matches (with captured quantities)
- `evidence_spec_hint` – `true` if spec fields like *Accessories Included* were present

### 2) `non_bundle_sample.csv` (from low-score tail)

Same columns as above; sampled from the **bottom 40%** by score using `random_state=42`.

### 3) `accessory_term_counts.csv`

Two columns:
- `term` – matched accessory term (normalized)
- `count` – frequency across all rows

---

## Scoring logic (transparent)

For each row we compute a **bundle score**:

- **+2.0** per unique **trigger phrase** matched  
  *(e.g., `with`, `includes`, `bundle`, `combo`, `kit`, `package`, `set`, `w/`, `+` between words, `&`)*
- **+1.0** per unique **accessory term** matched (capped at 6 per row)  
  *(e.g., `bag`, `mouse`, `keyboard`, `soundbar`, `sleeve`, `dock` … plus  `--extra-lexicon`)*
- **+2.0** per **multipack** match  
  *(e.g., `\b(\d+)[-\s]?(?:pack|pk)\b`, `\bset of (\d+)\b`, `\bx\s?(\d+)\b`)*
- **+2.0** if **spec hints** appear in `details.specifications`  
  *(any of: `Accessories Included`, `Package Contents`, `Included`, `In the Box`)*

This yields an interpretable score; **higher ⇒ more likely a bundle or multi-pack**.

> Heuristic guidance (adjust after reviewing  data):  
> • **≥ 4** often reads like a bundle or multi-pack.  
> • **2–3** are “borderline” (e.g., one trigger + a couple accessories).  
> • **0–1** are good **non-bundle** candidates.

---

## CLI options

```text
--input           Path to products CSV (required)
--outdir          Output directory for CSVs (required)
--top             How many bundle candidates to keep (default: 50)
--sample-non      How many non-bundles to sample (default: 50)
--extra-lexicon   Comma-separated accessory terms to add at runtime
```

**Tip:** Use `--extra-lexicon` to quickly add terms that are present in catalog (e.g., `"stylus,pen,monitor arm,lapdesk"`).

---

## What the evidence fields mean

- **evidence_triggers**  
  Phrases or symbols that imply composition: `with`, `includes`, `combo`, `kit`, `package`, `set`, `w/`, `+`, `&`.
  *Note:* `+` is only counted when it appears **between word characters** (`\w + \w`) to avoid math-like noise.

- **evidence_accessories**  
  Normalized terms from the accessory lexicon that appear in the title/description/spec hints (whole-word matches; handles bigrams like `mouse pad`).

- **evidence_multipack**  
  Multipack patterns captured with the matched quantity (e.g., `(\d+)[- ]?(pack|pk)`, `set of (\d+)`, `x(\d+)`).

- **evidence_spec_hint**  
  `true` if any of these spec keys exist: `Accessories Included`, `Package Contents`, `Included`, `In the Box`.

---

## Workflow recommendation

1. **Run the script** on  export.  
2. **Inspect `bundle_candidates.csv`** from the top; mark a few rows as true bundles vs false positives.  
3. **Open `accessory_term_counts.csv`** and identify strong terms to add via `--extra-lexicon`.  
4. **Re-run** and repeat until precision is acceptable.  
5. Share the shortlists; we’ll convert validated rules into a production `bundle_detection.py`, or train a small classifier seeded by these weak labels.

---

## Known limitations

- Heuristic triggers may flag legitimate single-SKU products containing words like **“set”** (“TV set”), or **“with”** in marketing copy. Manual review of the top list is expected.
- Accuracy depends on the **quality of titles/specs**. If accessories are only visible in images, heuristics won’t see them.
- The script doesn’t alter variant IDs or grouping; it’s **exploratory**.

---

## Extending to production

- **Rules approach:** move the refined triggers and lexicon into `bundle_detection.py` with unit tests and deterministic tagging (`bundle_type`, `main_product`, `accessories`).
- **Model approach:** label 300–800 rows from the candidates/non-bundles and train a lightweight classifier (e.g., TF-IDF + LR) to score bundles more robustly.

---

## Example: interpreting a candidate row

- `bundle_score = 7.0`  
- `evidence_triggers = "with, +"`  
- `evidence_accessories = "soundbar, remote"`  
- `evidence_multipack = ""`  
- `evidence_spec_hint = true`

**Interpretation:** Title mentions *with* or uses `+` between items; accessories like `soundbar` and `remote` are present; specs include a “what’s in the box” field. This is a **strong bundle**.

---

## Reproducibility

- **Non-bundle sampling** uses `random_state=42`.  
- Text cleaning uses NFKC normalization, HTML tag stripping, and whitespace squashing for consistent matching.

---

