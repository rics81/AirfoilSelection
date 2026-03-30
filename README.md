# ✈ AirfoilSelection

Automated aerodynamic profile selection pipeline.  Downloads polar data from
[airfoiltools.com](http://airfoiltools.com), computes aerodynamic indicators,
applies a geometry pre-filter, ranks profiles via three independent methods
(PCA-Factor Analysis, AHP, AHP-over-PCA) and produces a final ranking via
the **PrOPPAGA** method.

---

## Project structure

```
AirfoilSelection/
├── config.yaml               ← All user-adjustable parameters
├── main.py                   ← Pipeline orchestrator
├── pipeline/
│   ├── __init__.py
│   ├── loader.py             ← Dynamic importer for numbered modules
│   ├── 01_scraper.py         ← Step 1: web scraping → raw_polars.parquet
│   ├── 02_analysis.py        ← Step 2: quality check + indicator computation
│   ├── 03_features.py        ← Step 3: geometry pre-filter → features.parquet
│   ├── 04_ranking_pca.py     ← Step 4a: Factor-Analysis ranking
│   ├── 05_ranking_ahp.py     ← Step 4b: AHP ranking
│   ├── 06_ranking_ahp_pca.py ← Step 4c: AHP-on-PCA ranking
│   └── 07_proppaga.py        ← Step 5: PrOPPAGA final ranking
└── data/
    ├── raw_polars.parquet     ← Cached scrape output (auto-generated)
    ├── features.parquet       ← Filtered feature table (auto-generated)
    └── results/
        ├── ranking_pca.parquet
        ├── ranking_ahp.parquet
        ├── ranking_ahp_pca.parquet
        ├── ranking_proppaga.parquet
        ├── final_summary.parquet
        └── failed_airfoils.parquet
```

---

## Installation

```bash
pip install pandas numpy scipy scikit-learn factor_analyzer \
            requests beautifulsoup4 pyyaml pyarrow pingouin
```

---

## Quick start

```bash
# Full pipeline (will ask whether to re-scrape if cache exists)
python main.py

# Force a fresh scrape every time
python main.py --force-scrape

# Use the cached parquet, skip scraping
python main.py --skip-scrape

# Run only ranking and PrOPPAGA (steps 4 and 5), using cached features
python main.py --steps 4 5

# Use a different config file
python main.py --config my_config.yaml
```

---

## Configuration (`config.yaml`)

### Geometry pre-filter

Edit the `filter` section to constrain which airfoils reach the ranking steps.
Set any value to `null` to disable that bound.

```yaml
filter:
  thickness_min: 8.0    # % chord
  thickness_max: 15.0   # % chord
  camber_min: 0.0       # % chord  (0 = symmetric profiles allowed)
  camber_max: 6.0       # % chord
```

### Reynolds numbers

```yaml
scraper:
  reynolds_numbers:
    - "200000"
    - "500000"
    - "1000000"
```

### Top-N candidates for PrOPPAGA

```yaml
ranking:
  top_n_per_method: 5   # top-5 from PCA + top-5 from AHP + top-5 from AHP-PCA
```

### PrOPPAGA weights

The 33-element weight vector maps to the 33 indicator columns
(11 indicators × 3 Reynolds numbers, ordered `r2*`, `r5*`, `r10*`).
Only the relative magnitudes matter — weights are normalised internally.

```yaml
proppaga:
  weights: [33, 6, 30, ...]  # one value per indicator column
```

---

## Pipeline steps in detail

### Step 1 — Scraper (`01_scraper.py`)

Fetches every airfoil listed on airfoiltools.com:
- Polar curves (α, Cl, Cd, Cdp, Cm) for each Reynolds number.
- Geometry metadata (max thickness %, max camber %) from the detail page.

Results are stored in `data/raw_polars.parquet`.  On subsequent runs the
orchestrator asks whether to re-scrape or reuse the cached file.

### Step 2 — Analysis (`02_analysis.py`)

**Validation** (airfoils failing any check are dropped entirely):
- At least 3 Reynolds entries available.
- All series (α, Cl, Cd, Cm) fully numeric.
- Stall region detected (≥ 7 points after Cl peak).

**Indicators computed per Reynolds number** (prefix `r2`, `r5`, `r10`):

| Indicator        | Description                          | Direction   |
|------------------|--------------------------------------|-------------|
| `max_cl_cd`      | Maximum Cl/Cd ratio                  | higher = ✅  |
| `max_cl_cd_alpha`| α at max Cl/Cd                       | lower  = ✅  |
| `clMax`          | Maximum Cl                           | higher = ✅  |
| `clStdRate`      | Std of ΔCl before stall              | lower  = ✅  |
| `clMeanRate`     | Mean of ΔCl before stall             | higher = ✅  |
| `clStdStallRate` | Std of ΔCl in stall region           | lower  = ✅  |
| `clMeanStallRate`| Mean \|ΔCl\| in stall region        | lower  = ✅  |
| `cdMax`          | Max Cd (α ≥ 0, pre-stall)           | lower  = ✅  |
| `cdStdRate`      | Std of ΔCd (α ≥ 0, pre-stall)      | lower  = ✅  |
| `cdMeanRate`     | Mean of ΔCd (α ≥ 0, pre-stall)     | lower  = ✅  |
| `cmAlphaZero`    | \|Cm\| at α = 0                     | lower  = ✅  |

### Step 3 — Features & geometry filter (`03_features.py`)

Joins the indicator table with geometry metadata scraped in Step 1,
then applies the `filter` bounds from `config.yaml`.  Saves the final
feature table to `data/features.parquet`.

### Step 4 — Rankings

Three independent methods produce ranked lists:

| Module                  | Method        | Key idea                              |
|-------------------------|---------------|---------------------------------------|
| `04_ranking_pca.py`     | Factor-PCA    | Separate L/G factor models, variance-weighted score |
| `05_ranking_ahp.py`     | AHP           | CV-based entropy weights on raw indicators |
| `06_ranking_ahp_pca.py` | AHP-PCA       | AHP applied to joint PCA factor scores |

### Step 5 — PrOPPAGA (`07_proppaga.py`)

1. Collects the top-N candidates from each of the three rankings.
2. Z-score normalises indicators.
3. Multiplies lower-is-better columns by −1.
4. Re-normalises with a second z-score pass.
5. Computes a final score via weighted dot-product (config weights).
6. Saves `ranking_proppaga.parquet` and `final_summary.parquet`.

---

## Output files

| File                          | Contents                                              |
|-------------------------------|-------------------------------------------------------|
| `data/raw_polars.parquet`     | Raw scraped data (1 row per airfoil × Reynolds)       |
| `data/features.parquet`       | Wide indicator table after geometry filter             |
| `results/ranking_pca.parquet` | PCA ranking with scores and ranks                     |
| `results/ranking_ahp.parquet` | AHP ranking                                           |
| `results/ranking_ahp_pca.parquet` | AHP-PCA ranking                                   |
| `results/ranking_proppaga.parquet` | Final PrOPPAGA ranking                           |
| `results/final_summary.parquet`| All four rankings merged for each candidate          |
| `results/failed_airfoils.parquet` | Airfoils removed in Step 2 with failure reason   |

---

## Extending the pipeline

- **Add a new ranking method**: create `pipeline/0X_ranking_new.py` with a
  `run(df_features, cfg) -> pd.DataFrame` function, register it in
  `pipeline/loader.py`, and add it to `step4_rankings()` in `main.py`.
- **Change the geometry filter**: edit `config.yaml → filter` — no code changes needed.
- **Change PrOPPAGA weights**: edit `config.yaml → proppaga.weights`.
- **Add Reynolds numbers**: extend `config.yaml → scraper.reynolds_numbers`
  and add matching prefix entries in `02_analysis.py → REYNOLDS_PREFIX`.
