# Morocco Geospatial Explore — Geo AI Project 1 (Teacher-Style Learning Plan)

## TL;DR

> **Quick Summary**: Build a teacher-style Jupyter notebook + small Python utils that load Morocco admin boundaries (geoBoundaries) and walk the full ladder Geometry → CRS → Projection → Spatial Join → Spatial Analysis → Visualization, covering all 8 required operations with concept-then-code cells.
>
> **Deliverables**:
> - `notebooks/01_morocco_geospatial_explore.ipynb` — main learning notebook (14-16 sections, runs end-to-end)
> - `src/geolearn/` utils (`loaders.py`, `crs_utils.py`, `viz.py`) + `data/major_cities.csv` + cached GeoJSON
> - `maps/` outputs (static PNGs + Folium interactive HTML choropleths)
> - `pyproject.toml` / `requirements` via `uv` + `README.md` with run instructions
>
> **Estimated Effort**: Medium (greenfield, ~12 tasks)
> **Parallel Execution**: YES — 3 waves + FINAL verification
> **Critical Path**: T1 (env) → T2 (data) → T3 (utils) → T7 (sjoin lesson) → T12 (E2E execute + polish) → F1-F4

---

## Context

### Original Request
Learning project, first of Geo AI mastery series. Teacher mode. Python app/notebook loading Moroccan admin boundaries (Regions, Provinces, Communes, Major cities) performing point-in-polygon, spatial joins, distance, buffering, area, CRS transformations, geometric ops, choropleths. Goal: learn how geospatial data actually works. Ladder: Geometry → Point/Line/Polygon → CRS → Projection → Spatial Join → Spatial Analysis → Visualization.

### Interview Summary
**Key Discussions**:
- Format BOTH (notebook + src/ utils) — user chose Both
- Data geoBoundaries simplified GeoJSON — user chose geoBoundaries
- Viz Folium first (+ matplotlib static as baseline) — user chose Folium first
- Teaching concept-then-code per section — user chose Concept then code
- Env uv + venv — user chose uv + venv
- Tests NO automated tests, agent-QA only — user chose No tests

**Research Findings**:
- geoBoundaries MAR ADM0 (158KB) / ADM1 12 regions (245KB) / ADM2 75 provinces (573KB), direct `gpd.read_file(url)`, ODbL, EPSG:4326; no ADM3 communes — fallback: GADM note + hand-built cities CSV
- OCHA COD-AB (HCP official, P-codes, Aug 2026) as cited fallback; GADM v4.1 deepest but non-commercial; Western Sahara separate entity in all sources — explicit lesson callout
- Stack versions: GeoPandas 1.1.4, Shapely 2.1.2 (immutable, `.geoms`, `unary_union`), PyProj 3.7.2 (`always_xy=True`), Folium Choropleth pattern; Morocco UTM 32629 (west: Casa/Rabat/Marrakech) vs 32630 (east: Tangier/Fes/Oujda)
- Pitfalls to teach: area/distance in degrees, missing CRS, invalid geoms, axis order, CRS mismatch

### Metis Review
- Metis agent unavailable (user reported subagent broken). Self-gap analysis substituted:
- Gaps addressed: communes gap (document + optional GADM stretch), cities source (hand-built CSV with 8 cities), Western Sahara note, CRS zone split lesson, env pin (uv), no-test QA via nbconvert + evidence

---

## Work Objectives

### Core Objective
Learner finishes able to explain and demo every rung of the ladder on real Morocco data, with a runnable notebook proving each operation.

### Concrete Deliverables
- `pyproject.toml` (+ `uv.lock` if generated) with pinned geo stack
- `src/geolearn/__init__.py`, `loaders.py`, `crs_utils.py`, `viz.py`
- `data/major_cities.csv` (8 cities: Casablanca, Rabat, Marrakech, Fes, Tangier, Agadir, Oujda, Laayoune-note) + `data/README.md` + cached GeoJSONs
- `notebooks/01_morocco_geospatial_explore.ipynb` with all 8 ops + teacher markdown
- `maps/morocco_regions_static.png`, `maps/morocco_choropleth.html`, `maps/morocco_cities_buffer.html`
- `README.md` (setup with uv, run notebook, data sources + licenses)

### Definition of Done
- [ ] `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/executed.ipynb` exits 0 with zero failed cells
- [ ] All maps/ files exist and open (HTML contains `leaflet`, PNG non-empty)
- [ ] Learner can answer: what is CRS, why reproject before area, what predicate sjoin used

### Must Have
- All 8 ops present as separate notebook sections with concept-then-code + outputs
- geoBoundaries loading via utils (not hardcoded URLs in every cell)
- CRS lesson showing 4326 vs 32629/32630 with numeric before/after proof
- Western Sahara / license / 2015-reform (12 regions) notes
- Folium choropleth + static matplotlib choropleth

### Must NOT Have (Guardrails)
- NO ML/AI modeling, NO deployment configs, NO remote-sensing — later series projects
- NO Kepler.gl build-out (Folium first scope; mention as stretch only)
- NO hardcoded absolute paths, NO API keys, NO commercial-use violation (credit ODbL + GADM note)
- NO human-verification acceptance (all agent-executed)
- NO over-abstraction (max 3 small utils modules, no framework)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO (empty dir)
- **Automated tests**: None (user chose No tests)
- **Framework**: none
- **Primary verification**: `uv run` + `jupyter nbconvert --execute` + file/content assertions + Playwright screenshots for HTML maps

### QA Policy
Every task MUST include agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.
- **Notebook/code**: Bash (`uv run python`, `jupyter nbconvert --execute`) — assert exit 0, assert output values/files
- **Maps HTML**: Bash content check (`grep -i leaflet`) + Playwright screenshot where UI claim made
- **Static PNG**: file-size check + Read-as-image sanity

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — foundation, 4 parallel):
├── T1: uv env + deps + README skeleton [quick]
├── T2: data assets (cities CSV + cached GeoJSON + data README) [quick]
├── T3: src/geolearn utils (loaders + crs_utils + viz) [quick]
└── T4: notebook skeleton + TOC + teacher template [quick]

Wave 2 (After Wave 1 — core lessons, 5 parallel):
├── T5: Lesson Geometry basics (Point/Line/Polygon) [unspecified-high]
├── T6: Lesson CRS + Projection (4326 vs UTM) [unspecified-high]
├── T7: Lesson point-in-polygon + spatial join [deep]
├── T8: Lesson distance + buffer + area (metric) [deep]
└── T9: Lesson geometric ops + validity repair [deep]

Wave 3 (After Wave 2 — viz + E2E, 3 parallel→sequential tail):
├── T10: Static matplotlib maps + choropleth PNG [visual-engineering]
├── T11: Folium interactive + choropleth HTML [visual-engineering]
└── T12: E2E notebook execute + polish + README (depends: T5-T11) [unspecified-high]

Wave FINAL (After ALL — 4 parallel reviews, then user okay):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA incl. notebook execute (unspecified-high)
└── F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T2 → T3 → T7 → T12 → F1-F4 → user okay
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 5 (Wave 2)
```

### Dependency Matrix

- **T1**: None — T12, F1-F4
- **T2**: None — T3, T4, T5-T11, T12
- **T3**: T2 — T5-T11, T12
- **T4**: T2 — T5-T11, T12
- **T5**: T3, T4 — T12
- **T6**: T3, T4 — T7, T8, T12
- **T7**: T3, T4, T6 — T12
- **T8**: T3, T4, T6 — T12
- **T9**: T3, T4 — T12
- **T10**: T3, T4, T5 — T12
- **T11**: T3, T4, T7 — T12
- **T12**: T5-T11 — F1-F4

### Agent Dispatch Summary

- **Wave 1 (4)**: T1 → `quick`, T2 → `quick`, T3 → `quick`, T4 → `quick`
- **Wave 2 (5)**: T5 → `unspecified-high`, T6 → `unspecified-high`, T7 → `deep`, T8 → `deep`, T9 → `deep`
- **Wave 3 (3)**: T10 → `visual-engineering`, T11 → `visual-engineering`, T12 → `unspecified-high`
- **FINAL (4)**: F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. Env setup with uv + pinned geo stack + README skeleton

  **What to do**:
  - Init `pyproject.toml` (requires-python >=3.10) with `geopandas`, `shapely>=2`, `pyproj`, `folium`, `jupyter`, `matplotlib`, `pandas`; run `uv sync`
  - Create `README.md` skeleton (setup, run, data credits) and `.gitignore` (venv, __pycache__, .ipynb_checkpoints, maps/*.html kept)
  - Verify `uv run python -c "import geopandas, shapely, pyproj, folium"` prints versions

  **Must NOT do**:
  - No conda/pip instructions as primary; no Kepler.gl dependency; no version floats

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-config scaffolding, no logic
  - **Skills**: []
  - **Skills Evaluated but Omitted**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: T12 (E2E needs env)
  - **Blocked By**: None (can start immediately)

  **References**:
  - Pattern: none (greenfield) — keep files minimal
  - External: `https://geopandas.org/en/stable/docs/user_guide/projections.html` — why stack versions matter for CRS lesson

  **Acceptance Criteria**:
  - [ ] `uv sync` exits 0 and `uv run python -c "import geopandas"` exits 0

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: stack imports
    Tool: Bash
    Preconditions: fresh checkout
    Steps:
      1. Run `uv sync`
      2. Run `uv run python -c "import geopandas, shapely, pyproj, folium, matplotlib; print('stack ok')"`
      3. Assert output contains "stack ok"
    Expected Result: exit 0 + "stack ok"
    Failure Indicators: ImportError / resolver error
    Evidence: .sisyphus/evidence/task-1-stack-ok.log

  Scenario: no float deps
    Tool: Bash
    Preconditions: pyproject.toml exists
    Steps:
      1. Assert `geopandas`, `shapely>=2`, `pyproj`, `folium` pinned entries exist
    Expected Result: all four present
    Evidence: .sisyphus/evidence/task-1-deps.log
  ```

  **Commit**: YES (groups with 2-4)
  - Message: `chore(geo): env + data + utils skeleton`
  - Files: `pyproject.toml, README.md, .gitignore`
  - Pre-commit: `uv run python -c "import geopandas"`

- [ ] 2. Data assets — cities CSV + cached GeoJSON + data README

  **What to do**:
  - Create `data/major_cities.csv` with columns `city,lon,lat` — Casablanca (-7.5898,33.5731), Rabat (-6.8416,34.0209), Marrakech (-7.9864,31.6295), Fes (-5.0078,34.0181), Tangier (-5.8340,35.7595), Agadir (-9.6000,30.4278), Oujda (-1.9000,34.6814), Dakhla (-15.9378,23.6848, note-disputed)
  - Download (or document + cache) geoBoundaries simplified ADM0/ADM1/ADM2 GeoJSON to `data/` (filenames `mar_adm0.geojson`, `mar_adm1.geojson`, `mar_adm2.geojson`)
  - Write `data/README.md` with source URLs, ODbL credit, Western Sahara note, 2015-reform 12-regions note, GADM-communes stretch note

  **Must NOT do**:
  - No shapefile unzipping complexity; no OSM Overpass fetching; no invented coordinates

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Data curation + docs, small files
  - **Skills**: []
  - **Skills Evaluated but Omitted**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: T3, T4, T5-T11, T12
  - **Blocked By**: None

  **References**:
  - Pattern: none — new files
  - External: `https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/MAR/ADM1/geoBoundaries-MAR-ADM1_simplified.geojson` — ADM1 pattern; swap ADM0/ADM2 for others
  - External: `https://data.humdata.org/dataset/cod-ab-mar` — cite as official fallback

  **Acceptance Criteria**:
  - [ ] `uv run python -c "import geopandas as gpd; print(len(gpd.read_file('data/mar_adm1.geojson')))"` prints `12`
  - [ ] `data/major_cities.csv` has 8 rows + header

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: regions load
    Tool: Bash
    Preconditions: data/ cached
    Steps:
      1. Run `uv run python -c "import geopandas as gpd; g=gpd.read_file('data/mar_adm1.geojson'); print(len(g), g.crs.to_epsg())"`
      2. Assert output starts with "12 4326"
    Expected Result: 12 rows, EPSG 4326
    Failure Indicators: row count != 12 or CRS != 4326
    Evidence: .sisyphus/evidence/task-2-regions.log

  Scenario: cities valid coords
    Tool: Bash
    Preconditions: CSV exists
    Steps:
      1. Run `uv run python -c "import pandas as pd; d=pd.read_csv('data/major_cities.csv'); assert d[['lon','lat']].notna().all().all(); assert ((d.lon>-20)&(d.lon<0)).all(); print('cities ok', len(d))"`
    Expected Result: "cities ok 8"
    Failure Indicators: NaN or out-of-Morocco lon/lat
    Evidence: .sisyphus/evidence/task-2-cities.log
  ```

  **Commit**: YES (groups with 1,3,4)
  - Message: `chore(geo): env + data + utils skeleton`
  - Files: `data/major_cities.csv, data/*.geojson, data/README.md`
  - Pre-commit: `uv run python -c "import geopandas"`

- [ ] 3. Utils — loaders + CRS helpers + viz helpers

  **What to do**:
  - `src/geolearn/__init__.py` (empty + version)
  - `src/geolearn/loaders.py`: `load_regions()`, `load_provinces()`, `load_cities()` returning GeoDataFrames in EPSG:4326 with `set_crs` guard + `assert_crs_match()` helper
  - `src/geolearn/crs_utils.py`: `to_utm_zone(lon)` (lon<-6 → 32630 else 32629), `to_metric(gdf)` reproject, `area_km2()`, `distance_km()` wrappers
  - `src/geolearn/viz.py`: `static_choropleth()` (matplotlib) + `folium_choropleth()` returning Folium Map; teacher docstrings explaining WHY

  **Must NOT do**:
  - No more than 3 modules; no classes/frameworks; no Kepler.gl code

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small pure functions, 1-3 files
  - **Skills**: []
  - **Skills Evaluated but Omitted**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: T5-T11, T12
  - **Blocked By**: T2 (needs data paths)

  **References**:
  - Pattern: `data/major_cities.csv:city,lon,lat` — loader input contract
  - API: `geopandas.GeoDataFrame.set_crs/to_crs`, `gpd.points_from_xy`, `gpd.sjoin` — contracts utils wrap
  - External: `https://pyproj4.github.io/pyproj/stable/api/crs/crs.html` — always_xy pattern

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from geolearn.loaders import load_regions, load_cities; print(len(load_regions()), len(load_cities()))"` prints `12 8`

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: loaders round-trip
    Tool: Bash
    Preconditions: T2 data cached
    Steps:
      1. Run loaders import + print lengths + CRS EPSG
      2. Assert "12 8" and EPSG 4326
    Expected Result: correct counts + CRS
    Failure Indicators: ImportError / wrong counts
    Evidence: .sisyphus/evidence/task-3-loaders.log

  Scenario: metric helper sane
    Tool: Bash
    Preconditions: utils importable
    Steps:
      1. Run `uv run python -c "from geolearn.loaders import load_regions; from geolearn.crs_utils import area_km2; a=area_km2(load_regions()); print(a.sum())"`
      2. Assert total area between 400000 and 800000 km2 (Morocco ~446550 + tolerance for source)
    Expected Result: sum in range
    Failure Indicators: degrees-area nonsense (tiny number) or exception
    Evidence: .sisyphus/evidence/task-3-area.log
  ```

  **Commit**: YES (groups with 1,2,4)
  - Message: `chore(geo): env + data + utils skeleton`
  - Files: `src/geolearn/*`
  - Pre-commit: `uv run python -c "import geolearn"`

- [ ] 4. Notebook skeleton + TOC + teacher template

  **What to do**:
  - Create `notebooks/01_morocco_geospatial_explore.ipynb` with markdown TOC (the ladder), setup cell (`%pip` note + imports + `gdf.crs` check), one placeholder markdown+code pair per future lesson (T5-T11), closing "What you learned" checklist
  - Teacher voice: each placeholder has `Concept → Code → Why it works` headers so later tasks fill code only

  **Must NOT do**:
  - No filled lesson logic (later tasks own that); no hardcoded URLs (use loaders)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Markdown scaffolding + one imports cell
  - **Skills**: []
  - **Skills Evaluated but Omitted**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: T5-T11 (fill slots), T12
  - **Blocked By**: T2 (data paths for setup cell)

  **References**:
  - Pattern: `src/geolearn/loaders.py:load_regions` — setup cell must use it

  **Acceptance Criteria**:
  - [ ] `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/skel.ipynb` exits 0

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: skeleton executes
    Tool: Bash
    Preconditions: T1 env + T2 data
    Steps:
      1. Run nbconvert execute on skeleton
      2. Assert exit 0
    Expected Result: executes clean
    Failure Indicators: ModuleNotFound / missing data path
    Evidence: .sisyphus/evidence/task-4-skeleton.log

  Scenario: TOC complete
    Tool: Bash
    Preconditions: notebook exists
    Steps:
      1. Assert notebook JSON contains "Spatial Join", "Buffer", "Choropleth", "CRS"
    Expected Result: all four strings found
    Evidence: .sisyphus/evidence/task-4-toc.log
  ```

  **Commit**: YES (groups with 1-3)
  - Message: `chore(geo): env + data + utils skeleton`
  - Files: `notebooks/01_morocco_geospatial_explore.ipynb`
  - Pre-commit: `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/skel.ipynb`

- [ ] 5. Lesson — Geometry basics (Point / Line / Polygon with Shapely 2.x)

  **What to do**:
  - Fill notebook section: markdown Concept (what Point/Line/Polygon are + WKT glimpse) → code creating 1 Point (Casablanca), 1 LineString (Casa→Rabat), 1 Polygon (small bbox around Marrakech) with `shapely.Point/LineString/Polygon`, show `.geom_type`, `.bounds`, `.is_valid`
  - Teach immutability + `.geoms` for multi-parts in one callout cell

  **Must NOT do**:
  - No CRS math here (T6 owns it); no sjoin (T7 owns it)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Teaching correctness matters; small but conceptual
  - **Skills**: []
  - **Skills Evaluated but Omitted**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 8, 9)
  - **Blocks**: T12
  - **Blocked By**: T3, T4

  **References**:
  - API: `shapely.Point/LineString/Polygon`, `.bounds/.is_valid/.geom_type` — exact lesson APIs
  - External: `https://shapely.readthedocs.io/en/latest/manual.html` — creation + predicates

  **Acceptance Criteria**:
  - [ ] Section executes; Casablanca Point bounds equal `(-7.5898, 33.5731, -7.5898, 33.5731)` within 1e-6

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: geometry outputs
    Tool: Bash
    Preconditions: notebook section filled
    Steps:
      1. Execute notebook, extract Casablanca point cell output
      2. Assert bounds match Casa lon/lat
    Expected Result: exact Casa coords echoed
    Failure Indicators: swapped lat/lon
    Evidence: .sisyphus/evidence/task-5-geom.log

  Scenario: invalid input guarded
    Tool: Bash
    Preconditions: shapely installed
    Steps:
      1. Run `uv run python -c "from shapely import Polygon; p=Polygon([(0,0),(1,1)]); print(p.is_valid, p.is_empty)"`
      2. Assert graceful (False/True) not exception
    Expected Result: no crash, validity flags shown
    Evidence: .sisyphus/evidence/task-5-invalid.log
  ```

  **Commit**: YES (groups with 6-9)
  - Message: `feat(geo): core lessons geometry through geometric ops`
  - Files: `notebooks/01_morocco_geospatial_explore.ipynb`
  - Pre-commit: `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/executed.ipynb`

- [ ] 6. Lesson — CRS + Projection (EPSG:4326 vs UTM 32629/32630)

  **What to do**:
  - Fill section: Concept (degrees vs meters, why Morocco spans 2 UTM zones) → code `print(regions.crs)`, `to_crs(32629)` vs `to_crs(32630)`, numeric proof: `.geometry.area` in 4326 (tiny degrees²) vs metric area km² after reproject; map zone rule `lon<-6 → 32630 else 32629`
  - Callout: `always_xy=True` + never compute area in degrees

  **Must NOT do**:
  - No distance/buffer math beyond area proof (T8 owns it)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Most-misunderstood concept; needs numeric before/after
  - **Skills**: []
  - **Skills Evaluated but Omitted**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 7, 8, 9)
  - **Blocks**: T7, T8, T12
  - **Blocked By**: T3, T4

  **References**:
  - API: `src/geolearn/crs_utils.py:to_utm_zone/to_metric` — must use helpers
  - External: `https://geopandas.org/en/stable/docs/user_guide/projections.html` — to_crs pattern

  **Acceptance Criteria**:
  - [ ] Notebook shows degrees-area < 100 and metric total 400000–800000 km² side by side

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: crs proof numbers
    Tool: Bash
    Preconditions: section filled
    Steps:
      1. Execute notebook, assert degrees-area cell output < 100 and metric cell in range
    Expected Result: both numbers present, contrast obvious
    Failure Indicators: metric ≈ degrees (forgot to_crs)
    Evidence: .sisyphus/evidence/task-6-crs.log

  Scenario: zone rule
    Tool: Bash
    Preconditions: crs_utils present
    Steps:
      1. Run `uv run python -c "from geolearn.crs_utils import to_utm_zone; print(to_utm_zone(-7.58), to_utm_zone(-3.0))"`
      2. Assert "32629 32630"
    Expected Result: west 32629, east 32630
    Evidence: .sisyphus/evidence/task-6-zones.log
  ```

  **Commit**: YES (groups with 5,7-9)
  - Message: `feat(geo): core lessons geometry through geometric ops`
  - Files: `notebooks/01_morocco_geospatial_explore.ipynb, src/geolearn/crs_utils.py`
  - Pre-commit: `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/executed.ipynb`

- [ ] 7. Lesson — Point-in-polygon + Spatial join

  **What to do**:
  - Fill section: Concept (predicates `within`/`contains`/`intersects`) → code: cities points `sjoin` with regions (`predicate='within'`) listing each city → region; second join provinces `intersects`; one `dwithin` 50km example (Casa commuter ring); CRS-match guard before join
  - Teacher note: left vs inner join difference in one sentence + code comment

  **Must NOT do**:
  - No buffer-creation here (T8 owns buffering); joins only

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Core spatial-relational reasoning; predicate choice matters
  - **Skills**: []
  - **Skills Evaluated but Omitted**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 8, 9)
  - **Blocks**: T11, T12
  - **Blocked By**: T3, T4, T6

  **References**:
  - API: `geopandas.sjoin(left, right, how, predicate)` — exact join contract
  - Pattern: `src/geolearn/loaders.py:load_cities/load_regions` — join inputs

  **Acceptance Criteria**:
  - [ ] Notebook table shows Casablanca → `Casablanca-Settat`, Rabat → `Rabat-Salé-Kénitra`, Marrakech → `Marrakech-Safi` (source-spelling tolerant: substring match)

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: city to region mapping
    Tool: Bash
    Preconditions: section filled
    Steps:
      1. Execute notebook, parse join output for Casa/Rabat/Marrakech rows
      2. Assert expected region substrings present
    Expected Result: 3/3 correct mappings
    Failure Indicators: NaN region (CRS mismatch) or wrong predicate
    Evidence: .sisyphus/evidence/task-7-sjoin.log

  Scenario: crs mismatch guarded
    Tool: Bash
    Preconditions: utils present
    Steps:
      1. Run join with deliberately mismatched CRS, assert helper raises or warns and fixed join succeeds
    Expected Result: mismatch detected, not silent wrong result
    Evidence: .sisyphus/evidence/task-7-mismatch.log
  ```

  **Commit**: YES (groups with 5,6,8,9)
  - Message: `feat(geo): core lessons geometry through geometric ops`
  - Files: `notebooks/01_morocco_geospatial_explore.ipynb`
  - Pre-commit: `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/executed.ipynb`

- [ ] 8. Lesson — Distance + Buffer + Area (all metric)

  **What to do**:
  - Fill section: Concept (must be projected!) → code: Casa→Rabat distance km (expect ~80-100km), 50km buffer around Casa (in UTM, then back to 4326 for map), per-region area km² bar table with largest region highlighted
  - Explicit anti-pattern cell: buffer in degrees flagged WRONG with comment

  **Must NOT do**:
  - No choropleth rendering (T10/T11 own it); numbers + tiny preview only

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Unit-correctness reasoning; easy to get silently wrong
  - **Skills**: []
  - **Skills Evaluated but Omitted**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7, 9)
  - **Blocks**: T12
  - **Blocked By**: T3, T4, T6

  **References**:
  - API: `GeoSeries.distance/buffer/area` (metric only after to_crs) — lesson contract
  - Pattern: `src/geolearn/crs_utils.py:distance_km/area_km2` — must use

  **Acceptance Criteria**:
  - [ ] Casa→Rabat prints 80–100 km; Casa 50km buffer area prints 7500–8200 km² (πr² tolerance)

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: distance sane
    Tool: Bash
    Preconditions: section filled
    Steps:
      1. Execute notebook, assert Casa-Rabat km between 80 and 100
    Expected Result: in range
    Failure Indicators: degrees-distance (tiny) or thousands (wrong units)
    Evidence: .sisyphus/evidence/task-8-distance.log

  Scenario: buffer area sane
    Tool: Bash
    Preconditions: section filled
    Steps:
      1. Assert buffer-area cell between 7500 and 8200
      2. Assert degrees-buffer anti-pattern cell present (string "WRONG" in markdown)
    Expected Result: both hold
    Evidence: .sisyphus/evidence/task-8-buffer.log
  ```

  **Commit**: YES (groups with 5-7,9)
  - Message: `feat(geo): core lessons geometry through geometric ops`
  - Files: `notebooks/01_morocco_geospatial_explore.ipynb`
  - Pre-commit: `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/executed.ipynb`

- [ ] 9. Lesson — Geometric ops + validity repair (union/intersection/difference)

  **What to do**:
  - Fill section: Concept (set ops on polygons) → code: `unary_union` of 2 northern regions, `intersection` of Casa buffer with its region, `difference` demo; validity check `is_valid.sum()/len` + `make_valid()` repair demo on one buffered/intersected result
  - One-sentence teacher note when to use each op

  **Must NOT do**:
  - No map rendering; ops + printed areas only

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Topological correctness; validity subtlety
  - **Skills**: []
  - **Skills Evaluated but Omitted**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7, 8)
  - **Blocks**: T12
  - **Blocked By**: T3, T4

  **References**:
  - API: `shapely.ops.unary_union`, `GeoSeries.intersection/union/difference/make_valid` — lesson contract
  - External: `https://shapely.readthedocs.io/en/latest/migration.html` — Shapely 2.x notes

  **Acceptance Criteria**:
  - [ ] Union area equals sum of parts within 1%; intersection area < each input area; validity line prints `12 / 12 valid` (or repaired count)

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: set-op sanity
    Tool: Bash
    Preconditions: section filled
    Steps:
      1. Execute notebook, assert union≈sum and intersection<parts
    Expected Result: both inequalities hold
    Failure Indicators: union >> sum (double-count) or exception
    Evidence: .sisyphus/evidence/task-9-setops.log

  Scenario: validity repaired
    Tool: Bash
    Preconditions: section filled
    Steps:
      1. Assert output contains "valid" and no Traceback
    Expected Result: validity reported, repaired if needed
    Evidence: .sisyphus/evidence/task-9-valid.log
  ```

  **Commit**: YES (groups with 5-8)
  - Message: `feat(geo): core lessons geometry through geometric ops`
  - Files: `notebooks/01_morocco_geospatial_explore.ipynb`
  - Pre-commit: `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/executed.ipynb`

- [ ] 10. Static maps + choropleth PNG (matplotlib)

  **What to do**:
  - Fill section + `src/geolearn/viz.py:static_choropleth()`: Concept (choropleth = color by value) → code: regions plot colored by area_km² with legend + black edges, cities overlay as red dots with 3 labels, save `maps/morocco_regions_static.png` (dpi 150, ≥5KB)
  - Teacher note: when static beats interactive (reports, print)

  **Must NOT do**:
  - No Folium here (T11 owns it); one PNG only

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Cartographic output; layout/legend quality matters
  - **Skills**: []
  - **Skills Evaluated but Omitted**: `frontend-ui-ux` (web UI skill, not static maps)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 11)
  - **Blocks**: T12
  - **Blocked By**: T3, T4, T5

  **References**:
  - Pattern: `src/geolearn/viz.py:static_choropleth` — must implement + use
  - API: `GeoDataFrame.plot(column, legend, edgecolor)` — render contract

  **Acceptance Criteria**:
  - [ ] `maps/morocco_regions_static.png` exists, size >5KB, notebook displays it inline

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: png exists and rich
    Tool: Bash
    Preconditions: section filled
    Steps:
      1. Run `ls -lh maps/morocco_regions_static.png && find maps -name morocco_regions_static.png -size +5k`
      2. Assert both succeed
    Expected Result: PNG present and >5KB
    Failure Indicators: missing or tiny (blank plot)
    Evidence: .sisyphus/evidence/task-10-png.log

  Scenario: choropleth column
    Tool: Bash
    Preconditions: notebook executed
    Steps:
      1. Assert executed notebook contains "area_km" and "legend"
    Expected Result: choropleth visibly value-driven
    Evidence: .sisyphus/evidence/task-10-column.log
  ```

  **Commit**: YES (groups with 11,12)
  - Message: `feat(geo): visualization + e2e polish`
  - Files: `maps/morocco_regions_static.png, src/geolearn/viz.py, notebooks/01_morocco_geospatial_explore.ipynb`
  - Pre-commit: `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/executed.ipynb`

- [ ] 11. Folium interactive + choropleth HTML

  **What to do**:
  - Fill section + `src/geolearn/viz.py:folium_choropleth()`: Concept (tiles + GeoJson + Choropleth) → code: base map centered `[31.8,-6.0]` zoom 5, regions Choropleth by area_km² (`YlGn`, `key_on feature.properties.shapeName` or id fallback), cities `CircleMarker` popups, `LayerControl`; save `maps/morocco_choropleth.html` + `maps/morocco_cities_buffer.html`
  - Teacher note: `key_on` must match GeoJSON property — show the debugging print

  **Must NOT do**:
  - No Kepler.gl; max 2 HTML files

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Interactive web-map output; LayerControl/legend correctness
  - **Skills**: []
  - **Skills Evaluated but Omitted**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 10)
  - **Blocks**: T12
  - **Blocked By**: T3, T4, T7

  **References**:
  - Pattern: `src/geolearn/viz.py:folium_choropleth` — must implement + use
  - External: `https://python-visualization.github.io/folium/latest/user_guide/geojson/choropleth.html` — Choropleth signature

  **Acceptance Criteria**:
  - [ ] Both HTML files exist, each contains `leaflet` (case-insensitive) and `Casablanca`

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: html interactive
    Tool: Bash
    Preconditions: section filled
    Steps:
      1. Run `grep -il leaflet maps/morocco_choropleth.html maps/morocco_cities_buffer.html && grep -c Casablanca maps/*.html`
      2. Assert exit 0 and count >= 1
    Expected Result: Leaflet present + city popup present
    Failure Indicators: missing leaflet (bad save) or no popups
    Evidence: .sisyphus/evidence/task-11-html.log

  Scenario: html renders
    Tool: Playwright (screenshot)
    Preconditions: HTML files exist
    Steps:
      1. Open `maps/morocco_choropleth.html`, wait 3s, screenshot
      2. Assert screenshot shows colored polygons (non-blank, file >20KB)
    Expected Result: visual map captured
    Failure Indicators: blank/grey tiles-only
    Evidence: .sisyphus/evidence/task-11-map.png
  ```

  **Commit**: YES (groups with 10,12)
  - Message: `feat(geo): visualization + e2e polish`
  - Files: `maps/*.html, src/geolearn/viz.py, notebooks/01_morocco_geospatial_explore.ipynb`
  - Pre-commit: `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/executed.ipynb`

- [ ] 12. E2E notebook execute + polish + README (depends: T5-T11)

  **What to do**:
  - Run full `nbconvert --execute`, fix all failures, ensure outputs (tables, numbers, inline PNG, Folium embeds) render; add closing "What you learned" checklist mapping ladder → section; finalize `README.md` (uv setup, `uv run jupyter lab`, data credits ODbL, Western Sahara + 12-regions notes, stretch: GADM communes + Kepler.gl)
  - Sequential: ONLY after T5-T11 all complete (needs every lesson filled)

  **Must NOT do**:
  - No new lessons; polish + fixes only; no scope additions

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration + docs + debugging across all lessons
  - **Skills**: []
  - **Skills Evaluated but Omitted**: none

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 tail (sequential after T10, T11)
  - **Blocks**: F1-F4
  - **Blocked By**: T5, T6, T7, T8, T9, T10, T11

  **References**:
  - Pattern: `notebooks/01_morocco_geospatial_explore.ipynb` (all sections) — integration target
  - Pattern: `README.md`, `data/README.md` — docs to finalize

  **Acceptance Criteria**:
  - [ ] `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/executed.ipynb` exits 0, zero `ename: ERROR` in JSON
  - [ ] README contains `uv sync`, `jupyter lab`, `geoBoundaries`, `ODbL`

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: full execute clean
    Tool: Bash
    Preconditions: all lessons filled
    Steps:
      1. Run nbconvert execute, echo exit code
      2. Run `uv run python -c "import json; nb=json.load(open('/tmp/executed.ipynb')); errs=[c for c in nb['cells'] if 'outputs' in c for o in c['outputs'] if o.get('ename')]; print('errors:', len(errs)); assert len(errs)==0"`
    Expected Result: exit 0 + "errors: 0"
    Failure Indicators: any cell Traceback
    Evidence: .sisyphus/evidence/task-12-e2e.log

  Scenario: readme complete
    Tool: Bash
    Preconditions: README exists
    Steps:
      1. Assert README contains uv sync, jupyter lab, geoBoundaries, ODbL
    Expected Result: all four present
    Evidence: .sisyphus/evidence/task-12-readme.log
  ```

  **Commit**: YES (groups with 10,11)
  - Message: `feat(geo): visualization + e2e polish`
  - Files: `notebooks/01_morocco_geospatial_explore.ipynb, README.md`
  - Pre-commit: `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/executed.ipynb`

---

## Final Verification Wave

- [ ] F1. Plan Compliance Audit — `oracle`
  Read the plan end-to-end. For each Must Have verify implementation exists (read file, run command). For each Must NOT Have search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. Code Quality Review — `unspecified-high`
  Run import checks + `uv run python -m compileall src` + notebook execute. Review changed files for hardcoded absolute paths, bare excepts, missing CRS guards. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Tests [N/A] | Files [N clean/N issues] | VERDICT`

- [ ] F3. Real Manual QA — `unspecified-high`
  Start from clean state (`uv sync` fresh). Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (loaders used by notebook, viz used by maps). Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. Scope Fidelity Check — `deep`
  For each task read What to do, read actual diff. Verify 1:1 — everything built, nothing beyond spec (no Kepler.gl, no ML, no extra modules). Check Must NOT do compliance. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **T1-T4**: `chore(geo): env + data + utils skeleton` — pyproject.toml, src/geolearn/*, data/*
- **T5-T9**: `feat(geo): core lessons geometry through geometric ops` — notebooks/*.ipynb, src/*
- **T10-T12**: `feat(geo): visualization + e2e polish` — maps/*, README.md, notebooks/*.ipynb
- Pre-commit: `uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/executed.ipynb`

---

## Success Criteria

### Verification Commands
```bash
uv sync
uv run python -c "import geopandas, shapely, pyproj, folium; print('stack ok')"
uv run jupyter nbconvert --to notebook --execute notebooks/01_morocco_geospatial_explore.ipynb --output /tmp/executed.ipynb
ls -lh maps/ && grep -il leaflet maps/*.html && find maps -name '*.png' -size +5k
```

### Final Checklist
- [ ] All Must Have present, all Must NOT Have absent
- [ ] Notebook executes with zero failed cells
- [ ] All 8 ops demonstrated with before/after outputs
- [ ] Learner ladder explainable end-to-end
```