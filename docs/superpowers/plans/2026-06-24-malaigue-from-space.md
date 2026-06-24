# malaigue-from-space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Sentinel-2 prototype that tests whether Clay foundation-model embeddings can flag the July 2018 Thau lagoon malaïgue, judged against a physical spectral index and IFREMER REPHY in-situ data.

**Architecture:** A seven-module pipeline (`ingest`, `geo`, `index`, `rephy`, `embed`, `analyze`, `validate`, `report`) wired by a top-level run script. Phase 1 builds the data layer and a physics-only baseline that already detects the bloom. Phase 2 adds Clay embeddings on top and produces the honest verdict. Clay is a frozen feature extractor, never fine-tuned.

**Tech Stack:** Python 3.12 via uv, numpy, pandas, geopandas, shapely, rasterio, rioxarray, odc-stac, pystac-client, planetary-computer, scikit-learn, matplotlib, pytest. Phase 2 adds torch (CPU) and claymodel (Clay v1.5) plus huggingface_hub.

## Global Constraints

- Python 3.12 through uv. Fall back to 3.11 only if `claymodel` fails to install on 3.12. System Python 3.14 is unusable (no torch wheels).
- CPU only. No GPU. Keep per-date work to a handful of 256x256 chips.
- Clay is used frozen, as a feature extractor. No fine-tuning, no training.
- Anti-fluke rule: a positive embedding result is reported as real only if it agrees with all three anchors, the documented event (white waters confirmed ~5 July 2018, Bouzigues and Mèze sectors), the REPHY in-situ series, and the in-image NDCI index.
- Honest verdict: a clear, validated negative (embeddings miss it, index wins) is a success, not a failure.
- Deterministic seeds wherever clustering or sampling is used (`random_state=0`).
- Commits: conventional style (`feat:`, `test:`, `chore:`), no `Co-Authored-By` lines. Repo commit email is already set locally to the GitHub noreply.
- geopandas must do the real geospatial work: footprints, CRS, tiling, masking, overlays, maps.
- All paths below are relative to the repo root `~/malaigue-from-space`.

## File Structure

```
src/malaigue/
  __init__.py
  geo.py        AOI bbox, lagoon polygon, sector polygons, CRS reprojection, water mask, chip tiling
  ingest.py     STAC search over Planetary Computer, clear-date listing, clipped band-stack loading
  index.py      NDCI and turbidity proxies, water masking, per-sector statistics
  rephy.py      load and filter IFREMER REPHY in-situ series for Thau
  embed.py      load Clay v1.5, normalize chips, chip and patch embeddings, lagoon embedding
  analyze.py    embedding anomaly time series, spatial change map, patch clustering
  validate.py   temporal and spatial agreement metrics against the three anchors
  report.py     figures, evaluation writeup
  run.py        top-level pipeline orchestration for the real 2018 experiment
tests/
  test_geo.py test_ingest.py test_index.py test_rephy.py
  test_embed.py test_analyze.py test_validate.py test_report.py
  conftest.py   shared synthetic fixtures
data/           gitignored: STAC cache, REPHY extract, Clay checkpoint
outputs/figures/  committed final figures
docs/decisions.md, docs/evaluation.md
```

**Cross-module interfaces (defined once, used everywhere):**

- `geo.aoi_bbox_4326() -> tuple[float,float,float,float]` minx,miny,maxx,maxy
- `geo.lagoon_polygon_4326() -> shapely.geometry.Polygon`
- `geo.sectors_4326() -> geopandas.GeoDataFrame` columns `name, geometry`
- `geo.SCENE_CRS = "EPSG:32631"`
- `geo.reproject_gdf(gdf, dst_crs) -> GeoDataFrame`
- `geo.rasterize_mask(geom, transform, shape, crs) -> numpy.ndarray[bool]`
- `ingest.list_clear_dates(bbox, start, end, max_cloud) -> pandas.DataFrame[date,item_id,cloud]`
- `ingest.load_scene(item_id, bbox, bands, resolution=10) -> xarray.Dataset`
- `index.ndci(ds) -> xarray.DataArray`, `index.turbidity_red(ds) -> xarray.DataArray`
- `index.sector_stats(da, sectors_gdf) -> pandas.DataFrame[name,mean,median,p90,count]`
- `rephy.thau_series(csv_path, params, start, end) -> pandas.DataFrame[date,station,param,value]`
- `embed.load_clay(ckpt, device="cpu") -> object`
- `embed.lagoon_embedding(model, ds, water_mask, date, latlon) -> numpy.ndarray[1024]`
- `embed.patch_embeddings(model, chip, date, latlon) -> numpy.ndarray[Hp,Wp,1024]`
- `analyze.anomaly_timeseries(emb_by_date, baseline_dates) -> pandas.DataFrame[date,distance]`
- `analyze.spatial_change(patch_crisis, patch_baseline) -> numpy.ndarray[Hp,Wp]`
- `validate.temporal_alignment(anom_df, crisis_window) -> dict`
- `validate.spatial_overlap(anom_raster, index_raster, top_q=0.9) -> float`

---

# Phase 1 — Data layer and physics baseline

## Task 1: Project scaffolding and environment

**Files:**
- Create: `pyproject.toml`, `src/malaigue/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: an installable package `malaigue`, a working `uv` env, `pytest` runner.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "malaigue"
version = "0.1.0"
description = "Detecting the 2018 Thau lagoon malaigue from Sentinel-2 with a foundation model"
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy",
  "pandas",
  "geopandas",
  "shapely",
  "rasterio",
  "rioxarray",
  "xarray",
  "odc-stac",
  "pystac-client",
  "planetary-computer",
  "scikit-learn",
  "matplotlib",
]

[dependency-groups]
dev = ["pytest"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/malaigue"]
```

- [ ] **Step 2: Create package and test files**

`src/malaigue/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

`tests/test_smoke.py`:
```python
def test_package_imports():
    import malaigue
    assert malaigue.__version__ == "0.1.0"
```

- [ ] **Step 3: Create the env and install**

Run: `cd ~/malaigue-from-space && uv venv --python 3.12 && uv sync`
Expected: a `.venv/` is created and dependencies resolve. If `geopandas`/`rasterio` fail to build, they should still resolve from wheels on Linux; if not, note the error and stop.

- [ ] **Step 4: Run the smoke test**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/malaigue/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold package and uv environment"
```

## Task 2: Geometry — AOI, lagoon, sectors, reprojection

**Files:**
- Create: `src/malaigue/geo.py`, `tests/test_geo.py`

**Interfaces:**
- Produces: `aoi_bbox_4326`, `lagoon_polygon_4326`, `sectors_4326`, `SCENE_CRS`, `reproject_gdf`.

- [ ] **Step 1: Write failing tests**

`tests/test_geo.py`:
```python
import geopandas as gpd
from shapely.geometry import Polygon
from malaigue import geo

def test_aoi_bbox_covers_thau():
    minx, miny, maxx, maxy = geo.aoi_bbox_4326()
    # Thau lagoon sits around 3.5-3.7 E, 43.36-43.46 N
    assert minx < 3.55 < maxx
    assert miny < 43.42 < maxy

def test_lagoon_polygon_inside_bbox():
    poly = geo.lagoon_polygon_4326()
    assert isinstance(poly, Polygon)
    minx, miny, maxx, maxy = geo.aoi_bbox_4326()
    assert poly.bounds[0] >= minx - 0.01 and poly.bounds[2] <= maxx + 0.01

def test_sectors_have_named_polygons():
    s = geo.sectors_4326()
    assert set(["bouzigues", "meze", "main_basin"]).issubset(set(s["name"]))
    assert s.crs.to_string() == "EPSG:4326"

def test_reproject_to_utm():
    s = geo.sectors_4326()
    out = geo.reproject_gdf(s, geo.SCENE_CRS)
    assert out.crs.to_string() == geo.SCENE_CRS
    # UTM 31N easting for Thau is ~700 km
    assert 600000 < out.geometry.iloc[0].centroid.x < 800000
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_geo.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: module 'malaigue.geo'`.

- [ ] **Step 3: Implement `geo.py`**

```python
"""Geospatial primitives for the Thau lagoon AOI."""
import geopandas as gpd
from shapely.geometry import Polygon, box

SCENE_CRS = "EPSG:32631"  # UTM zone 31N, the Sentinel-2 tile CRS over Thau

def aoi_bbox_4326():
    """(minx, miny, maxx, maxy) bounding box around the etang de Thau in EPSG:4326."""
    return (3.52, 43.35, 3.73, 43.47)

def lagoon_polygon_4326():
    """Approximate outline of the lagoon water body. Refine from OSM at build time
    (Overpass: relation 'Etang de Thau'); these vertices are a usable coarse start."""
    return Polygon([
        (3.535, 43.385), (3.565, 43.435), (3.61, 43.455),
        (3.67, 43.445), (3.705, 43.415), (3.66, 43.375),
        (3.60, 43.365), (3.555, 43.37), (3.535, 43.385),
    ])

def sectors_4326():
    """Named sub-sectors used for validation overlays. Coarse polygons; the two
    crisis sectors (Bouzigues NE shore, Meze NW shore) plus the open main basin."""
    bouzigues = Polygon([(3.64, 43.44), (3.675, 43.45), (3.69, 43.435),
                         (3.66, 43.425), (3.64, 43.44)])
    meze = Polygon([(3.58, 43.43), (3.61, 43.45), (3.625, 43.435),
                    (3.595, 43.418), (3.58, 43.43)])
    main_basin = Polygon([(3.55, 43.39), (3.60, 43.43), (3.655, 43.42),
                          (3.63, 43.385), (3.575, 43.378), (3.55, 43.39)])
    return gpd.GeoDataFrame(
        {"name": ["bouzigues", "meze", "main_basin"]},
        geometry=[bouzigues, meze, main_basin],
        crs="EPSG:4326",
    )

def reproject_gdf(gdf, dst_crs):
    """Reproject a GeoDataFrame to dst_crs."""
    return gdf.to_crs(dst_crs)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_geo.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/malaigue/geo.py tests/test_geo.py
git commit -m "feat: AOI, lagoon and sector geometry with reprojection"
```

## Task 3: Geometry — water mask rasterization and chip tiling

**Files:**
- Modify: `src/malaigue/geo.py`
- Modify: `tests/test_geo.py`

**Interfaces:**
- Produces: `rasterize_mask(geom, transform, shape, crs)`, `chip_windows(bounds, chip_px=256, res=10)`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_geo.py`:
```python
import numpy as np
from affine import Affine

def test_rasterize_mask_marks_inside_pixels():
    # 100x100 grid at 10 m starting at UTM origin (700000, 4810000), north-up
    transform = Affine(10, 0, 700000, 0, -10, 4810000)
    from shapely.geometry import box as shp_box
    geom = shp_box(700100, 4809100, 700900, 4809900)  # an inner square
    mask = geo.rasterize_mask(geom, transform, (100, 100), geo.SCENE_CRS)
    assert mask.dtype == bool
    assert mask.shape == (100, 100)
    assert mask.sum() > 0
    assert not mask[0, 0]  # corner is outside the inner square

def test_chip_windows_tile_bounds():
    bounds = (700000, 4809000, 702560, 4811560)  # 2560 x 2560 m
    wins = geo.chip_windows(bounds, chip_px=256, res=10)
    assert len(wins) == 1  # exactly one 256 px chip fits
    col_off, row_off, w, h = wins[0]
    assert (w, h) == (256, 256)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_geo.py -k "rasterize or chip" -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Implement in `geo.py`**

```python
import numpy as np
from rasterio.features import rasterize

def rasterize_mask(geom, transform, shape, crs):
    """Boolean mask (True inside geom) on a raster grid given by transform+shape."""
    arr = rasterize(
        [(geom, 1)], out_shape=shape, transform=transform,
        fill=0, dtype="uint8", all_touched=False,
    )
    return arr.astype(bool)

def chip_windows(bounds, chip_px=256, res=10):
    """Non-overlapping (col_off, row_off, width, height) windows tiling bounds
    with chip_px square chips at the given resolution in metres."""
    minx, miny, maxx, maxy = bounds
    n_cols = int((maxx - minx) // (chip_px * res))
    n_rows = int((maxy - miny) // (chip_px * res))
    windows = []
    for r in range(max(n_rows, 1)):
        for c in range(max(n_cols, 1)):
            windows.append((c * chip_px, r * chip_px, chip_px, chip_px))
    return windows
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_geo.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/malaigue/geo.py tests/test_geo.py
git commit -m "feat: water-mask rasterization and chip tiling"
```

## Task 4: Ingest — STAC search and clear-date listing

**Files:**
- Create: `src/malaigue/ingest.py`, `tests/test_ingest.py`

**Interfaces:**
- Produces: `list_clear_dates(bbox, start, end, max_cloud) -> DataFrame[date,item_id,cloud]`.
- Consumes: `geo.aoi_bbox_4326`.

- [ ] **Step 1: Spike to confirm the STAC endpoint (no code committed)**

Run:
```bash
uv run python -c "
import planetary_computer, pystac_client
c = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=planetary_computer.sign_inplace)
s = c.search(collections=['sentinel-2-l2a'], bbox=[3.52,43.35,3.73,43.47], datetime='2018-07-01/2018-07-10', query={'eo:cloud_cover':{'lt':20}})
items = list(s.items())
print(len(items), [ (i.id, i.properties['eo:cloud_cover']) for i in items[:5] ])
"
```
Expected: prints a non-zero count and item ids for early July 2018. If Planetary Computer is unreachable, switch the URL to Element84 Earth Search `https://earth-search.aws.element84.com/v1` and collection `sentinel-2-l2a` (no signing needed) and record that in `docs/decisions.md`.

- [ ] **Step 2: Write failing test (mocked client)**

`tests/test_ingest.py`:
```python
import pandas as pd
from malaigue import ingest

class _Item:
    def __init__(self, id, date, cloud):
        self.id = id
        self.properties = {"datetime": date, "eo:cloud_cover": cloud}

def test_list_clear_dates_filters_and_sorts(monkeypatch):
    fake = [
        _Item("S2_b", "2018-07-06T10:00:00Z", 5.0),
        _Item("S2_a", "2018-07-01T10:00:00Z", 12.0),
        _Item("S2_cloudy", "2018-07-03T10:00:00Z", 80.0),
    ]
    monkeypatch.setattr(ingest, "_search_items",
                        lambda bbox, start, end, max_cloud: [i for i in fake if i.properties["eo:cloud_cover"] < max_cloud])
    df = ingest.list_clear_dates((3.52, 43.35, 3.73, 43.47), "2018-07-01", "2018-07-10", max_cloud=20)
    assert list(df["item_id"]) == ["S2_a", "S2_b"]  # sorted by date ascending
    assert df["cloud"].max() < 20
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 4: Implement `ingest.py` (search + listing)**

```python
"""Sentinel-2 L2A access via the Planetary Computer STAC API."""
import pandas as pd
import planetary_computer
import pystac_client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

def _client():
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

def _search_items(bbox, start, end, max_cloud):
    search = _client().search(
        collections=[COLLECTION], bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(search.items())

def list_clear_dates(bbox, start, end, max_cloud=20):
    """Low-cloud Sentinel-2 acquisitions over bbox, sorted by date ascending."""
    rows = []
    for it in _search_items(bbox, start, end, max_cloud):
        rows.append({
            "date": pd.to_datetime(it.properties["datetime"]).date(),
            "item_id": it.id,
            "cloud": float(it.properties["eo:cloud_cover"]),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/malaigue/ingest.py tests/test_ingest.py
git commit -m "feat: STAC search and clear-date listing for Sentinel-2"
```

## Task 5: Ingest — load a clipped band stack

**Files:**
- Modify: `src/malaigue/ingest.py`, `tests/test_ingest.py`

**Interfaces:**
- Produces: `load_scene(item_id, bbox, bands, resolution=10) -> xarray.Dataset` with `rio` CRS/transform; band variables named by asset key (e.g. `B04`, `B05`, `B03`, `SCL`).
- Consumes: STAC item ids from `list_clear_dates`.

- [ ] **Step 1: Implement `load_scene` (data-dependent, smoke-tested)**

Append to `ingest.py`:
```python
import odc.stac
import pystac

# Clay v1.5 Sentinel-2 band order; plus SCL for cloud/water context.
CLAY_S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

def _signed_item(item_id):
    search = _client().search(collections=[COLLECTION], ids=[item_id])
    return next(search.items())

def load_scene(item_id, bbox, bands=None, resolution=10):
    """Load a band stack clipped to bbox (EPSG:4326) as an xarray.Dataset in UTM."""
    bands = bands or CLAY_S2_BANDS
    item = _signed_item(item_id)
    ds = odc.stac.load(
        [item], bands=bands, bbox=list(bbox),
        resolution=resolution, chunks={},
    )
    # collapse the single-time dimension
    return ds.isel(time=0) if "time" in ds.dims else ds
```

- [ ] **Step 2: Write a guarded live smoke test**

Append to `tests/test_ingest.py`:
```python
import os
import pytest

@pytest.mark.skipif(os.environ.get("MALAIGUE_LIVE") != "1",
                    reason="set MALAIGUE_LIVE=1 to hit the live STAC API")
def test_load_scene_live_small():
    from malaigue import geo
    bbox = geo.aoi_bbox_4326()
    df = ingest.list_clear_dates(bbox, "2018-07-01", "2018-07-12", max_cloud=30)
    assert len(df) > 0
    ds = ingest.load_scene(df["item_id"].iloc[0], bbox, bands=["B04", "B05"])
    assert "B04" in ds and "B05" in ds
    assert ds.rio.crs is not None
    assert ds["B04"].shape[0] > 10
```

- [ ] **Step 3: Run the offline tests, then the live test once**

Run: `uv run pytest tests/test_ingest.py -v` (live test skipped)
Then: `MALAIGUE_LIVE=1 uv run pytest tests/test_ingest.py::test_load_scene_live_small -v`
Expected: offline PASS; live PASS and confirms real data loads. If live fails, fix per the Task 4 fallback note before continuing.

- [ ] **Step 4: Commit**

```bash
git add src/malaigue/ingest.py tests/test_ingest.py
git commit -m "feat: load clipped Sentinel-2 band stacks via odc-stac"
```

## Task 6: Index — NDCI, turbidity, water masking, sector stats

**Files:**
- Create: `src/malaigue/index.py`, `tests/test_index.py`

**Interfaces:**
- Produces: `ndci(ds)`, `turbidity_red(ds)`, `apply_water_mask(da, mask)`, `sector_stats(da, sectors_gdf)`.
- Consumes: an `xarray.Dataset` with `B04`, `B05` (from `ingest.load_scene`); `geo` reprojection and rasterization.

- [ ] **Step 1: Write failing tests**

`tests/test_index.py`:
```python
import numpy as np
import xarray as xr
from malaigue import index

def _toy_ds():
    b04 = xr.DataArray(np.array([[0.10, 0.10], [0.20, 0.20]]), dims=("y", "x"))
    b05 = xr.DataArray(np.array([[0.30, 0.30], [0.20, 0.20]]), dims=("y", "x"))
    return xr.Dataset({"B04": b04, "B05": b05})

def test_ndci_formula():
    ds = _toy_ds()
    out = index.ndci(ds)
    # (0.30-0.10)/(0.30+0.10) = 0.5 ; (0.20-0.20)/0.40 = 0.0
    assert np.isclose(out.values[0, 0], 0.5)
    assert np.isclose(out.values[1, 0], 0.0)

def test_apply_water_mask_sets_land_nan():
    ds = _toy_ds()
    nd = index.ndci(ds)
    mask = np.array([[True, False], [True, False]])
    masked = index.apply_water_mask(nd, mask)
    assert np.isnan(masked.values[0, 1])
    assert not np.isnan(masked.values[0, 0])
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_index.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `index.py`**

```python
"""Physical water-quality proxies from Sentinel-2 surface reflectance."""
import numpy as np
import pandas as pd

def ndci(ds):
    """Normalized Difference Chlorophyll Index = (B05 - B04) / (B05 + B04)."""
    b04, b05 = ds["B04"], ds["B05"]
    return (b05 - b04) / (b05 + b04)

def turbidity_red(ds):
    """Simple turbidity proxy: red-band (B04) surface reflectance."""
    return ds["B04"]

def apply_water_mask(da, mask):
    """Set non-water pixels to NaN. mask is a boolean array, True over water."""
    return da.where(mask)

def sector_stats(da, sectors_gdf):
    """Per-sector summary statistics of a masked DataArray.
    sectors_gdf must be in the same CRS as da and carry a 'name' column."""
    from malaigue import geo
    transform = da.rio.transform()
    shape = (da.rio.height, da.rio.width)
    rows = []
    for _, row in sectors_gdf.iterrows():
        m = geo.rasterize_mask(row.geometry, transform, shape, da.rio.crs)
        vals = da.values[m]
        vals = vals[~np.isnan(vals)]
        rows.append({
            "name": row["name"],
            "mean": float(np.mean(vals)) if vals.size else np.nan,
            "median": float(np.median(vals)) if vals.size else np.nan,
            "p90": float(np.percentile(vals, 90)) if vals.size else np.nan,
            "count": int(vals.size),
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_index.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/malaigue/index.py tests/test_index.py
git commit -m "feat: NDCI and turbidity proxies with water masking and sector stats"
```

## Task 7: REPHY — in-situ ground truth for Thau

**Files:**
- Create: `src/malaigue/rephy.py`, `tests/test_rephy.py`

**Interfaces:**
- Produces: `thau_series(csv_path, params, start, end) -> DataFrame[date,station,param,value]`.

- [ ] **Step 1: Spike — download and inspect the REPHY extract (no code committed)**

The REPHY dataset is published on SEANOE (DOI 10.17882/47248). Download the metropolitan CSV into `data/rephy/` and inspect the schema, because the column names drive the parser.

Run:
```bash
uv run python -c "
import pandas as pd
p = 'data/rephy/REPHY.csv'  # adjust to the downloaded filename
df = pd.read_csv(p, sep=';', encoding='latin-1', nrows=50, low_memory=False)
print(list(df.columns))
print(df.head(3).to_dict('records'))
"
```
Expected: prints the real column names. Record in `docs/decisions.md` which columns map to date, station/lieu, parameter, value, and the Thau station label (e.g. a `Lieu` containing "Thau"). Update the constants in Step 3 to match.

- [ ] **Step 2: Write failing test against a synthetic REPHY-shaped CSV**

`tests/test_rephy.py`:
```python
import pandas as pd
from malaigue import rephy

def test_thau_series_filters(tmp_path):
    csv = tmp_path / "rephy.csv"
    pd.DataFrame({
        "Date": ["2018-07-05", "2018-07-05", "2016-01-01"],
        "Lieu": ["Thau - Bouzigues", "Arcachon", "Thau - Meze"],
        "Parametre": ["Chlorophylle-a", "Chlorophylle-a", "Oxygene dissous"],
        "Valeur": [42.0, 1.0, 3.0],
    }).to_csv(csv, sep=";", index=False, encoding="latin-1")
    out = rephy.thau_series(str(csv), params=["Chlorophylle-a", "Oxygene dissous"],
                            start="2018-01-01", end="2018-12-31")
    assert set(out["station"]) <= {"Thau - Bouzigues", "Thau - Meze"}
    assert (out["value"] == 1.0).sum() == 0   # Arcachon dropped
    assert len(out) == 1                        # only the 2018 Thau chlorophyll row
```

- [ ] **Step 3: Implement `rephy.py`**

```python
"""Load IFREMER REPHY in-situ series, filtered to the Thau lagoon."""
import pandas as pd

# Column names confirmed in the Task 7 spike; adjust there if the real file differs.
COL_DATE = "Date"
COL_STATION = "Lieu"
COL_PARAM = "Parametre"
COL_VALUE = "Valeur"
STATION_KEY = "Thau"

def thau_series(csv_path, params, start, end):
    """Tidy long-form Thau series for the requested parameters and date range."""
    df = pd.read_csv(csv_path, sep=";", encoding="latin-1", low_memory=False)
    df = df.rename(columns={
        COL_DATE: "date", COL_STATION: "station",
        COL_PARAM: "param", COL_VALUE: "value",
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["station"].str.contains(STATION_KEY, case=False, na=False)]
    df = df[df["param"].isin(params)]
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).reset_index(drop=True)[["date", "station", "param", "value"]]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_rephy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/malaigue/rephy.py tests/test_rephy.py
git commit -m "feat: load and filter REPHY in-situ series for Thau"
```

**Phase 1 milestone:** at this point the pipeline can fetch real Sentinel-2 over Thau, compute the NDCI bloom map and per-sector stats, and load the REPHY ground truth. That is already a working, defensible physics-only malaïgue detector.

---

# Phase 2 — Foundation-model layer and verdict

## Task 8: Embeddings — install Clay, confirm the API, wrap it

**Files:**
- Modify: `pyproject.toml` (add torch CPU, claymodel, huggingface_hub)
- Create: `src/malaigue/embed.py`, `tests/test_embed.py`

**Interfaces:**
- Produces: `load_clay(ckpt, device="cpu")`, `normalize_chip(stack, band_order)`, `patch_embeddings(model, chip, date, latlon)`, `lagoon_embedding(model, ds, water_mask, date, latlon)`.
- Consumes: an `xarray.Dataset` band stack, a boolean water mask.

- [ ] **Step 1: Add Phase 2 dependencies**

Add to `pyproject.toml` dependencies: `"torch"`, `"huggingface_hub"`, and `"claymodel @ git+https://github.com/Clay-foundation/model.git"`. For CPU torch on Linux this resolves from the default index. Then:

Run: `uv sync`
Expected: resolves. If `claymodel` fails on Python 3.12, recreate the env with `uv venv --python 3.11 && uv sync` and note it in `docs/decisions.md`.

- [ ] **Step 2: Download the checkpoint**

Run:
```bash
uv run python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='made-with-clay/Clay', filename='clay-v1.5.ckpt', local_dir='data/clay')
print(p)
"
```
Expected: the checkpoint path under `data/clay/`. If the repo id or filename differ, find the correct ones on the Clay HuggingFace page and record them in `docs/decisions.md`.

- [ ] **Step 3: Spike — confirm the real encoder API and output shapes (no code committed)**

This is the load-bearing unknown. Load the model and run one synthetic chip to learn the exact call signature and the encoder output structure (does it return a single pooled vector, or the full token sequence including a class token).

Run:
```bash
uv run python -c "
import torch, inspect
from claymodel.module import ClayMAEModule
m = ClayMAEModule.load_from_checkpoint('data/clay/clay-v1.5.ckpt', map_location='cpu')
m.eval()
print('encoder.forward signature:', inspect.signature(m.model.encoder.forward))
# inspect what the datacube/batch dict keys are expected to be:
print([n for n in dir(m.model.encoder) if not n.startswith('_')][:40])
"
```
Expected: prints the encoder forward signature (Clay v1.5 takes a `datacube` dict with keys such as `pixels`, `time`, `latlon`, `waves`, `gsd`). Record the confirmed signature, the patch size, and the token-sequence shape in `docs/decisions.md`. The wrapper in Step 5 is written against these confirmed facts; adjust the two marked lines if they differ.

- [ ] **Step 4: Write the embedding tests (synthetic, skip if Clay absent)**

`tests/test_embed.py`:
```python
import importlib.util
import numpy as np
import pytest
from malaigue import embed

clay_absent = importlib.util.find_spec("claymodel") is None

def test_normalize_chip_centers_values():
    stack = np.ones((10, 8, 8), dtype="float32") * 0.2
    mean = np.full(10, 0.2); std = np.full(10, 0.1)
    out = embed.normalize_chip(stack, mean, std)
    assert np.allclose(out, 0.0)
    assert out.shape == (10, 8, 8)

@pytest.mark.skipif(clay_absent, reason="claymodel not installed")
def test_patch_embeddings_shape():
    model = embed.load_clay("data/clay/clay-v1.5.ckpt", device="cpu")
    chip = np.random.rand(10, 256, 256).astype("float32")
    pe = embed.patch_embeddings(model, chip, date="2018-07-05", latlon=(43.42, 3.62))
    assert pe.ndim == 3 and pe.shape[2] == 1024   # (Hp, Wp, 1024)
```

- [ ] **Step 5: Implement `embed.py` against the confirmed API**

```python
"""Clay v1.5 as a frozen Sentinel-2 feature extractor."""
import datetime as dt
import numpy as np
import torch

# Clay v1.5 Sentinel-2 normalization (per-band mean/std, surface reflectance).
# Pulled from the Clay metadata.yaml during the Task 8 spike; confirm and pin.
S2_MEAN = np.array([1369, 1597, 1741, 2053, 2569, 2763, 2906, 2961, 2092, 1428], dtype="float32")
S2_STD = np.array([2026, 2141, 2238, 2362, 2418, 2479, 2502, 2530, 1815, 1411], dtype="float32")
S2_WAVELENGTHS = [0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]
EMBED_DIM = 1024

def load_clay(ckpt, device="cpu"):
    from claymodel.module import ClayMAEModule
    model = ClayMAEModule.load_from_checkpoint(ckpt, map_location=device)
    model.eval().to(device)
    return model

def normalize_chip(stack, mean=None, std=None):
    """(x - mean) / std per band. stack is (bands, H, W)."""
    mean = S2_MEAN if mean is None else mean
    std = S2_STD if std is None else std
    return (stack - mean[:, None, None]) / std[:, None, None]

def _datacube(chip, date, latlon, device="cpu"):
    """Build the Clay encoder input. Field names confirmed in the Task 8 spike."""
    norm = normalize_chip(chip.astype("float32"))
    pixels = torch.from_numpy(norm).unsqueeze(0).to(device)          # [1,10,256,256]
    d = dt.date.fromisoformat(date)
    week = d.isocalendar().week
    time = torch.tensor([[np.sin(2*np.pi*week/52), np.cos(2*np.pi*week/52), 0.0, 0.0]],
                        dtype=torch.float32, device=device)
    lat, lon = latlon
    latlon_t = torch.tensor([[np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat)),
                              np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon))]],
                            dtype=torch.float32, device=device)
    waves = torch.tensor(S2_WAVELENGTHS, dtype=torch.float32, device=device)
    gsd = torch.tensor([10.0], dtype=torch.float32, device=device)
    # ADJUST IF SPIKE DIFFERS: key names / pooling below come from the confirmed API.
    return {"pixels": pixels, "time": time, "latlon": latlon_t, "waves": waves, "gsd": gsd}

def patch_embeddings(model, chip, date, latlon, device="cpu"):
    """Per-patch embeddings reshaped to (Hp, Wp, 1024). Drops the leading class token."""
    cube = _datacube(chip, date, latlon, device)
    with torch.no_grad():
        tokens = model.model.encoder(cube)          # ADJUST IF SPIKE DIFFERS
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[0]
    seq = tokens[0]                                  # [num_tokens, 1024]
    patches = seq[1:]                                # drop class/group token
    n = patches.shape[0]
    side = int(round(n ** 0.5))
    return patches[: side * side].reshape(side, side, EMBED_DIM).cpu().numpy()

def lagoon_embedding(model, ds, water_mask, date, latlon, chip_px=256, device="cpu"):
    """Single 1024-d embedding for the lagoon: mean of patch embeddings over water.
    Centers one chip on the lagoon and pools patches whose footprint is water."""
    from malaigue import ingest
    bands = [ds[b].values for b in ingest.CLAY_S2_BANDS]
    stack = np.stack(bands).astype("float32")
    stack = _center_crop(stack, chip_px)
    mask = _center_crop(water_mask[None].astype("float32"), chip_px)[0] > 0.5
    pe = patch_embeddings(model, stack, date, latlon, device)
    pm = _downsample_mask(mask, pe.shape[:2])
    sel = pe[pm]
    if sel.shape[0] == 0:
        return pe.reshape(-1, EMBED_DIM).mean(0)
    return sel.mean(0)

def _center_crop(arr, size):
    _, h, w = arr.shape
    top = max((h - size) // 2, 0); left = max((w - size) // 2, 0)
    out = arr[:, top:top + size, left:left + size]
    if out.shape[1] < size or out.shape[2] < size:
        pad = ((0, 0), (0, size - out.shape[1]), (0, size - out.shape[2]))
        out = np.pad(out, pad)
    return out

def _downsample_mask(mask, target_hw):
    th, tw = target_hw
    fh, fw = mask.shape[0] // th, mask.shape[1] // tw
    return mask[: th * fh, : tw * fw].reshape(th, fh, tw, fw).mean((1, 3)) > 0.5
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_embed.py -v`
Expected: `test_normalize_chip_centers_values` PASS; `test_patch_embeddings_shape` PASS if Clay installed, else SKIP. If the shape assertion fails, fix the marked lines using the Step 3 spike output before continuing.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/malaigue/embed.py tests/test_embed.py docs/decisions.md
git commit -m "feat: Clay v1.5 frozen feature extractor with chip and patch embeddings"
```

## Task 9: Analyze — anomaly time series and spatial change

**Files:**
- Create: `src/malaigue/analyze.py`, `tests/test_analyze.py`

**Interfaces:**
- Produces: `anomaly_timeseries(emb_by_date, baseline_dates)`, `spatial_change(patch_crisis, patch_baseline)`, `cluster_patches(patches, k, seed=0)`.
- Consumes: embeddings from `embed`.

- [ ] **Step 1: Write failing tests**

`tests/test_analyze.py`:
```python
import numpy as np
import datetime as dt
from malaigue import analyze

def test_anomaly_timeseries_flags_outlier():
    base = np.array([1.0, 0.0, 0.0])
    emb = {dt.date(2018,5,1): base, dt.date(2018,6,1): base,
           dt.date(2018,7,5): np.array([0.0, 1.0, 0.0])}
    df = analyze.anomaly_timeseries(emb, baseline_dates=[dt.date(2018,5,1), dt.date(2018,6,1)])
    peak = df.sort_values("distance").iloc[-1]
    assert peak["date"] == dt.date(2018,7,5)
    assert np.isclose(df[df.date==dt.date(2018,5,1)]["distance"].iloc[0], 0.0, atol=1e-6)

def test_spatial_change_is_zero_when_identical():
    a = np.random.rand(4, 4, 8).astype("float32")
    out = analyze.spatial_change(a, a)
    assert out.shape == (4, 4)
    assert np.allclose(out, 0.0, atol=1e-6)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `analyze.py`**

```python
"""Downstream analysis of Clay embeddings."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def _cosine_distance(u, v):
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return 1.0 - float(np.dot(u, v) / (nu * nv))

def anomaly_timeseries(emb_by_date, baseline_dates):
    """Cosine distance of each date's embedding from the mean baseline embedding."""
    baseline = np.mean([emb_by_date[d] for d in baseline_dates], axis=0)
    rows = [{"date": d, "distance": _cosine_distance(v, baseline)}
            for d, v in sorted(emb_by_date.items())]
    return pd.DataFrame(rows)

def spatial_change(patch_crisis, patch_baseline):
    """Per-patch cosine distance between two (Hp,Wp,D) embedding grids."""
    a = patch_crisis.reshape(-1, patch_crisis.shape[-1])
    b = patch_baseline.reshape(-1, patch_baseline.shape[-1])
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den[den == 0] = 1.0
    dist = 1.0 - num / den
    return dist.reshape(patch_crisis.shape[:2])

def cluster_patches(patches, k, seed=0):
    """KMeans labels over a (Hp,Wp,D) grid, returned as (Hp,Wp)."""
    h, w, d = patches.shape
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(
        patches.reshape(-1, d))
    return labels.reshape(h, w)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/malaigue/analyze.py tests/test_analyze.py
git commit -m "feat: embedding anomaly time series, spatial change and clustering"
```

## Task 10: Validate — agreement against the three anchors

**Files:**
- Create: `src/malaigue/validate.py`, `tests/test_validate.py`

**Interfaces:**
- Produces: `temporal_alignment(anom_df, crisis_window)`, `spearman_vs_insitu(anom_df, rephy_df, param)`, `spatial_overlap(anom_raster, index_raster, top_q=0.9)`, `fraction_in_sectors(anom_raster, transform, crs, sectors_gdf)`.
- Consumes: outputs of `analyze`, `index`, `rephy`, `geo`.

- [ ] **Step 1: Write failing tests**

`tests/test_validate.py`:
```python
import numpy as np
import pandas as pd
import datetime as dt
from malaigue import validate

def test_temporal_alignment_peak_in_window():
    df = pd.DataFrame({"date": [dt.date(2018,5,1), dt.date(2018,7,5)],
                       "distance": [0.0, 0.8]})
    out = validate.temporal_alignment(df, ("2018-06-25", "2018-07-10"))
    assert out["peak_date"] == dt.date(2018,7,5)
    assert out["peak_in_window"] is True

def test_spatial_overlap_identical_top_quantile():
    a = np.arange(16, dtype="float32").reshape(4, 4)
    iou = validate.spatial_overlap(a, a.copy(), top_q=0.75)
    assert iou == 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_validate.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `validate.py`**

```python
"""Agreement metrics tying embeddings to the three anti-fluke anchors."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def temporal_alignment(anom_df, crisis_window):
    """Does the anomaly peak fall inside the documented crisis window."""
    start, end = pd.to_datetime(crisis_window[0]).date(), pd.to_datetime(crisis_window[1]).date()
    peak = anom_df.sort_values("distance").iloc[-1]
    pd_date = peak["date"]
    return {"peak_date": pd_date, "peak_distance": float(peak["distance"]),
            "peak_in_window": bool(start <= pd_date <= end)}

def spearman_vs_insitu(anom_df, rephy_df, param):
    """Spearman correlation between the embedding anomaly and an in-situ parameter,
    matched on nearest date."""
    ins = rephy_df[rephy_df["param"] == param].copy()
    ins["date"] = pd.to_datetime(ins["date"])
    a = anom_df.copy(); a["date"] = pd.to_datetime(a["date"])
    merged = pd.merge_asof(a.sort_values("date"), ins.sort_values("date"),
                           on="date", direction="nearest", tolerance=pd.Timedelta("7D")).dropna()
    if len(merged) < 3:
        return {"rho": np.nan, "n": len(merged)}
    rho, p = spearmanr(merged["distance"], merged["value"])
    return {"rho": float(rho), "p": float(p), "n": len(merged)}

def spatial_overlap(anom_raster, index_raster, top_q=0.9):
    """IoU of the top-quantile hotspots of two rasters."""
    a = anom_raster[~np.isnan(anom_raster)]
    i = index_raster[~np.isnan(index_raster)]
    if a.size == 0 or i.size == 0:
        return np.nan
    am = anom_raster >= np.quantile(a, top_q)
    im = index_raster >= np.quantile(i, top_q)
    union = np.logical_or(am, im).sum()
    return float(np.logical_and(am, im).sum() / union) if union else np.nan

def fraction_in_sectors(anom_raster, transform, crs, sectors_gdf, top_q=0.9):
    """Share of anomaly hotspot pixels falling inside each named sector."""
    from malaigue import geo
    a = anom_raster[~np.isnan(anom_raster)]
    hot = anom_raster >= np.quantile(a, top_q)
    total = hot.sum()
    out = {}
    for _, row in sectors_gdf.iterrows():
        m = geo.rasterize_mask(row.geometry, transform, anom_raster.shape, crs)
        out[row["name"]] = float(np.logical_and(hot, m).sum() / total) if total else np.nan
    return out
```

- [ ] **Step 4: Add scipy and run to verify pass**

Add `"scipy"` to `pyproject.toml` dependencies, then `uv sync`.
Run: `uv run pytest tests/test_validate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/malaigue/validate.py tests/test_validate.py
git commit -m "feat: temporal and spatial agreement metrics for the three anchors"
```

## Task 11: Report — figures and writeup helpers

**Files:**
- Create: `src/malaigue/report.py`, `tests/test_report.py`

**Interfaces:**
- Produces: `plot_index_map(da, path)`, `plot_anomaly_map(raster, path)`, `plot_timeseries(anom_df, rephy_df, path)`, `write_evaluation(metrics, path)`.

- [ ] **Step 1: Write failing test (file is produced)**

`tests/test_report.py`:
```python
import numpy as np
import pandas as pd
import datetime as dt
from malaigue import report

def test_plot_anomaly_map_writes_png(tmp_path):
    raster = np.random.rand(16, 16)
    out = tmp_path / "anom.png"
    report.plot_anomaly_map(raster, str(out))
    assert out.exists() and out.stat().st_size > 0

def test_write_evaluation_contains_verdict(tmp_path):
    out = tmp_path / "evaluation.md"
    report.write_evaluation({"verdict": "index wins", "spatial_iou": 0.12}, str(out))
    text = out.read_text()
    assert "index wins" in text and "spatial_iou" in text
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `report.py`**

```python
"""Figures and the evaluation writeup."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_index_map(da, path, title="NDCI"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(da.values, cmap="viridis"); ax.set_title(title); fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)

def plot_anomaly_map(raster, path, title="Embedding anomaly (cosine distance)"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(raster, cmap="magma"); ax.set_title(title); fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)

def plot_timeseries(anom_df, rephy_df, path):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(anom_df["date"], anom_df["distance"], "o-", color="crimson", label="embedding anomaly")
    ax1.set_ylabel("embedding anomaly", color="crimson")
    if rephy_df is not None and len(rephy_df):
        ax2 = ax1.twinx()
        for p, g in rephy_df.groupby("param"):
            ax2.plot(g["date"], g["value"], "s--", alpha=0.6, label=p)
        ax2.set_ylabel("REPHY in-situ")
    fig.autofmt_xdate(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)

def write_evaluation(metrics, path):
    lines = ["# Evaluation\n"]
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/malaigue/report.py tests/test_report.py
git commit -m "feat: figure helpers and evaluation writeup"
```

## Task 12: End-to-end run on 2018 and the verdict

**Files:**
- Create: `src/malaigue/run.py`
- Create: `docs/decisions.md`, `docs/evaluation.md` (filled by the run)
- Create: `README.md`
- Create: `outputs/figures/` (committed PNGs)

**Interfaces:**
- Consumes: every module above.

- [ ] **Step 1: Write the orchestration script**

`src/malaigue/run.py`:
```python
"""End-to-end Thau malaigue experiment for summer 2018."""
import datetime as dt
import numpy as np
from malaigue import geo, ingest, index, rephy, embed, analyze, validate, report

CRISIS_WINDOW = ("2018-06-25", "2018-07-10")
BASELINE = ("2018-04-01", "2018-05-31")
SEASON = ("2018-05-01", "2018-09-30")
LATLON = (43.42, 3.62)

def main():
    bbox = geo.aoi_bbox_4326()
    dates = ingest.list_clear_dates(bbox, *SEASON, max_cloud=20)
    print(dates)

    model = embed.load_clay("data/clay/clay-v1.5.ckpt")
    lagoon4326 = geo.lagoon_polygon_4326()

    emb_by_date, ndci_by_date = {}, {}
    for _, r in dates.iterrows():
        ds = ingest.load_scene(r["item_id"], bbox)
        ds = ds.rio.reproject(geo.SCENE_CRS) if ds.rio.crs.to_string() != geo.SCENE_CRS else ds
        lag = geo.reproject_gdf(
            geo.sectors_4326().assign(geometry=lagoon4326).iloc[[0]], geo.SCENE_CRS).geometry.iloc[0]
        mask = geo.rasterize_mask(lag, ds.rio.transform(),
                                  (ds.rio.height, ds.rio.width), ds.rio.crs)
        emb_by_date[r["date"]] = embed.lagoon_embedding(model, ds, mask, str(r["date"]), LATLON)
        ndci_by_date[r["date"]] = index.apply_water_mask(index.ndci(ds), mask)

    baseline_dates = [d for d in emb_by_date if dt.date(2018,4,1) <= d <= dt.date(2018,5,31)]
    anom = analyze.anomaly_timeseries(emb_by_date, baseline_dates)

    insitu = rephy.thau_series("data/rephy/REPHY.csv",
                               params=["Chlorophylle-a", "Oxygene dissous"], start="2018-01-01", end="2018-12-31")
    temporal = validate.temporal_alignment(anom, CRISIS_WINDOW)
    rho = validate.spearman_vs_insitu(anom, insitu, "Chlorophylle-a")

    report.plot_timeseries(anom, insitu, "outputs/figures/timeseries.png")
    metrics = {"temporal": temporal, "spearman_chl": rho}
    report.write_evaluation(metrics, "docs/evaluation.md")
    print(metrics)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full experiment**

Run: `MALAIGUE_LIVE=1 uv run python -m malaigue.run`
Expected: prints the clear-date table, computes the anomaly series and metrics, writes `outputs/figures/timeseries.png` and `docs/evaluation.md`. Inspect the figure and the printed metrics. The patch-level spatial map is added on top in Task 13.

- [ ] **Step 3: Write the honest verdict and decisions**

Fill `docs/evaluation.md` with the numbers and a plain-language verdict: did the embedding anomaly peak in the crisis window, does it correlate with REPHY chlorophyll, and does the spatial hotspot overlap NDCI and the Bouzigues/Meze sectors. State clearly whether Clay carried the signal or the index won. Fill `docs/decisions.md` with the choices made during the spikes (STAC endpoint, Clay API, REPHY columns).

- [ ] **Step 4: Write the README**

Write `README.md` in careful prose, plain language, no AI tells, no em-dashes, no emojis, following the user's writing rules. Cover: the question, the data, the method, the three-anchor validation, the honest result, the limitations (L2A over water, REPHY cadence), and how to reproduce. Frame Clay as exploration used frozen, not mastered.

- [ ] **Step 5: Commit**

```bash
git add src/malaigue/run.py docs/evaluation.md docs/decisions.md README.md outputs/figures/
git commit -m "feat: end-to-end 2018 Thau experiment, figures and verdict"
```

- [ ] **Step 6: The defense gate**

Confirm Sami can explain cold the six checkpoints from the spec section 7: L2A and bands; CRS and tiling; what NDCI measures; what a Clay embedding is, chip versus patch; what the cosine anomaly means and its failure modes; how the three anchors rule out a fluke. The prototype is CV-ready only when all six pass.

## Task 13: Patch-level spatial change map and its overlap with NDCI

**Files:**
- Modify: `src/malaigue/analyze.py`, `tests/test_analyze.py` (add `block_mean`)
- Modify: `src/malaigue/run.py` (add `spatial_map`, wire into `main`)

**Interfaces:**
- Produces: `analyze.block_mean(a, ny, nx)`, `run.spatial_map(model, bbox, lagoon4326, crisis_id, base_id)`.
- Consumes: `embed.patch_embeddings`, `analyze.spatial_change`, `index.ndci`, `validate.spatial_overlap`.

- [ ] **Step 1: Write the failing test for `block_mean`**

Append to `tests/test_analyze.py`:
```python
def test_block_mean_reduces_grid():
    a = np.arange(16, dtype="float32").reshape(4, 4)
    out = analyze.block_mean(a, 2, 2)
    assert out.shape == (2, 2)
    assert np.isclose(out[0, 0], np.mean([0, 1, 4, 5]))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_analyze.py::test_block_mean_reduces_grid -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Implement `block_mean` in `analyze.py`**

```python
def block_mean(a, ny, nx):
    """Reduce 2D array a to shape (ny, nx) by averaging non-overlapping blocks,
    ignoring NaNs. Aligns a full-resolution index map onto the patch grid."""
    h, w = a.shape
    fh, fw = h // ny, w // nx
    a = a[: ny * fh, : nx * fw]
    return np.nanmean(a.reshape(ny, fh, nx, fw), axis=(1, 3))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: PASS.

- [ ] **Step 5: Add `spatial_map` to `run.py` and wire it into `main`**

Add near the top of `run.py`: `import geopandas as gpd` and `import numpy as np` (if not present). Insert above `main`:
```python
def spatial_map(model, bbox, lagoon4326, crisis_id, base_id, latlon=LATLON):
    def _chip_and_ndci(item_id):
        ds = ingest.load_scene(item_id, bbox)
        if ds.rio.crs.to_string() != geo.SCENE_CRS:
            ds = ds.rio.reproject(geo.SCENE_CRS)
        lag = geo.reproject_gdf(gpd.GeoDataFrame(geometry=[lagoon4326], crs="EPSG:4326"),
                                geo.SCENE_CRS).geometry.iloc[0]
        mask = geo.rasterize_mask(lag, ds.rio.transform(),
                                  (ds.rio.height, ds.rio.width), ds.rio.crs)
        stack = np.stack([ds[b].values for b in ingest.CLAY_S2_BANDS]).astype("float32")
        stack = embed._center_crop(stack, 256)
        ndci = index.apply_water_mask(index.ndci(ds), mask)
        ndci_chip = embed._center_crop(ndci.values[None], 256)[0]
        return stack, ndci, ndci_chip
    chip_c, ndci_c, ndci_chip_c = _chip_and_ndci(crisis_id)
    chip_b, _, _ = _chip_and_ndci(base_id)
    pe_c = embed.patch_embeddings(model, chip_c, "2018-07-05", latlon)
    pe_b = embed.patch_embeddings(model, chip_b, "2018-05-01", latlon)
    anom = analyze.spatial_change(pe_c, pe_b)
    ndci_grid = analyze.block_mean(ndci_chip_c, anom.shape[0], anom.shape[1])
    return {"anom": anom, "ndci_full": ndci_c, "ndci_grid": ndci_grid}
```
In `main`, move the `metrics = {"temporal": temporal, "spearman_chl": rho}` assignment up to just after `rho` is computed, then after the timeseries plot add:
```python
    crisis_id = dates[(dates["date"] >= dt.date(2018, 6, 25)) &
                      (dates["date"] <= dt.date(2018, 7, 10))]["item_id"].iloc[0]
    base_id = dates[dates["date"].isin(baseline_dates)]["item_id"].iloc[0]
    sp = spatial_map(model, bbox, lagoon4326, crisis_id, base_id)
    iou = validate.spatial_overlap(sp["anom"], sp["ndci_grid"], top_q=0.9)
    report.plot_anomaly_map(sp["anom"], "outputs/figures/anomaly_map.png")
    report.plot_index_map(sp["ndci_full"], "outputs/figures/ndci_crisis.png", title="NDCI crisis")
    metrics["spatial_iou_anom_vs_ndci"] = iou
```

- [ ] **Step 6: Run live and inspect**

Run: `MALAIGUE_LIVE=1 uv run python -m malaigue.run`
Expected: writes `outputs/figures/anomaly_map.png` and `ndci_crisis.png`, and `spatial_iou_anom_vs_ndci` appears in the metrics and `docs/evaluation.md`. Inspect whether the embedding hotspot overlaps the NDCI hotspot and the Bouzigues and Meze shores.

- [ ] **Step 7: Commit**

```bash
git add src/malaigue/analyze.py tests/test_analyze.py src/malaigue/run.py docs/evaluation.md outputs/figures/
git commit -m "feat: patch-level spatial change map and NDCI overlap"
```

---

## Self-review notes

- Spec coverage: ingest, geo, index, rephy, embed, analyze, validate, report all have tasks; the three-anchor validation is Task 10; the honest verdict and defense gate are Task 12. Patch-level spatial map is covered by `embed.patch_embeddings` + `analyze.spatial_change` + `validate.spatial_overlap`, wired concretely in Task 13.
- Known empirical unknowns are handled by explicit spikes, not placeholders: STAC reachability (Task 4), REPHY schema (Task 7), the Clay encoder API and normalization constants (Task 8). Each spike has a concrete command and an expected output, and the dependent code marks the exact lines to adjust.
- Type consistency: `emb_by_date` is `dict[date, ndarray]` in embed/analyze/run; `anom_df` has columns `date,distance` in analyze/validate/report; rasters are 2-D numpy arrays throughout.
```
