# malaigue-from-space — Design

Detecting the 2018 Thau lagoon malaïgue from Sentinel-2 with a foundation model.

- Date: 2026-06-24
- Status: Design, approved to proceed (Sami, 2026-06-24)
- Author: Sami Ennedoui, with Claude Code

## 1. Purpose and context

This is one small, honest, defensible geospatial and deep-learning prototype, built for the CLS DataLab alternance "Data Scientist, Deep Learning et innovation géospatiale". The goal is a real CV line and an interview talking point, framed as exploration, not faked mastery. The value is whether every step can be defended cold, not who typed the code. It mirrors the doctrine of the battery-cycle-life project: reproduce a known result, stress-test it, defend it.

A malaïgue is an anoxic crisis in the étang de Thau. In summer 2018 a severe one killed the entire mussel stock and a large share of the oysters, with losses near 5.9 million euros. The crisis turns the water milky turquoise as sulfur bacteria bloom, preceded by a chlorophyll bloom, so it leaves an optical signature. Earlier work already surveyed Thau's malaïgues from satellite at coarse resolution using MERIS chlorophyll imagery at 300 m per pixel.

The question this prototype asks: can a generic Earth-observation foundation model, Clay, flag the 2018 crisis from Sentinel-2 at 10 m with no task-specific training, and does its signal agree with a simple physical water index and with in-situ measurements.

## 2. The honest verdict structure

The project is worth building whichever way the result falls.

- If Clay embeddings flag the crisis and agree with the index and in-situ data: a generic EO foundation model carries enough signal to flag a water crisis with no fine-tuning.
- If they miss it and the physical index wins: foundation models are not automatically sensitive to out-of-distribution water-quality signals, and a targeted index still beats them here.

Both outcomes are real findings and both are defensible in an interview. The project does not depend on the embeddings winning.

## 3. Anti-fluke design

This is the core requirement. A positive result is believed only if it agrees with three independent anchors:

1. Documented event. The 2018 malaïgue triggered in late June, with white waters confirmed by drone and satellite by 5 July 2018 in the Bouzigues and Mèze sectors.
2. Independent in-situ data. IFREMER REPHY open data for chlorophyll-a, dissolved oxygen and turbidity over Thau, published on SEANOE.
3. Physical control in-image. A spectral chlorophyll and turbidity index from the same Sentinel-2 bands, established science validated over Thau and over the Mar Menor lagoon in Spain.

A coincidence cannot simultaneously match a dated event, an in-situ record, and a physics index. This triple agreement is the explicit success and failure test, not a vibe check.

## 4. Scope

In scope for v1, built in a single pass:

- Spine: a lagoon-level time series of Clay embeddings across summer 2018 plus a normal baseline, with an anomaly score per date, to detect the spike at the crisis date.
- Spatial map: a within-lagoon change map at about 80 m using Clay patch embeddings, localizing the change to the affected sectors, shown next to the index map.
- The physical index control.
- Validation against the three anchors, with numbers.
- A reproducible repo, the maps and figures, a short honest writeup, and a six-point defense gate.

Out of scope, named so the framing stays honest:

- Operational or near-real-time monitoring.
- Water-specific atmospheric correction such as C2RCC or ACOLITE. v1 uses L2A and names this as a limitation.
- Multi-lagoon generalization.
- Fine-tuning Clay or training a new model. Clay is used only as a frozen feature extractor.
- VLMs such as SkyCLIP or RemoteCLIP, and agentic workflows. These are in the job description but are not this prototype.

## 5. Data

- Sentinel-2 L2A surface reflectance from the Microsoft Planetary Computer STAC API, collection `sentinel-2-l2a`, with Element84 Earth Search as a fallback STAC. Fetch the ten bands Clay uses, windowed to the Thau bounding box, low cloud cover only.
- Area of interest: the étang de Thau, approximate bounding box longitude 3.52 to 3.73 E, latitude 43.35 to 43.47 N, refined at build from a coastline or OpenStreetMap polygon. Sub-sector polygons for Bouzigues, Mèze and the main basin support the validation overlays.
- Dates:
  - Normal baseline: one or more clear scenes from a calm period, spring 2018 and a summer 2017 scene.
  - Crisis: the clear Sentinel-2 acquisitions closest to 5 July 2018.
  - Time series: every low-cloud Sentinel-2 scene over Thau from May to September 2018.
- In-situ: REPHY chlorophyll-a, oxygen and turbidity for the Thau stations across 2017 to 2019, from SEANOE.
- Model: Clay v1.5 pretrained weights and band metadata from the Clay Foundation release.

## 6. Architecture

Seven modules with clear boundaries. Each is understandable and testable on its own and communicates through files such as GeoTIFF, GeoParquet or GeoJSON, CSV and npy, plus small typed functions.

1. `ingest`: query the STAC API, filter by bounding box, date and cloud cover, and download the needed bands as a small stacked array per scene. Output is a per-scene band stack plus metadata for date, CRS and transform.
2. `geo`: the geopandas core. Build the lagoon and sub-sector polygons, reproject to the scene CRS, tile the AOI into Clay chips, build land and water masks, and provide the geometry used for every overlay and map. This is where geopandas becomes a real skill.
3. `index`: the physical control. Compute NDCI as `(B5 - B4) / (B5 + B4)` for the chlorophyll proxy and a turbidity proxy from the red band, masked to water. Output is per-pixel index rasters plus per-sector summary statistics.
4. `embed`: load Clay, run the encoder on chips to get the 1024-dimensional chip embedding, and extract patch-token embeddings for the roughly 80 m spatial map. Output is chip embeddings per date and gridded patch embeddings per date.
5. `analyze`: produce two results. The time series is the distance of each date's lagoon embedding from the normal-baseline embedding, with anomaly flags. The spatial result is per-patch change between crisis and baseline by cosine distance, and clustering of patch embeddings, giving a change and anomaly map.
6. `validate`: join everything and score agreement. Temporal: does the embedding-anomaly time series peak within the documented crisis window of late June to early July 2018, and does it co-move with the REPHY chlorophyll and oxygen series, measured by Spearman correlation across the available dates. Spatial: overlap between the top-quantile embedding-anomaly pixels and the top-quantile NDCI pixels, measured by intersection over union, and the fraction of the anomaly that falls inside the Bouzigues and Mèze polygons versus the whole lagoon. Output is the agreement metrics and a written verdict.
7. `report`: render the maps and figures, baseline versus crisis true color, the index map, the embedding anomaly map, the time-series plot and the sector overlays, plus the writeup. Output goes to the docs figures, `decisions.md` and `evaluation.md`.

## 7. Explainability checkpoints

After each stage Sami should be able to state what it does and why, in plain terms. The interview defense gate is six things he must explain cold:

1. Why Sentinel-2 L2A and which bands, and what L2A means.
2. What CRS reprojection and tiling do, and why they are necessary.
3. What NDCI measures physically and why it tracks a bloom.
4. What a Clay embedding is, chip versus patch, and what the dimensions represent.
5. What the embedding anomaly, a cosine distance from baseline, means and where it can mislead.
6. How the three anchors make a positive result trustworthy and rule out a fluke.

The prototype is CV-ready only once all six pass cold, exactly like the battery project gate.

## 8. Success criteria

- Reproducible: a fresh checkout, `uv sync`, and a documented run reproduce every figure from raw STAC queries.
- Honest result delivered: the time-series anomaly and the spatial map are produced, and the verdict states clearly, with numbers, whether Clay's embeddings agree with the index, REPHY and the event.
- Defensible: all six checkpoints pass.
- geopandas genuinely used: footprints, CRS, tiling, masking, overlays and maps.

Detecting the malaïgue with embeddings is not itself a success criterion. A clear, validated negative result is a success.

## 9. Risks and mitigations

- Clay land bias: embeddings may not respond to a color change within water. This is the research question, and the index control guarantees a meaningful comparison either way.
- Patch extraction: if patch-token embeddings are awkward to extract cleanly, the chip-level time-series spine still stands and the spatial map degrades gracefully to coarser chip-level tiling.
- Atmospheric correction over water: L2A is land-tuned. It is named as a limitation, and the strong white-water signal should survive it.
- Cloud cover: the summer Mediterranean is usually clear, and the five-day revisit gives several candidate dates.
- In-situ cadence: REPHY is sampled roughly every two weeks, which may not align tightly with a satellite date. Where in-situ is thin, anchor to the documented chronology.
- Planetary Computer access changes: Element84 Earth Search is the fallback STAC.

## 10. Environment and reproducibility

- Python 3.12 through uv, since system Python 3.14 has no PyTorch wheels yet.
- torch on CPU, the Clay model code, geopandas, rasterio and rioxarray, shapely, pystac-client, planetary-computer, scikit-learn, numpy, pandas, matplotlib.
- CPU only. A handful of chips per date means seconds of inference.
- Dependencies pinned through the uv lockfile. Deterministic seeds where clustering is used.

## 11. Repository layout

```
malaigue-from-space/
  pyproject.toml, uv.lock
  README.md                 honest, written with care, no AI tells
  src/malaigue/
    ingest.py  geo.py  index.py  embed.py  analyze.py  validate.py  report.py
  notebooks/                optional, exploratory
  data/                     gitignored, STAC cache and REPHY extract
  outputs/figures/          committed final figures
  docs/
    decisions.md            running design decisions
    evaluation.md           the verdict, with numbers
    superpowers/specs/2026-06-24-malaigue-from-space-design.md
```

## 12. Honesty and CV framing

- Present this as designed and built with AI-assisted implementation, claiming engineering judgment, the anti-fluke evaluation design, and the ability to defend it, not from-scratch hand-coding. Same bucket as the battery project.
- After this, geopandas becomes a real CV skill. Do not claim mastery of foundation models or VLMs. Frame Clay as exploration, used as a frozen feature extractor.
- Keep the facts and French audit standard from the alternance doctrine for any public copy, including the README and any CV line.
