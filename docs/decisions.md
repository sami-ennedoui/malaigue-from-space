# Decisions and build log

Running log of empirical findings and choices made during the build. Workers read
this first and append new findings last.

## Data access (Task 4 spike, 2026-06-24)
- STAC endpoint: Microsoft Planetary Computer, `https://planetarycomputer.microsoft.com/api/stac/v1`,
  collection `sentinel-2-l2a`, assets signed with `planetary_computer.sign_inplace`. Reachable and working.
- Sentinel-2 tile over Thau: **T31TEJ**, UTM zone 31N (**EPSG:32631**).
- UTM easting for Thau near 3.6 E is ~554 km (central meridian 3 E). Corrected an over-estimated
  bound in `test_geo.py` accordingly.

## Clear Sentinel-2 scenes over Thau, summer 2018 (cloud < 20%)
13 scenes over bbox (3.52, 43.35, 3.73, 43.47), 2018-05-01 to 2018-09-30:

| date | cloud % | role |
|------|---------|------|
| 2018-05-18 | 19.6 | baseline candidate (a bit cloudy) |
| 2018-06-27 | 1.9 | pre-crisis / onset |
| 2018-07-02 | 10.3 | crisis |
| 2018-07-07 | 0.8 | crisis peak scene (white waters confirmed ~5 July) |
| 2018-07-17 | 2.9 | post |
| 2018-07-27 | 4.0 | post |
| 2018-08-06 | 1.9 | post |
| 2018-08-16 | 8.2 | post |
| 2018-08-21 | 4.7 | post |
| 2018-08-26 | 4.2 | post |
| 2018-08-31 | 2.9 | post |
| 2018-09-20 | 0.9 | recovery |

- Crisis scene for the spatial map: `S2A_MSIL2A_20180707T104021_R008_T31TEJ_20201011T125724` (0.8% cloud).
- Baseline: 2018-05-18 is the only pre-June clear scene in-window but is 19.6% cloud. Pull an April 2018
  scene for a cleaner baseline at the real-run step (Task 12).
- The dense July to September run gives a good embedding time series.

## Open items to confirm later
- Atmospheric correction: using L2A (land-tuned) as a known limitation over water.

## Clay v1.5 (Task 8 spike, 2026-06-24)
- Install: `claymodel` from git (Clay-foundation/model). MUST pin BOTH torch and torchvision to the
  pytorch-cpu index (download.pytorch.org/whl/cpu). A PyPI torchvision against the +cpu torch breaks
  with "operator torchvision::nms does not exist". Installed torch 2.12.1+cpu, torchvision 0.27.1+cpu.
- Checkpoint: HF repo `made-with-clay/Clay`, file `v1.5/clay-v1.5.ckpt` (~5.2 GB Lightning checkpoint,
  bundles optimizer state). RAM here is 15 GB total / ~7.8 GB free, so do NOT torch.load the whole
  file: load with `mmap=True` and copy only `model.encoder.*` into the encoder.
- metadata.yaml is NOT shipped in the installed package; fetched from
  raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml into data/clay/.
- Hyperparameters: model_size "large", patch_size 8, dim 1024, embeddings_level "mean".
- Lean load that works: build ClayMAEModule(model_size="large", mask_ratio=0.0, patch_size=8,
  shuffle=False, metadata_path=...), then load ONLY model.encoder.* (n_missing=0, n_unexpected=0).
  Skips the dinov2 teacher and proj-head shape mismatches; the default small SAM-ViT teacher is built
  but never used.
- encoder.forward(datacube), datacube = {pixels [1,10,256,256], time [1,4], latlon [1,4],
  waves [10], gsd scalar}. Returns a tuple: out[0]=(1,1025,1024) cls+patches, out[1]=(1,1024) pooled.
  Patch embeddings = out[0][0,1:].reshape(32,32,1024); chip embedding = out[1][0].
- S2 band order/means/stds/wavelengths from metadata sentinel-2-l2a; order
  blue,green,red,rededge1,rededge2,rededge3,nir,nir08,swir16,swir22 == our CLAY_S2_BANDS
  (B02..B12, B8A). Pixels fed as raw L2A DN (no /10000).
- Calibration caveat: 2018 L2A reprocessed; BOA offset convention not adjusted. Minor vs the strong
  white-water signal; noted as a limitation.
- Verified on CPU: load + one chip -> (32,32,1024) in ~21 s.

## Integration note (Task 5)
- The `.rio` accessor requires `import rioxarray` somewhere in the process; added to `ingest.py`.
  odc-stac datasets then expose `.rio.crs/.transform/.height/.width` via the `spatial_ref` coord.

## REPHY in-situ (Task 7 spike, 2026-06-24)
- File: `REPHY_Med_1987-2022.csv` (273 MB, the Mediterranean half of the SEANOE zip).
  Semicolon-separated, latin-1, 56 Quadrige columns.
- Thau filter: `Lieu de surveillance : Entité de classement : Libellé` == "104 - Etang de Thau"
  (49,896 rows). Main long-term station "Bouzigues (a)" (a crisis sector); also Marseillan (a),
  Thau - Crique de l'Angle, Mèze zone a/b.
- Parameter labels: "Chlorophylle a", "Oxygène dissous", "Turbidité FNU".
- Columns used: "Passage : Date" (dd/mm/yyyy), the entité, "Lieu de surveillance : Libellé" (station),
  "Résultat : Libellé paramètre", "Résultat : Valeur de la mesure" (French comma decimals, handled).

### 2018 in-situ malaïgue signal (Bouzigues a, dissolved oxygen, mg/L)
- 2018-06-19: ~6.5 to 7.1 (healthy baseline)
- 2018-07-02: dips to ~4.0 to 4.4 (early-July onset, matches the documented ~5 July white waters)
- 2018-07-18 / 07-30: back to ~5.8 to 6.2
- 2018-08-13: 0.04 at Bouzigues (near-total anoxia); chlorophyll a 5.11 and turbidity 5.69 at
  Marseillan (vs ~0.5 to 1.0 and ~1 normally) — the most severe in-situ anoxic peak of the summer.
- Takeaway: the 2018 malaïgue was multi-pulse (early-July onset, severe mid-August anoxic peak).
  REPHY is biweekly, so it samples the pulses coarsely.

### Implication for the satellite experiment (feeds Task 12)
- Crisis scenes to test: 2018-07-07 (0.8% cloud, near the July onset) and 2018-08-16 (8.2% cloud,
  3 days after the 13 Aug anoxia peak).
- Baseline: 2018-06-27 (1.9%) is near-baseline (O2 still normal on 06-19/06-25); consider also an
  April or May scene for a cleaner pre-bloom reference.
- The embedding time series across all 13 summer scenes is checked against this two-pulse in-situ
  trace (Spearman vs O2 and chlorophyll).
