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
- Clay v1.5 encoder API and normalization constants (Task 8 spike).
- REPHY CSV schema and Thau station labels (Task 7 spike).
- Atmospheric correction: using L2A (land-tuned) as a known limitation over water.
