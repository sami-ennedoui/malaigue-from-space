# malaigue-from-space

[![tests](https://github.com/sami-ennedoui/malaigue-from-space/actions/workflows/ci.yml/badge.svg)](https://github.com/sami-ennedoui/malaigue-from-space/actions/workflows/ci.yml)

Testing whether the Clay v1.5 foundation model reacts to the 2018 étang de Thau malaïgue in its
Sentinel-2 embeddings, with no task-specific training. The result is negative.

A malaïgue is an anoxic crisis in the Thau lagoon, on the French Mediterranean coast. The summer
2018 event killed the lagoon's entire mussel stock and a large share of the oysters. The question
here is whether a generic Earth observation model, used as a frozen feature extractor, produces
embeddings that react to such an event. The embedding signal is measured against a physical
chlorophyll index, NDCI, and the IFREMER REPHY in-situ record.

Across the 2018 season the lagoon embedding barely moves, it does not correlate with in-situ oxygen
or chlorophyll, and its spatial pattern does not match the chlorophyll index. The signal that tracks
the crisis is the in-situ dissolved oxygen, which fell to near zero in mid August. At the lagoon
scale the chlorophyll index and the embeddings are both flat, the embeddings most of all. The full
write-up, with numbers and figures, is in [docs/evaluation.md](docs/evaluation.md).

## Results

Across 15 clear Sentinel-2 scenes over tile T31TEJ in 2018:

- The largest embedding deviation from the spring baseline is a cosine distance of 0.059, which is
  negligible.
- The embedding anomaly does not correlate with the in-situ series. Against dissolved oxygen,
  Spearman 0.21 at p = 0.49. Against chlorophyll, 0.05 at p = 0.87.
- The spatial overlap between the embedding hotspots and the chlorophyll index at Bouzigues is 0.025.

![Embedding anomaly through summer 2018, with REPHY oxygen and chlorophyll](outputs/figures/timeseries.png)

The embedding anomaly stays small and flat through the crisis, while the in-situ series confirm the
event was real.

![Clay embedding change at Bouzigues, 17 July against the spring baseline](outputs/figures/anomaly_map.png)

![NDCI chlorophyll index at Bouzigues on 17 July](outputs/figures/ndci_crisis.png)

The embedding change at Bouzigues is smooth and seasonal rather than a localized crisis signature,
and it does not line up with the chlorophyll index beside it.

## Validation

The embedding signal is checked against three independent references: the documented event with its
date and affected sectors, the REPHY in-situ record, and the in-image NDCI index. Dissolved oxygen
at Bouzigues fell to 0.04 mg/L in mid August, which places the crisis in time without using the
imagery. The embeddings stay flat across that window while still responding to the change of season,
where the April to July change is about 0.2. The flat response is a property of the model on this
signal rather than missing or bad data.

## How it works

The pipeline is a set of small modules wired together by a run script.

- `ingest` queries the Element84 Earth Search STAC API for Sentinel-2 L2A scenes over Thau and
  loads the bands clipped to the lagoon.
- `geo` holds the geospatial core: the lagoon and sector polygons, CRS reprojection, tiling, and the
  water mask, with geopandas.
- `index` computes NDCI, a chlorophyll proxy, as the physical reference.
- `rephy` loads the IFREMER REPHY in-situ series for Thau.
- `embed` loads Clay v1.5 and runs it on CPU as a frozen feature extractor, returning per-chip and
  per-patch embeddings.
- `analyze` turns embeddings into a seasonal anomaly time series and a spatial change map.
- `validate` scores the agreement between the embeddings and the three references.
- `report` draws the figures and writes the metrics.

## Data

- Sentinel-2 L2A surface reflectance from Element84 Earth Search, free and open.
- Clay v1.5 weights from the Clay foundation on Hugging Face.
- The REPHY in-situ dataset from IFREMER, published on SEANOE under an open licence.

None of these are stored in the repository. The pipeline fetches the imagery on demand, and the
model and the in-situ extract are downloaded once. The exact sources are listed in
[docs/decisions.md](docs/decisions.md).

## Reproducing

The project uses uv and runs on CPU. It pins Python 3.12, since the model stack does not yet ship
wheels for newer versions.

```
uv sync
```

Download the Clay v1.5 checkpoint and its band metadata, and the REPHY Mediterranean extract, into
`data/`. The sources are in `docs/decisions.md`. Then run the experiment:

```
uv run python -m malaigue.run
```

It lists the clear 2018 scenes, embeds the lagoon for each, builds the anomaly trace and the
Bouzigues map, and writes the figures to `outputs/figures/` and the metrics to `outputs/metrics.md`.
The tests run with:

```
uv run pytest
```

## Limitations and next steps

- The lagoon-wide average dilutes a crisis that is localized to specific zones and short windows.
- NDCI is a chlorophyll proxy, while the white-water phase of a malaïgue is closer to a turbidity
  signal, so a turbidity index may capture it better.
- A brief surface event can fall between cloud-free revisits.
- Sentinel-2 L2A atmospheric correction is tuned for land rather than water, where a dedicated
  correction such as C2RCC or ACOLITE would be more faithful.
- Clay is used frozen and zero-shot. Light fine-tuning, or a small probe trained on a few labelled
  water states, could change the result.

## Scope

Clay is used only as a frozen feature extractor, not trained or fine-tuned. This tests whether its
embeddings carry a water-quality signal in this setup. It is not a benchmark of the model or a
statement about foundation models in general.
