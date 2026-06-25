# malaigue-from-space

[![tests](https://github.com/sami-ennedoui/malaigue-from-space/actions/workflows/ci.yml/badge.svg)](https://github.com/sami-ennedoui/malaigue-from-space/actions/workflows/ci.yml)

Testing whether the Clay foundation model can flag the 2018 étang de Thau malaïgue from
Sentinel-2 embeddings.

A malaïgue is an anoxic crisis in the Thau lagoon, on the French Mediterranean coast. In the
summer of 2018 a severe one killed the entire mussel stock and a large share of the oysters.
This project asks one question and answers it honestly: can a generic Earth observation
foundation model, Clay v1.5, flag that crisis from free Sentinel-2 imagery, with no training of
its own.

The short answer is no. The embeddings barely move during the crisis, they do not correlate with
the in-situ oxygen or chlorophyll, and their spatial pattern does not match a physical chlorophyll
index. The clean detector of this crisis is the in-situ oxygen, which crashed to near zero in mid
August. The chlorophyll index and the embeddings are both muted at the lagoon scale, the embeddings
most of all. The full reasoning, with numbers and figures, is in
[docs/evaluation.md](docs/evaluation.md).

## Results

Across 15 clear Sentinel-2 scenes over tile T31TEJ in 2018, the lagoon embedding barely moves, it
does not correlate with the in-situ measurements, and its spatial pattern does not match the
chlorophyll index:

- The largest embedding deviation from the spring baseline is a cosine distance of 0.059, which is
  negligible.
- Embedding against in-situ oxygen, Spearman 0.21 with p = 0.49. Against chlorophyll, 0.05 with
  p = 0.87. Both are statistically nothing.
- The spatial overlap of the embedding hotspots and the chlorophyll index at Bouzigues is 0.025.

![Embedding anomaly through summer 2018, with REPHY oxygen and chlorophyll](outputs/figures/timeseries.png)

The red embedding anomaly stays small and flat through the crisis. The in-situ series prove the
crisis was real, which is what makes the negative trustworthy.

![Clay embedding change at Bouzigues, 17 July against the spring baseline](outputs/figures/anomaly_map.png)

![NDCI chlorophyll index at Bouzigues on 17 July](outputs/figures/ndci_crisis.png)

The embedding change at Bouzigues is smooth and seasonal rather than a localized crisis signature,
and it does not line up with the chlorophyll index next to it.

## Why a negative result is the point

The pipeline is built so the answer is trustworthy whichever way it falls. A positive result
would be believed only if it agreed with three independent anchors at once: the documented event
with its date and affected sectors, the IFREMER REPHY in-situ record, and an in-image spectral
index. The negative is trustworthy for the same reason. The in-situ data proves the crisis was
real, since dissolved oxygen at Bouzigues fell to 0.04 in mid August, and the embeddings still do
not react. They are not broken either, since they clearly track the change of season. They are
simply insensitive to this water quality signal.

## How it works

The pipeline is seven small modules and a script that wires them together.

- `ingest` queries the Element84 Earth Search STAC API for Sentinel-2 L2A scenes over Thau and
  loads the bands clipped to the lagoon.
- `geo` is the geospatial core: the lagoon and sector polygons, CRS reprojection, tiling, and the
  water mask, all with geopandas.
- `index` computes NDCI, a chlorophyll proxy, as the physical control.
- `rephy` loads the IFREMER REPHY in-situ series for Thau.
- `embed` loads Clay v1.5 and runs it on CPU as a frozen feature extractor, returning per chip and
  per patch embeddings.
- `analyze` turns embeddings into a seasonal anomaly time series and a spatial change map.
- `validate` scores the agreement between the embeddings and the three anchors.
- `report` draws the figures and writes the metrics.

## Data

- Sentinel-2 L2A surface reflectance from Element84 Earth Search, free and open.
- The Clay v1.5 weights from the Clay foundation on Hugging Face.
- The REPHY in-situ dataset from IFREMER, published on SEANOE under an open licence.

None of these are stored in the repository. The pipeline fetches the imagery on demand, and you
download the model and the in-situ extract once. The exact sources are listed in
[docs/decisions.md](docs/decisions.md).

## Reproducing it

The project uses uv and runs on CPU, with no GPU needed. It pins Python 3.12, since the model
stack does not yet ship wheels for newer versions.

```
uv sync
```

Download the Clay v1.5 checkpoint and its band metadata, and the REPHY Mediterranean extract,
into `data/`. The sources are in `docs/decisions.md`. Then run the experiment:

```
uv run python -m malaigue.run
```

It lists the clear 2018 scenes, embeds the lagoon for each, builds the anomaly trace and the
Bouzigues map, and writes the figures to `outputs/figures/` and the metrics to
`outputs/metrics.md`. The tests run with:

```
uv run pytest
```

## Limitations and what comes next

The negative result is honest about its own scope. The lagoon wide average dilutes a localized
crisis. NDCI is a chlorophyll proxy, while the white water phase of a malaïgue is closer to a
turbidity signal. The brief surface peak can fall between cloud free revisits. The L2A
atmospheric correction is tuned for land rather than water. Clay is also used frozen and zero
shot. Fine tuning, a turbidity index, per zone analysis, and a water specific correction are the
natural next steps, and they are discussed in [docs/evaluation.md](docs/evaluation.md).

## Honest scope

Clay is used only as a frozen feature extractor, never trained or fine tuned here. This is an
exploration of whether its embeddings carry a water quality signal, not a benchmark of the model
or a claim about what foundation models can do in general.
