# Evaluation: does Clay see the 2018 Thau malaïgue?

This is the written verdict. The raw metrics from the latest run are in `outputs/metrics.md`,
and the figures are in `outputs/figures/`.

## The question

In the summer of 2018 the étang de Thau suffered a severe malaïgue, an anoxic crisis that
killed the entire mussel stock and a large share of the oysters. The question here is narrow:
does a generic Earth observation foundation model, Clay v1.5, used frozen and with
no task specific training, produce Sentinel-2 embeddings that flag this crisis. The embedding
signal is judged against a physical chlorophyll index, NDCI, and against the IFREMER REPHY
in-situ record.

## Verdict

No. Clay's embeddings do not track the malaïgue. The clean detector of this crisis is the in-situ
oxygen, which crashed to near zero. The optical signals, both the chlorophyll index and the
embeddings, are muted at the lagoon scale, and the embeddings most of all. The reasoning is below.

## The numbers

Across 15 clear Sentinel-2 scenes over tile T31TEJ, from April to September 2018, with the
spring scenes as the baseline:

- Temporal. The lagoon level embedding barely moves all season. Its largest cosine distance
  from the spring baseline is 0.059, and that maximum falls on 17 July, inside the documented
  crisis window. The magnitude is negligible, so the in window timing is weak evidence at best.
- Correlation. The embedding anomaly does not correlate with the in-situ series. Against
  dissolved oxygen, Spearman rho is 0.21 with p = 0.49, and the sign is even wrong, since a real
  detector would rise as oxygen falls. Against chlorophyll, rho is 0.05 with p = 0.87. Both are
  statistically nothing.
- Spatial. Over the Bouzigues sector, where the in-situ oxygen later crashed to 0.04, the
  embedding change hotspots and the NDCI hotspots overlap with an intersection over union of
  0.025, essentially zero.

## Reading the figures

- `timeseries.png`. The embedding anomaly is small and noisy with no crisis structure. The
  in-situ oxygen dips through the summer, and the large chlorophyll spike is a separate autumn
  bloom in November, not the July to August malaïgue.
- `anomaly_map.png`. The April to July embedding change at Bouzigues is real but smooth and
  roughly uniform across the water, around 0.2. That is seasonal and illumination change, not a
  localized crisis signature.
- `ndci_crisis.png`. On 17 July the lagoon water is uniformly low chlorophyll with only shoreline
  edge effects, and no bloom hotspot in the open water.

## Why the result holds

The cross-checks did their job. The REPHY in-situ data independently proves the crisis was
real, since dissolved oxygen at Bouzigues fell to 0.04 in mid August. So the embeddings staying
flat is a fact about the model, not about missing or bad data. And the embeddings are not dead:
they clearly respond to the season, since the April to July change is about 0.2. They simply do
not respond to the water quality crisis. The crisis is anchored by a dated event and an independent
in-situ record, so Clay staying flat through it is not a coincidence. The chlorophyll index is
itself muted at the lagoon scale, so this is less a strong index beating the model and more that
the model carries no usable signal here while the in-situ does.

## Limitations

These are the reasons the negative should be read as "not with this setup" rather than a final
word, and they are the natural next experiments.

- Averaging. The lagoon level embedding averages over the whole lagoon, which dilutes a crisis
  that is localized to specific zones and short windows. The in-situ is a point measurement at a
  station.
- The index choice. NDCI is a chlorophyll proxy. The dramatic phase of a malaïgue is the milky
  white water of sulfur bacteria, which is a high reflectance and turbidity signal more than a
  chlorophyll one, so a turbidity index might capture it better.
- Timing. The white water was confirmed around 5 July, and the nearest clear acquisitions were
  2 July at 11 percent cloud and 7 July at under 1 percent. A brief surface event can fall
  between revisits.
- Atmospheric correction. Sentinel-2 L2A is tuned for land. Over water a dedicated correction
  such as C2RCC or ACOLITE would be more faithful.
- Frozen and zero shot. Clay is used only as a feature extractor. Light fine tuning, or a small
  probe trained on a few labeled water states, could change the picture.

## Scope

A reproducible Sentinel-2 and geopandas pipeline that tests one question about a foundation
model and reports a negative result for this setup. Clay is used only as a frozen feature
extractor, so this is not a claim about what foundation models can or cannot do in general.
