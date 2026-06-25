# Can Clay's frozen embeddings see the 2018 Thau malaïgue?

## Summary

In August 2018 a malaïgue, a severe anoxic crisis, killed most of the shellfish in the étang de
Thau on the French Mediterranean coast. This report tests whether the Clay v1.5 geospatial
foundation model, run on Sentinel-2 imagery as a frozen encoder with no fine-tuning, produces
embeddings that respond to the event. Across fifteen cloud-free scenes from April to September
2018 the lagoon embedding shows no response that tracks the crisis. Its largest change from the
spring baseline is a cosine distance of 0.059, it does not correlate with the in-situ oxygen or
chlorophyll records, and its spatial pattern does not match a chlorophyll index. The one
measurement that marks the crisis is the in-situ dissolved oxygen, which fell to 0.04 mg/L at
Bouzigues on 13 August. The chlorophyll index and the embeddings are both weak at the lagoon
scale, and the embeddings carry no usable signal at all. The finding is specific to this frozen,
zero-shot, lagoon-averaged setup, not a statement about the model in general.

## Background

The étang de Thau is a shallow lagoon on the Mediterranean coast, used for oyster and mussel
farming. A malaïgue is a summer anoxic crisis: under high temperature and low wind the water
stratifies, oxygen near the bottom is used up, and sulfur bacteria can turn the water milky
white. The 2018 event was severe and is documented by date and by sector, which makes it a clean
target to test against.

Geospatial foundation models such as Clay are trained on large volumes of satellite imagery
without labels, and are meant to produce general embeddings that transfer to many tasks. Most
published evaluations measure that transfer on land cover or segmentation. A water-quality crisis
is a different kind of target, short-lived, localized, and far from the usual training
distribution. The question here is whether the embeddings carry any signal for it without being
trained to.

## Data

- Sentinel-2 L2A surface reflectance from the Element84 Earth Search archive. Fifteen scenes over
  tile T31TEJ under 20 percent cloud, April to September 2018.
- Clay v1.5, run as a frozen encoder on CPU. No weights are updated.
- NDCI, the normalized difference chlorophyll index, computed from the red and red-edge bands as
  a physical reference for chlorophyll.
- REPHY, the IFREMER in-situ monitoring record for Thau, giving dissolved oxygen and chlorophyll a
  at the Bouzigues station.

## Method

Each scene is reduced to one lagoon embedding by masking the imagery to the lagoon and passing it
through the frozen encoder. The seasonal anomaly of a scene is the cosine distance between its
embedding and a spring baseline, taken as the mean of the April and May scenes. The embedding
signal is then compared with three independent references.

1. Timing. Whether the largest anomaly of the season falls inside the documented crisis window,
   late June to late August.
2. Correlation. The Spearman correlation between the embedding anomaly and the in-situ oxygen and
   chlorophyll series, each matched to the nearest scene within seven days.
3. Space. Over the Bouzigues sector, the overlap between the patch-level embedding change and the
   NDCI, measured as the intersection over union of their top deciles.

A genuine detector would peak during the crisis, rise as oxygen falls, and overlap the
chlorophyll index in space. The raw metrics are in [../outputs/metrics.md](../outputs/metrics.md).

## Results

| Reference | Metric | Value |
| --- | --- | --- |
| Embedding against the spring baseline | largest cosine distance | 0.059, on 17 July |
| Embedding anomaly against dissolved oxygen | Spearman rho, n = 13 | 0.21, p = 0.49 |
| Embedding anomaly against chlorophyll a | Spearman rho, n = 15 | 0.05, p = 0.87 |
| Embedding change against NDCI, Bouzigues | intersection over union | 0.025 |

The embedding anomaly stays small, never above 0.06 in cosine distance, and varies from scene to
scene with no pattern that matches the crisis. Its largest value falls on 17 July, inside the
crisis window, but at this magnitude and against the scatter of the other scenes the timing
carries no weight. Neither correlation is distinguishable from zero, and the oxygen value is
slightly positive, the opposite of what a detector would show. The spatial overlap at Bouzigues
is 0.025, effectively none.

![Embedding anomaly through 2018 against the REPHY oxygen and chlorophyll records](../outputs/figures/timeseries.png)

The embedding anomaly, in red on the left axis, is small and scattered, with no rise that lines up
with the summer crisis. Dissolved oxygen falls through the summer to its August low. The
chlorophyll peak in late autumn is a separate bloom, outside the malaïgue window.

![Patch-level embedding change at Bouzigues, 17 July against the spring baseline](../outputs/figures/anomaly_map.png)

![NDCI at Bouzigues on 17 July](../outputs/figures/ndci_crisis.png)

At the crisis sector the embedding change is the broad seasonal shift, roughly uniform near 0.2
over open water and lower along the shore. The NDCI on the same day is uniformly low across the
lagoon, with raised values only at the shoreline edge. The two do not line up.

## Discussion

The result is a property of the model on this signal, not an artifact of the data. The crisis is
real and independently dated: the in-situ oxygen at Bouzigues fell to 0.04 mg/L on 13 August,
near-total anoxia. The embeddings are not inert either. Their patch-level change from April to
July is about 0.2, an order of magnitude larger than the within-season anomaly, and it is the
broad seasonal shift in illumination and water state. So the encoder moves with the season but
not with the crisis.

The chlorophyll index does not mark the crisis at the lagoon scale either. The lagoon-mean NDCI
stays between 0.04 and 0.10 all summer and is in fact highest on 18 May, before the event, with
no July or August rise. The detector that works is the in-situ oxygen, a point measurement near
the bottom. This shapes how the negative should be read. It is not a strong optical index beating
a weak model. It is that an optical, surface-reflectance view, whether a hand-built index or a
learned embedding, misses a crisis whose clean signature is subsurface oxygen.

This is in line with what systematic benchmarks report. Broad evaluations of geospatial
foundation models such as [PANGAEA](https://arxiv.org/abs/2412.04204) find that they do not
consistently beat task-specific baselines, and frozen embeddings in particular have shown limited
sensitivity to time. The test here is a single-event probe in that spirit, not a benchmark.

## Limitations

Each limitation is also a concrete next step.

- Averaging. One embedding stands for the whole lagoon, which dilutes a crisis confined to a few
  sectors and a few days. A per-sector or per-pixel embedding would test this.
- Index choice. NDCI tracks chlorophyll, while the white-water phase of a malaïgue is closer to a
  turbidity and high-reflectance signal. A turbidity index might capture more of it.
- Timing. The white water was reported around 5 July, and the nearest cloud-free scenes were 2
  and 7 July. A short surface event can fall between revisits.
- Atmospheric correction. Sentinel-2 L2A is corrected for land. Over water, a dedicated correction
  such as C2RCC or ACOLITE is more faithful.
- Frozen and zero-shot. The encoder is never trained on this task. A linear probe or light
  fine-tuning on a few labeled water states is the obvious next experiment.

## Conclusion

Run frozen and zero-shot, Clay v1.5's Sentinel-2 embeddings do not detect the 2018 Thau malaïgue
at the lagoon scale. The crisis is captured by in-situ oxygen, only weakly reflected by a
chlorophyll index, and absent from the embeddings. Whether a probe trained on these embeddings
would do better is the natural follow-up.
