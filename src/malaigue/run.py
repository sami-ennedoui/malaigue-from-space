"""End-to-end Thau malaigue experiment for summer 2018.

Builds a lagoon-level Clay embedding time series across the season, compares it to the
NDCI physical index and the REPHY in-situ record, maps the embedding change at ~80 m over
the Bouzigues crisis sector, and writes an honest verdict to docs/evaluation.md.
"""
import datetime as dt
import os
import pickle

import geopandas as gpd
import numpy as np

from malaigue import analyze, embed, geo, index, ingest, report, rephy, validate

CKPT = "data/clay/v1.5/clay-v1.5.ckpt"
META = "data/clay/metadata.yaml"
REPHY_CSV = "data/rephy/REPHY_Med_1987-2022.csv"
CACHE = "data/cache/embeddings.pkl"
LATLON = (43.42, 3.62)
SEASON = ("2018-04-01", "2018-09-30")
BASELINE = (dt.date(2018, 4, 1), dt.date(2018, 6, 10))
CRISIS_WINDOW = ("2018-06-25", "2018-08-20")


def _scene(item_id, bbox):
    ds = ingest.load_scene(item_id, bbox)
    if ds.rio.crs.to_string() != geo.SCENE_CRS:
        ds = ds.rio.reproject(geo.SCENE_CRS)
    return ds


def _lagoon_mask(ds, lagoon_utm):
    return geo.rasterize_mask(lagoon_utm, ds.rio.transform(),
                              (ds.rio.height, ds.rio.width), ds.rio.crs)


def _crop_chip(stack, transform, point_xy, size=embed.CHIP_PX):
    col = int(round((point_xy[0] - transform.c) / transform.a))
    row = int(round((point_xy[1] - transform.f) / transform.e))
    _, h, w = stack.shape
    top = min(max(row - size // 2, 0), max(h - size, 0))
    left = min(max(col - size // 2, 0), max(w - size, 0))
    return stack[:, top:top + size, left:left + size]


def spatial_map(model, bbox, lagoon_utm, point_xy, crisis_id, base_id, crisis_date, base_date):
    """Patch-level (~80 m) embedding change at a 2.56 km window centered on point_xy."""
    def chip(item_id):
        ds = _scene(item_id, bbox)
        mask = _lagoon_mask(ds, lagoon_utm)
        tr = ds.rio.transform()
        stack = np.stack([ds[b].values for b in ingest.CLAY_S2_BANDS]).astype("float32")
        sub = _crop_chip(stack, tr, point_xy)
        msub = _crop_chip(mask[None].astype("float32"), tr, point_xy)[0] > 0.5
        ndci_sub = _crop_chip(index.apply_water_mask(index.ndci(ds), mask).values[None], tr, point_xy)[0]
        return sub, msub, ndci_sub
    sub_c, msub_c, ndci_sub_c = chip(crisis_id)
    sub_b, _, _ = chip(base_id)
    pe_c = embed.patch_embeddings(model, sub_c, crisis_date, LATLON)
    pe_b = embed.patch_embeddings(model, sub_b, base_date, LATLON)
    a = analyze.spatial_change(pe_c, pe_b)
    water_grid = embed._downsample_mask(msub_c, a.shape)
    a = np.where(water_grid, a, np.nan)  # focus on lagoon water
    ndci_grid = analyze.block_mean(ndci_sub_c, a.shape[0], a.shape[1])
    return {"anom": a, "ndci_crop": ndci_sub_c, "ndci_grid": ndci_grid}


def _verdict(temporal, rho_o2, rho_chl, iou):
    signals = []
    if temporal["peak_in_window"]:
        signals.append("anomaly peaks within the crisis window")
    if rho_o2.get("rho") is not None and not np.isnan(rho_o2["rho"]) and rho_o2["rho"] < -0.3:
        signals.append(f"anti-correlates with oxygen (rho={rho_o2['rho']:.2f})")
    if rho_chl.get("rho") is not None and not np.isnan(rho_chl["rho"]) and rho_chl["rho"] > 0.3:
        signals.append(f"tracks chlorophyll (rho={rho_chl['rho']:.2f})")
    if iou is not None and not np.isnan(iou) and iou > 0.2:
        signals.append(f"hotspots overlap NDCI (IoU={iou:.2f})")
    if len(signals) >= 2:
        return "POSITIVE: Clay embeddings carry the malaigue signal (" + "; ".join(signals) + ")"
    return ("NEGATIVE: Clay embeddings do not clearly track the malaigue; the physical index and "
            "in-situ remain the reliable detectors. Signals: " + ("; ".join(signals) if signals else "none"))


def main():
    os.makedirs("outputs/figures", exist_ok=True)
    bbox = geo.aoi_bbox_4326()
    model = embed.load_clay(CKPT, META)
    lagoon_utm = geo.reproject_gdf(
        gpd.GeoDataFrame(geometry=[geo.lagoon_polygon_4326()], crs="EPSG:4326"),
        geo.SCENE_CRS).geometry.iloc[0]
    sectors_utm = geo.reproject_gdf(geo.sectors_4326(), geo.SCENE_CRS)
    bz = sectors_utm[sectors_utm["name"] == "bouzigues"].geometry.iloc[0].centroid
    bouzigues_xy = (bz.x, bz.y)

    # Embedding the season is the slow part; cache it (set MALAIGUE_CACHE=1) so the
    # downstream analysis can be iterated without re-embedding every scene.
    use_cache = os.environ.get("MALAIGUE_CACHE") == "1"
    if use_cache and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            dates, emb_by_date, ndci_mean, item_by_date = pickle.load(f)
        print(f"loaded cached embeddings for {len(emb_by_date)} dates")
    else:
        dates = ingest.list_clear_dates(bbox, *SEASON, max_cloud=20, tile="31TEJ")
        print("clear scenes:\n", dates.to_string(index=False))
        emb_by_date, ndci_mean, item_by_date = {}, {}, {}
        for _, r in dates.iterrows():
            ds = _scene(r["item_id"], bbox)
            mask = _lagoon_mask(ds, lagoon_utm)
            emb_by_date[r["date"]] = embed.lagoon_embedding(model, ds, mask, str(r["date"]), LATLON)
            nd = index.apply_water_mask(index.ndci(ds), mask)
            ndci_mean[r["date"]] = float(np.nanmean(nd.values))
            item_by_date[r["date"]] = r["item_id"]
            print(f"  {r['date']}  embedded | lagoon mean NDCI={ndci_mean[r['date']]:.3f}")
        if use_cache:
            os.makedirs("data/cache", exist_ok=True)
            with open(CACHE, "wb") as f:
                pickle.dump((dates, emb_by_date, ndci_mean, item_by_date), f)

    baseline_dates = sorted(d for d in emb_by_date if BASELINE[0] <= d <= BASELINE[1])
    if not baseline_dates:
        baseline_dates = [min(emb_by_date)]
    anom = analyze.anomaly_timeseries(emb_by_date, baseline_dates)
    anom["ndci"] = anom["date"].map(ndci_mean)

    insitu = rephy.thau_series(REPHY_CSV, params=["Chlorophylle a", "Oxygène dissous"],
                               start="2018-01-01", end="2018-12-31")

    temporal = validate.temporal_alignment(anom, CRISIS_WINDOW)
    rho_o2 = validate.spearman_vs_insitu(anom, insitu, "Oxygène dissous")
    rho_chl = validate.spearman_vs_insitu(anom, insitu, "Chlorophylle a")

    cw0, cw1 = dt.date(2018, 6, 25), dt.date(2018, 8, 20)
    crisis_date = anom[anom["date"].between(cw0, cw1)].sort_values("distance").iloc[-1]["date"]
    base_date = baseline_dates[0]
    sp = spatial_map(model, bbox, lagoon_utm, bouzigues_xy,
                     item_by_date[crisis_date], item_by_date[base_date],
                     str(crisis_date), str(base_date))
    iou = validate.spatial_overlap(sp["anom"], sp["ndci_grid"], top_q=0.9)

    report.plot_timeseries(anom, insitu, "outputs/figures/timeseries.png")
    report.plot_anomaly_map(sp["anom"], "outputs/figures/anomaly_map.png",
                            title=f"Clay embedding change, Bouzigues, {crisis_date} vs {base_date}")
    report.plot_index_map(sp["ndci_crop"], "outputs/figures/ndci_crisis.png",
                          title=f"NDCI, Bouzigues, {crisis_date}")

    verdict = _verdict(temporal, rho_o2, rho_chl, iou)
    metrics = {
        "n_scenes": int(len(dates)),
        "baseline_dates": [str(d) for d in baseline_dates],
        "crisis_date_used": str(crisis_date),
        "temporal_peak": temporal,
        "spearman_anom_vs_O2": rho_o2,
        "spearman_anom_vs_chl": rho_chl,
        "spatial_iou_anom_vs_ndci": iou,
        "verdict": verdict,
    }
    report.write_evaluation(metrics, "docs/evaluation.md")
    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
