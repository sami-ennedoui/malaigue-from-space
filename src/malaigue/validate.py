"""Agreement metrics tying embeddings to the three anti-fluke anchors."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def temporal_alignment(anom_df, crisis_window):
    """Does the anomaly peak fall inside the documented crisis window."""
    start = pd.to_datetime(crisis_window[0]).date()
    end = pd.to_datetime(crisis_window[1]).date()
    peak = anom_df.sort_values("distance").iloc[-1]
    pd_date = peak["date"]
    return {
        "peak_date": pd_date,
        "peak_distance": float(peak["distance"]),
        "peak_in_window": bool(start <= pd_date <= end),
    }


def spearman_vs_insitu(anom_df, rephy_df, param):
    """Spearman correlation between the embedding anomaly and an in-situ parameter,
    matched on nearest date within 7 days."""
    ins = rephy_df[rephy_df["param"] == param].copy()
    ins["date"] = pd.to_datetime(ins["date"])
    a = anom_df.copy()
    a["date"] = pd.to_datetime(a["date"])
    merged = pd.merge_asof(
        a.sort_values("date"), ins.sort_values("date"),
        on="date", direction="nearest", tolerance=pd.Timedelta("7D"),
    ).dropna(subset=["value", "distance"])
    if len(merged) < 3:
        return {"rho": np.nan, "n": int(len(merged))}
    rho, p = spearmanr(merged["distance"], merged["value"])
    return {"rho": float(rho), "p": float(p), "n": int(len(merged))}


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
    if a.size == 0:
        return {row["name"]: np.nan for _, row in sectors_gdf.iterrows()}
    hot = anom_raster >= np.quantile(a, top_q)
    total = hot.sum()
    out = {}
    for _, row in sectors_gdf.iterrows():
        m = geo.rasterize_mask(row.geometry, transform, anom_raster.shape, crs)
        out[row["name"]] = float(np.logical_and(hot, m).sum() / total) if total else np.nan
    return out
