"""Downstream analysis of Clay embeddings."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def _cosine_distance(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
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
    """Per-patch cosine distance between two (Hp, Wp, D) embedding grids."""
    a = patch_crisis.reshape(-1, patch_crisis.shape[-1])
    b = patch_baseline.reshape(-1, patch_baseline.shape[-1])
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den[den == 0] = 1.0
    dist = 1.0 - num / den
    return dist.reshape(patch_crisis.shape[:2])


def cluster_patches(patches, k, seed=0):
    """KMeans labels over a (Hp, Wp, D) grid, returned as (Hp, Wp)."""
    h, w, d = patches.shape
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(
        patches.reshape(-1, d))
    return labels.reshape(h, w)


def block_mean(a, ny, nx):
    """Reduce 2D array a to shape (ny, nx) by averaging non-overlapping blocks,
    ignoring NaNs. Aligns a full-resolution index map onto the patch grid."""
    h, w = a.shape
    fh, fw = h // ny, w // nx
    a = a[: ny * fh, : nx * fw]
    return np.nanmean(a.reshape(ny, fh, nx, fw), axis=(1, 3))
