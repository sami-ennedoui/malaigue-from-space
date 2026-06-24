import datetime as dt

import numpy as np

from malaigue import analyze


def test_anomaly_timeseries_flags_outlier():
    base = np.array([1.0, 0.0, 0.0])
    emb = {
        dt.date(2018, 5, 1): base,
        dt.date(2018, 6, 1): base,
        dt.date(2018, 7, 5): np.array([0.0, 1.0, 0.0]),
    }
    df = analyze.anomaly_timeseries(emb, baseline_dates=[dt.date(2018, 5, 1), dt.date(2018, 6, 1)])
    peak = df.sort_values("distance").iloc[-1]
    assert peak["date"] == dt.date(2018, 7, 5)
    assert np.isclose(df[df.date == dt.date(2018, 5, 1)]["distance"].iloc[0], 0.0, atol=1e-6)


def test_spatial_change_is_zero_when_identical():
    a = np.random.rand(4, 4, 8).astype("float32")
    out = analyze.spatial_change(a, a)
    assert out.shape == (4, 4)
    assert np.allclose(out, 0.0, atol=1e-6)


def test_cluster_patches_shape():
    p = np.random.rand(4, 4, 5).astype("float32")
    lab = analyze.cluster_patches(p, k=2, seed=0)
    assert lab.shape == (4, 4)
    assert set(np.unique(lab)) <= {0, 1}


def test_block_mean_reduces_grid():
    a = np.arange(16, dtype="float32").reshape(4, 4)
    out = analyze.block_mean(a, 2, 2)
    assert out.shape == (2, 2)
    assert np.isclose(out[0, 0], np.mean([0, 1, 4, 5]))
