import datetime as dt

import numpy as np
import pandas as pd

from malaigue import validate


def test_temporal_alignment_peak_in_window():
    df = pd.DataFrame({
        "date": [dt.date(2018, 5, 1), dt.date(2018, 7, 5)],
        "distance": [0.0, 0.8],
    })
    out = validate.temporal_alignment(df, ("2018-06-25", "2018-07-10"))
    assert out["peak_date"] == dt.date(2018, 7, 5)
    assert out["peak_in_window"] is True


def test_spatial_overlap_identical_top_quantile():
    a = np.arange(16, dtype="float32").reshape(4, 4)
    iou = validate.spatial_overlap(a, a.copy(), top_q=0.75)
    assert iou == 1.0


def test_spearman_vs_insitu_monotonic():
    anom = pd.DataFrame({
        "date": pd.to_datetime(["2018-06-01", "2018-07-01", "2018-08-01"]),
        "distance": [0.1, 0.5, 0.9],
    })
    rephy = pd.DataFrame({
        "date": pd.to_datetime(["2018-06-02", "2018-07-02", "2018-08-02"]),
        "param": ["Chlorophylle a"] * 3,
        "value": [1.0, 5.0, 9.0],
    })
    out = validate.spearman_vs_insitu(anom, rephy, "Chlorophylle a")
    assert out["n"] == 3
    assert out["rho"] > 0.9
