import pandas as pd
from malaigue import ingest


class _Item:
    def __init__(self, id, date, cloud):
        self.id = id
        self.properties = {"datetime": date, "eo:cloud_cover": cloud}


def test_list_clear_dates_filters_and_sorts(monkeypatch):
    fake = [
        _Item("S2_b", "2018-07-06T10:00:00Z", 5.0),
        _Item("S2_a", "2018-07-01T10:00:00Z", 12.0),
        _Item("S2_cloudy", "2018-07-03T10:00:00Z", 80.0),
    ]
    monkeypatch.setattr(
        ingest, "_search_items",
        lambda bbox, start, end, max_cloud: [i for i in fake if i.properties["eo:cloud_cover"] < max_cloud],
    )
    df = ingest.list_clear_dates((3.52, 43.35, 3.73, 43.47), "2018-07-01", "2018-07-10", max_cloud=20)
    assert list(df["item_id"]) == ["S2_a", "S2_b"]  # sorted by date ascending
    assert df["cloud"].max() < 20


import os
import pytest


@pytest.mark.skipif(os.environ.get("MALAIGUE_LIVE") != "1",
                    reason="set MALAIGUE_LIVE=1 to hit the live STAC API")
def test_load_scene_live_small():
    from malaigue import geo
    bbox = geo.aoi_bbox_4326()
    df = ingest.list_clear_dates(bbox, "2018-07-01", "2018-07-12", max_cloud=30)
    assert len(df) > 0
    ds = ingest.load_scene(df["item_id"].iloc[0], bbox, bands=["B04", "B05"])
    assert "B04" in ds and "B05" in ds
    assert ds.rio.crs is not None
    assert ds["B04"].shape[0] > 10
    # force a tiny compute to confirm real pixels download
    assert float(ds["B04"].isel(x=slice(0, 2), y=slice(0, 2)).mean()) >= 0
